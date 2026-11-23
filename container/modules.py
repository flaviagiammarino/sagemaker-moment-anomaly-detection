"""
Adapted from MOMENT (https://github.com/moment-timeseries-foundation-model/moment),
Copyright (c) 2024 Auton Lab, Carnegie Mellon University, MIT License.
"""

import json
from pathlib import Path
from typing import Any
import torch
from safetensors.torch import load_file
from torch import nn
from transformers import T5Config, T5EncoderModel

# The released MOMENT-1 checkpoints were all pretrained on 512-step windows
# split into non-overlapping 8-step patches.
SEQ_LEN = 512
PATCH_LEN = 8
N_PATCHES = SEQ_LEN // PATCH_LEN  # 64 patches per window

# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class PositionalEmbedding(nn.Module):
    """
    Positional vectors added to each patch embedding. They are fixed sinusoids,
    not learned, and their values are loaded from the checkpoint: the zeros here
    are only a placeholder with the checkpoint's shape and key. The checkpoint
    table covers 5000 positions, but only uses the first N_PATCHES.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.register_buffer("pe", torch.zeros(1, max_len, d_model))

    def forward(self) -> torch.Tensor:
        # One vector per patch position: [N_PATCHES x d_model].
        return self.pe[0, :N_PATCHES]


class PatchEmbedding(nn.Module):
    """
    Turn a normalized sequence into encoder tokens. Each channel is cut into
    non-overlapping patches of `patch_len` steps; each patch is mapped to a
    `d_model` vector by a linear layer (no bias), and the positional vector
    for its place in the window is added.
    """

    def __init__(self, d_model: int, patch_len: int):
        super().__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        # Only used in pretraining to hide patches; kept so the checkpoints load with strict=True
        self.mask_embedding = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch_size x n_channels x seq_len] -> [batch_size x n_channels x n_patches x d_model]
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        # The same positional vector is added to every sample and channel
        return self.value_embedding(patches) + self.position_embedding()


class ReconstructionHead(nn.Module):
    """
    Map each encoder output back to the `patch_len` values of its patch with a
    linear layer, then join the patches into a sequence.
    """

    def __init__(self, d_model: int, patch_len: int):
        super().__init__()
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[batch_size x n_channels x n_patches x d_model] -> [batch_size x n_channels x seq_len]"""
        return self.linear(x).flatten(start_dim=2, end_dim=3)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MOMENT(nn.Module):
    """
    Pretrained MOMENT for inference (reconstruction only).

    Each window must have exactly SEQ_LEN time steps. Padding and missing values are
    not supported: every time step must be finite in every channel.
    """

    def __init__(self, t5_config: dict[str, Any]):
        super().__init__()
        self.d_model = t5_config["d_model"]
        self.patch_embedding = PatchEmbedding(self.d_model, PATCH_LEN)
        self.encoder = T5EncoderModel(T5Config.from_dict(t5_config)).get_encoder()
        self.head = ReconstructionHead(self.d_model, PATCH_LEN)

    @staticmethod
    def instance_statistics(
            x: torch.Tensor,
            eps: float = 1e-5
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mean and standard deviation of each sequence (per sample and channel), for
        reversible instance normalization (RevIN): the model normalizes its input
        with them and reverses the normalization on its output.

        Input:
            x : [batch_size x n_channels x seq_len]
        Output:
            mean : [batch_size x n_channels x 1]
            stdev : [batch_size x n_channels x 1]
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # no Bessel correction, as in pretraining

        return mean, var.sqrt() + eps

    @classmethod
    def from_pretrained(cls, weights_dir: str | Path) -> "MOMENT":
        """
        Load a released checkpoint directory (config.json + model.safetensors), frozen.
        """

        weights_dir = Path(weights_dir)
        config = json.loads((weights_dir / "config.json").read_text())
        model = cls(config["t5_config"])
        model.load_state_dict(load_file(weights_dir / "model.safetensors"), strict=True)
        model.requires_grad_(False)
        return model.eval()

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct a batch of sequences.

        Input:
            x : [batch_size x n_channels x seq_len]
        """

        # Check the window length (fixed by the pretrained checkpoints)
        batch_size, n_channels, seq_len = x.shape
        if seq_len != SEQ_LEN:
            raise ValueError(f"Expected windows of {SEQ_LEN} steps, got {seq_len}.")

        # Reversible instance normalization (RevIN): scale each sequence to mean 0
        # and standard deviation 1. The statistics are local to this call, so
        # concurrent calls on one model cannot mix them up.
        mean, stdev = self.instance_statistics(x)  # [batch_size x n_channels x 1]
        z = (x - mean) / stdev  # [batch_size x n_channels x seq_len]

        # Patch embedding: split the 512 time steps of each channel into 64 patches
        # of 8 steps, map each patch to a d_model vector with an 8 x d_model linear
        # layer (no bias), and add the positional encoding of its place in the window
        h = self.patch_embedding(z)  # [batch_size x n_channels x N_PATCHES x d_model]

        # Each channel is encoded as its own sequence: merge the batch and channel dimensions
        h = h.reshape(batch_size * n_channels, N_PATCHES, self.d_model)  # [(batch_size * n_channels) x N_PATCHES x d_model]

        # T5 encoder: a stack of self-attention and feed-forward blocks (24 for MOMENT-1-large)
        h = self.encoder(inputs_embeds=h).last_hidden_state  # [(batch_size * n_channels) x N_PATCHES x d_model]

        # Split the batch and channel dimensions again
        h = h.reshape(batch_size, n_channels, N_PATCHES, self.d_model)  # [batch_size x n_channels x N_PATCHES x d_model]

        # Reconstruction head: map each patch embedding back to its 8 values and join
        # the patches into a sequence
        x_ = self.head(h)  # [batch_size x n_channels x seq_len]

        # Reverse the normalization: back to the original units of each sequence
        x_ = x_ * stdev + mean  # [batch_size x n_channels x seq_len]

        return x_
