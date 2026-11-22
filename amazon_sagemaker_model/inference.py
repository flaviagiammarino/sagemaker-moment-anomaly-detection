import warnings
warnings.filterwarnings("ignore")

import io
import json
import pandas as pd
import torch
from modules import MOMENT, SEQ_LEN

# Extract the environment configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_data_parallel = torch.cuda.device_count() > 1
kwargs = {"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {}

@torch.inference_mode()
def reconstruct(
        model: MOMENT | torch.nn.DataParallel,
        series: torch.Tensor,
        stride: int = SEQ_LEN // 4,
        batch_size: int = 256,
) -> torch.Tensor:
    """
    Reconstruct a time series sequence by sequence. Where sequences overlap
    (stride < SEQ_LEN), each time step is the average of its reconstructions.

    Input:
        series : [n_channels x n_samples]
    Output:
        reconstruction : [n_channels x n_samples]
    """

    # The model's weights are float32: integer or float64 series would fail
    series = series.float()  # [n_channels x n_samples]

    # Missing or infinite values are not supported
    if not torch.isfinite(series).all():
        raise ValueError("Missing (NaN) or infinite values are not supported.")

    # The series must fill at least one full window
    n_channels, n_samples = series.shape  # series: [n_channels x n_samples]
    if n_samples < SEQ_LEN:
        raise ValueError(f"Series must have at least {SEQ_LEN} time steps, got {n_samples}.")

    # Start of each window: one every `stride` steps, and the last window ends at the end
    # of the series, so no window needs padding
    starts = list(range(0, n_samples - SEQ_LEN, stride)) + [n_samples - SEQ_LEN]

    # Cut the series into windows of SEQ_LEN steps
    x = torch.stack([series[:, s: s + SEQ_LEN] for s in starts])  # [n_windows x n_channels x SEQ_LEN]

    # Feed the windows to the model in batches
    dataloader = torch.utils.data.DataLoader(
        dataset=x,
        batch_size=batch_size,
        shuffle=False,
        **kwargs
    )

    # Reconstruct each batch on the device and collect the reconstructions on the CPU
    x_ = torch.cat([model(x.to(device)).cpu() for x in dataloader])  # [n_windows x n_channels x SEQ_LEN]

    # Average the reconstructions of each time step over the windows that contain it
    total = torch.zeros(n_channels, n_samples)  # sum of the reconstructions of each time step
    count = torch.zeros(n_channels, n_samples)  # number of windows that contain each time step
    for s, x_k in zip(starts, x_):  # x_k: [n_channels x SEQ_LEN]
        total[:, s: s + SEQ_LEN] += x_k
        count[:, s: s + SEQ_LEN] += 1
    x_ = total / count  # [n_channels x n_samples]

    return x_

def model_fn(model_dir):
    '''
    Load the model: model_dir contains the checkpoint's config.json and model.safetensors.
    '''
    model = MOMENT.from_pretrained(model_dir).to(device)
    if use_data_parallel:
        model = torch.nn.DataParallel(model).to(device)
    return model

def input_fn(input_data, content_type):
    '''
    Load the data: a JSON object with
        csv    : long-format CSV text with columns id, timestamp, value
        stride : optional, steps between window starts (default SEQ_LEN // 4)
    '''
    body = json.loads(input_data)
    data = pd.read_csv(io.StringIO(body["csv"]), parse_dates=["timestamp"])
    stride = int(body.get("stride", SEQ_LEN // 4))
    return data, stride

def predict_fn(data, model):
    '''
    Reconstruct each time series and compute the reconstruction error.
    '''
    data, stride = data

    # Put the rows of each time series in time order
    data = data.sort_values(["id", "timestamp"]).reset_index(drop=True)

    # Reconstruct each time series on its own, as a univariate series
    data["reconstruction"] = data.groupby("id")["value"].transform(
        lambda value: reconstruct(model, torch.tensor(value.to_numpy()).unsqueeze(0), stride)[0].numpy()
    )

    # Reconstruction error
    data["error"] = (data["value"] - data["reconstruction"]).abs()

    return data

def output_fn(prediction, accept):
    '''
    Return the predictions: CSV text with columns id, timestamp, value, reconstruction, error.
    '''
    csv_buffer = io.StringIO()
    prediction.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()
