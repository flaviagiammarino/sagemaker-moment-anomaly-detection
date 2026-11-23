import warnings
warnings.filterwarnings("ignore")

import io
import pandas as pd
import torch
from modules import MOMENT, SEQ_LEN

# Extract the environment configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_data_parallel = torch.cuda.device_count() > 1

@torch.inference_mode()
def reconstruct(
        model: MOMENT | torch.nn.DataParallel,
        series: dict[str, torch.Tensor],
        stride: int = SEQ_LEN // 4,
        batch_size: int = 256,
) -> list[torch.Tensor]:
    """
    Reconstruct several univariate time series in shared batches. Where windows
    overlap (stride < SEQ_LEN), each time step is the average of its reconstructions.

    Input:
        series : dict of [n_samples_i] tensors, one per time series, keyed by id
    Output:
        reconstructions : list of [n_samples_i] tensors, in the same order
    """

    # A stride above SEQ_LEN would leave time steps that no window covers
    if not 1 <= stride <= SEQ_LEN:
        raise ValueError(f"stride must be in [1, {SEQ_LEN}], got {stride}.")

    # The model's weights are float32: integer or float64 series would fail
    ids, series = list(series), [s.float() for s in series.values()]

    for id, s in zip(ids, series):
        # Missing or infinite values are not supported
        if not torch.isfinite(s).all():
            raise ValueError(f"Series {id}: missing (NaN) or infinite values are not supported.")
        # Each series must fill at least one full window
        if len(s) < SEQ_LEN:
            raise ValueError(f"Series {id}: need at least {SEQ_LEN} time steps, got {len(s)}.")

    # Window starts of each series: one every `stride` steps, and the last window ends
    # at the end of the series, so no window needs padding
    starts = [
        list(range(0, len(s) - SEQ_LEN, stride)) + [len(s) - SEQ_LEN]
        for s in series
    ]

    # Cut every series into windows of SEQ_LEN steps and stack them all together
    windows = torch.cat([
        s.unfold(0, SEQ_LEN, 1)[st]  
        for s, st in zip(series, starts)
    ]).unsqueeze(1)  # [n_windows x 1 x SEQ_LEN]

    # Feed the windows to the model in batches, in order. A batch can mix windows of
    # different series: each window is processed on its own
    dataloader = torch.utils.data.DataLoader(
        dataset=windows,
        batch_size=batch_size,
        shuffle=False,
    )

    # Reconstruct each batch on the device and collect the reconstructions on the CPU
    outputs = torch.cat([
        model(batch.to(device)).cpu() for batch in dataloader
    ]).squeeze(1)  # [n_windows x SEQ_LEN]

    # Give each series its own windows back: one [n_windows_i x SEQ_LEN] block per series
    outputs = outputs.split([len(st) for st in starts])

    # Average the reconstructions of each time step over the windows that contain it
    reconstructions = []
    for s, st, out in zip(series, starts, outputs):
        total = torch.zeros(len(s))  # sum of the reconstructions of each time step
        count = torch.zeros(len(s))  # number of windows that contain each time step
        for start, window in zip(st, out):  # window: [SEQ_LEN]
            total[start: start + SEQ_LEN] += window
            count[start: start + SEQ_LEN] += 1
        reconstructions.append(total / count)  # [n_samples_i]

    return reconstructions

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
    Load the data: long-format CSV text with columns id, timestamp, value.
    '''
    if isinstance(input_data, (bytes, bytearray)):
        input_data = input_data.decode("utf-8")
    return pd.read_csv(io.StringIO(input_data), dtype={"id": str}, parse_dates=["timestamp"])

def predict_fn(data, model):
    '''
    Reconstruct each time series.
    '''
    if data.empty:
        raise ValueError("The data contains no rows.")

    # Rows without an id would be left out of every series
    if data["id"].isna().any():
        raise ValueError("Missing ids are not supported.")

    # Put the rows of each time series in time order: each id is then a contiguous block
    data = data.sort_values(["id", "timestamp"]).reset_index(drop=True)

    # One univariate tensor per time series, keyed by id, in row order
    series = {
        id: torch.tensor(value.to_numpy())
        for id, value in data.groupby("id", sort=False)["value"]
    }

    # Reconstruct all the series together
    data["reconstruction"] = torch.cat(reconstruct(model, series)).numpy()

    return data

def output_fn(prediction, accept):
    '''
    Return the predictions: CSV text with columns id, timestamp, value, reconstruction.
    '''
    csv_buffer = io.StringIO()
    prediction.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()