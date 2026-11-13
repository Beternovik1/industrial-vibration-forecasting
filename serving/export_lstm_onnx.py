from pathlib import Path

import onnx
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn

from datasets.sulzer import (
    FEATURE_COLUMNS,
    TEMPERATURE_B_COLUMN,
    get_sulzer_loaders,
    load_sulzer_frame,
)
from models.rnn import LSTMForecaster

RUN_ID = "2900d1cab6184fc9a2b2359b47f4c953"
CHECKPOINT_PATH = (
    Path("mlruns/artifacts/1")
    / RUN_ID
    / "artifacts/checkpoints/best_lstm.pth"
)
CSV_PATH = Path("data/raw/CFG_42_EDS_Data-_1_.csv")
MODEL_PATH = Path("serving/model_repository/sulzer_lstm/1/model.onnx")
WINDOW_START = pd.Timestamp("2023-05-18 13:31")
WINDOW_END = pd.Timestamp("2023-05-18 14:00")


class LSTMWithNormalization(nn.Module):
    def __init__(self, model, mean, std):
        super(LSTMWithNormalization, self).__init__()
        self.model = model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, raw_window):
        normalized_window = (raw_window - self.mean) / self.std
        normalized_prediction = self.model(normalized_window)
        return normalized_prediction * self.std[0] + self.mean[0]


def load_selected_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Selected MLflow checkpoint not found: {CHECKPOINT_PATH}")

    _, _, _, mean, std, _, _ = get_sulzer_loaders(CSV_PATH)
    mean = torch.tensor(mean.loc[list(FEATURE_COLUMNS)].to_numpy(), dtype=torch.float32)
    std = torch.tensor(std.loc[list(FEATURE_COLUMNS)].to_numpy(), dtype=torch.float32)

    model = LSTMForecaster(hidden_size=32)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True))
    model.eval()

    serving_model = LSTMWithNormalization(model, mean, std)
    serving_model.eval()
    return serving_model, mean, std


def load_sample_window():
    data = load_sulzer_frame(CSV_PATH)
    data.loc[data[TEMPERATURE_B_COLUMN].eq(0.0), TEMPERATURE_B_COLUMN] = float("nan")
    window = data.loc[WINDOW_START:WINDOW_END, list(FEATURE_COLUMNS)]

    if window.shape != (30, 4) or window.isna().any().any():
        raise ValueError("The requested 30-minute inference window is incomplete.")

    return torch.tensor(window.to_numpy(dtype="float32")).unsqueeze(0)


def main():
    model, mean, std = load_selected_model()
    sample = load_sample_window()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (sample,),
        str(MODEL_PATH),
        input_names=["RAW_WINDOW"],
        output_names=["PREDICTED_VIBRATION"],
        dynamic_axes={
            "RAW_WINDOW": {0: "batch_size"},
            "PREDICTED_VIBRATION": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,
    )

    onnx.checker.check_model(onnx.load(str(MODEL_PATH)))
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])

    with torch.inference_mode():
        torch_prediction = model(sample)
    onnx_prediction = session.run(
        ["PREDICTED_VIBRATION"], {"RAW_WINDOW": sample.numpy()}
    )[0]
    torch.testing.assert_close(
        torch_prediction,
        torch.from_numpy(onnx_prediction),
        rtol=1e-4,
        atol=1e-5,
    )

    print("Selected MLflow run:", RUN_ID)
    print("Training means:", mean.tolist())
    print("Training standard deviations:", std.tolist())
    print("ONNX export and ONNX Runtime parity passed:", MODEL_PATH)


if __name__ == "__main__":
    main()
