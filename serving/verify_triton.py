import json
from datetime import timedelta
from urllib.request import Request, urlopen

import torch

from serving.export_lstm_onnx import (
    RUN_ID,
    WINDOW_END,
    WINDOW_START,
    load_sample_window,
    load_selected_model,
)

TRITON_URL = "http://localhost:8000/v2/models/sulzer_lstm/infer"
MODEL_NAME = "sulzer_lstm"


def main():
    model, _, _ = load_selected_model()
    sample = load_sample_window()

    with torch.inference_mode():
        torch_prediction = model(sample)

    request_body = {
        "inputs": [
            {
                "name": "RAW_WINDOW",
                "shape": [1, 30, 4],
                "datatype": "FP32",
                "data": sample.squeeze(0).tolist(),
            }
        ],
        "outputs": [{"name": "PREDICTED_VIBRATION"}],
    }
    request = Request(
        TRITON_URL,
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))

    triton_prediction = torch.tensor(
        result["outputs"][0]["data"], dtype=torch.float32
    ).reshape_as(torch_prediction)
    torch.testing.assert_close(
        torch_prediction,
        triton_prediction,
        rtol=1e-4,
        atol=1e-5,
    )

    difference = abs(torch_prediction.item() - triton_prediction.item())
    print("MLflow run:", RUN_ID)
    print("Input timestamps:", WINDOW_START, "through", WINDOW_END)
    print("Input shape/order: (1, 30, 4) = vibration, torque, temperature A, temperature B")
    print("Forecast timestamp:", WINDOW_END + timedelta(minutes=10))
    print("PyTorch prediction (mm/s RMS):", torch_prediction.item())
    print("Triton prediction (mm/s RMS):", triton_prediction.item())
    print("Absolute difference:", difference)
    print("Prediction parity passed.")


if __name__ == "__main__":
    main()
