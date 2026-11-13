import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn

from datasets.sulzer import VIBRATION_COLUMN, get_sulzer_loaders
from callbacks.early_stopping import EarlyStopping
from engine.trainer import evaluate_repeat_last_mae, fit
from models.rnn import LSTMForecaster, RNNForecaster
from utils.device import get_device


def main():
    parser = argparse.ArgumentParser(description="Train a Sulzer vibration forecaster.")
    parser.add_argument("--model", choices=["rnn", "lstm"], default="lstm")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("epochs must be greater than zero.")
    if args.hidden_size <= 0:
        raise ValueError("hidden-size must be greater than zero.")

    torch.manual_seed(args.seed)
    device = get_device()
    print("Training on", device)
    print("Selected model:", args.model)

    (
        train_loader,
        validation_loader,
        _test_loader,
        mean,
        std,
        window_counts,
        _temperature_b_alarm_flag,
    ) = get_sulzer_loaders(
        Path("data/raw/CFG_42_EDS_Data-_1_.csv"),
        batch_size=args.batch_size,
    )
    print("Window counts:")
    print(window_counts)

    if args.model == "rnn":
        model = RNNForecaster(hidden_size=args.hidden_size)
    else:
        model = LSTMForecaster(hidden_size=args.hidden_size)
    model = model.to(device)

    vibration_mean = float(mean[VIBRATION_COLUMN])
    vibration_std = float(std[VIBRATION_COLUMN])
    import mlflow

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("sulzer-vibration-forecasting")

    with mlflow.start_run(run_name=args.model):
        mlflow.log_params(
            {
                "model": args.model,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "hidden_size": args.hidden_size,
                "seed": args.seed,
                "window_minutes": 30,
                "forecast_horizon_minutes": 10,
                "train_windows": len(train_loader.dataset),
                "validation_windows": len(validation_loader.dataset),
            }
        )

        baseline_mae = evaluate_repeat_last_mae(
            validation_loader,
            device,
            vibration_mean,
            vibration_std,
        )
        print(f"Validation repeat-last MAE: {baseline_mae:.4f} mm/s RMS")
        mlflow.log_metric("repeat_last_validation_mae_mm_s", baseline_mae)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=1e-4,
        )
        early_stopping = EarlyStopping(
            patience=6,
            min_delta=1e-3,
        )
        history, best_validation_mae = fit(
            model,
            train_loader,
            validation_loader,
            criterion,
            optimizer,
            device,
            args.epochs,
            early_stopping,
            vibration_mean,
            vibration_std,
        )

        for epoch_index, train_loss in enumerate(history["train_loss"], start=1):
            validation_mae = history["validation_mae_mm_s"][epoch_index - 1]
            mlflow.log_metric("train_mse_normalized", train_loss, step=epoch_index)
            mlflow.log_metric("validation_mae_mm_s", validation_mae, step=epoch_index)
        mlflow.log_metric("best_validation_mae_mm_s", best_validation_mae)

        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifacts_dir / f"best_{args.model}.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(str(model_path), artifact_path="checkpoints")
        print(f"Best validation MAE: {best_validation_mae:.4f} mm/s RMS")
        print("Saved model to", model_path)
        print("MLflow run ID:", mlflow.active_run().info.run_id)
    return history


if __name__ == "__main__":
    main()
