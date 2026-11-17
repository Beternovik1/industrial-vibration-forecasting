from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_history(history, model_name, baseline_mae, output_path):
    epochs = range(1, len(history["train_mse_normalized"]) + 1)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, (mse_axis, mae_axis) = plt.subplots(1, 2, figsize=(12, 5))
    mse_axis.plot(epochs, history["train_mse_normalized"], label="Training MSE")
    mse_axis.plot(epochs, history["validation_mse_normalized"], label="Validation MSE")
    mse_axis.set_title("Training and validation loss")
    mse_axis.set_xlabel("Epoch")
    mse_axis.set_ylabel("Normalized MSE")
    mse_axis.grid(True)
    mse_axis.legend()

    mae_axis.plot(epochs, history["validation_mae_mm_s"], label="Validation MAE")
    mae_axis.axhline(baseline_mae, linestyle="--", label="Repeat-last baseline")
    mae_axis.set_title("Validation forecast error")
    mae_axis.set_xlabel("Epoch")
    mae_axis.set_ylabel("MAE (mm/s RMS)")
    mae_axis.grid(True)
    mae_axis.legend()

    figure.suptitle(f"{model_name.upper()} training history")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
