from utils.plotting import plot_history


def test_plot_history_saves_training_and_validation_curves(tmp_path):
    history = {
        "train_mse_normalized": [0.20, 0.12],
        "validation_mse_normalized": [0.22, 0.15],
        "validation_mae_mm_s": [0.31, 0.28],
    }
    output_path = tmp_path / "plots" / "lstm_training_history.png"

    saved_path = plot_history(history, "lstm", 0.30, output_path)

    assert saved_path.exists()
    assert saved_path.stat().st_size > 0
