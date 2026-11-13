import torch


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        predictions = model(inputs)
        loss = criterion(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    if total_samples == 0:
        raise ValueError("Cannot train with an empty DataLoader.")
    return total_loss / total_samples


def evaluate_mae(model, dataloader, device, vibration_mean, vibration_std):
    model.eval()
    total_absolute_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)

            # Compare in the original vibration units, not normalized units.
            predictions = predictions * vibration_std + vibration_mean
            targets = targets * vibration_std + vibration_mean
            total_absolute_error += (predictions - targets).abs().sum().item()
            total_samples += targets.size(0)

    if total_samples == 0:
        raise ValueError("Cannot calculate MAE with an empty DataLoader.")
    return total_absolute_error / total_samples


def evaluate_repeat_last_mae(dataloader, device, vibration_mean, vibration_std):
    total_absolute_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Vibration is feature zero; repeat its final observed input value.
            predictions = inputs[:, -1, 0].unsqueeze(-1)
            predictions = predictions * vibration_std + vibration_mean
            targets = targets * vibration_std + vibration_mean
            total_absolute_error += (predictions - targets).abs().sum().item()
            total_samples += targets.size(0)

    if total_samples == 0:
        raise ValueError("Cannot calculate the baseline MAE with an empty DataLoader.")
    return total_absolute_error / total_samples


def fit(
    model,
    train_loader,
    validation_loader,
    criterion,
    optimizer,
    device,
    epochs,
    early_stopping,
    vibration_mean,
    vibration_std,
):
    history = {
        "train_loss": [],
        "validation_mae_mm_s": [],
    }
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero.")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        validation_mae = evaluate_mae(
            model,
            validation_loader,
            device,
            vibration_mean,
            vibration_std,
        )
        history["train_loss"].append(train_loss)
        history["validation_mae_mm_s"].append(validation_mae)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train MSE: {train_loss:.4f} | "
            f"Validation MAE: {validation_mae:.4f} mm/s RMS"
        )

        early_stopping(model, validation_mae)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    early_stopping.restore_best_model(model)
    return history, early_stopping.best_metric
