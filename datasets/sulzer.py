"""Read the Sulzer CSV and build PyTorch loaders for vibration forecasting."""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

DATE_COLUMN = "Date"
DATE_FORMAT = "%m/%d/%Y %H:%M"
FEATURE_COLUMNS = (
    "Analogue_Signals.30-VI-3274-A",
    "Analogue_Signals.30-NI-3271-A",
    "Analogue_Signals.30-TI-3273",
    "Analogue_Signals.30-TI-3275",
)
VIBRATION_COLUMN = FEATURE_COLUMNS[0]
TEMPERATURE_B_COLUMN = FEATURE_COLUMNS[3]
WINDOW_MINUTES = 30
FORECAST_HORIZON_MINUTES = 10
VALIDATION_START = pd.Timestamp("2023-05-18 00:00:00")
TEST_START = pd.Timestamp("2023-07-12 00:00:00")


class SulzerWindowDataset(Dataset):
    """Return one 30-minute, four-signal input and its normalized vibration target."""

    def __init__(self, split_data, valid_positions):
        # Keep timestamps so each anchor and target can be mapped to an exact row.
        self.valid_positions = valid_positions
        self.features = torch.tensor(
            split_data.loc[:, FEATURE_COLUMNS].to_numpy(dtype="float32"),
            dtype=torch.float32,
        )

        # The target row is looked up at anchor + 10 minutes, not guessed from a label.
        target_times = valid_positions + pd.Timedelta(minutes=FORECAST_HORIZON_MINUTES)
        self.anchor_rows = split_data.index.get_indexer(valid_positions)
        self.target_rows = split_data.index.get_indexer(target_times)
        if (self.anchor_rows < 0).any() or (self.target_rows < 0).any():
            raise ValueError("A window timestamp is missing from its date split.")

        vibration_position = FEATURE_COLUMNS.index(VIBRATION_COLUMN)
        target_rows = torch.tensor(self.target_rows, dtype=torch.long)
        self.targets = self.features[target_rows, vibration_position].unsqueeze(-1)

    def __len__(self):
        # The dataset length is the number of valid timestamp anchors.
        return len(self.valid_positions)

    def __getitem__(self, index):
        # Slice 30 consecutive rows ending at the anchor timestamp.
        end_row = int(self.anchor_rows[index]) + 1
        start_row = end_row - WINDOW_MINUTES
        x = self.features[start_row:end_row]

        # Return the vibration at the precomputed exact +10-minute target row.
        y = self.targets[index]
        return x, y


def load_sulzer_frame(csv_path):
    """Read numeric signals and preserve every minute on the source timeline."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Sulzer CSV not found: {csv_path}")

    data = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    required_columns = {DATE_COLUMN, *FEATURE_COLUMNS}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing columns: {sorted(missing_columns)}")

    dates = pd.to_datetime(data[DATE_COLUMN], format=DATE_FORMAT, errors="coerce")
    valid_dates = dates.notna()
    if (~valid_dates).any():
        print("Ignoring rows without a valid timestamp:", int((~valid_dates).sum()))
    data = data.loc[valid_dates, list(FEATURE_COLUMNS)].copy()
    data.index = pd.DatetimeIndex(dates.loc[valid_dates], name="Date_parsed")
    if data.index.has_duplicates:
        raise ValueError("The CSV contains duplicate timestamps.")

    numeric_data = data.apply(pd.to_numeric, errors="coerce").sort_index()
    invalid_signals = data.notna() & numeric_data.isna()
    if invalid_signals.any().any():
        print("Treating non-numeric sensor cells as missing:", int(invalid_signals.sum().sum()))

    # Reindexing inserts blank rows for missing timestamps instead of compressing time.
    minute_index = pd.date_range(
        numeric_data.index.min(), numeric_data.index.max(), freq="1min", name="Date_parsed"
    )
    return numeric_data.reindex(minute_index)


def split_by_date(data):
    """Split a timestamp-indexed frame using half-open source-clock intervals."""
    timestamps = data.index
    train = data.loc[timestamps < VALIDATION_START]
    validation = data.loc[(timestamps >= VALIDATION_START) & (timestamps < TEST_START)]
    test = data.loc[timestamps >= TEST_START]
    return train, validation, test


def find_valid_window_positions(split_data):
    """Find anchors with 30 complete inputs and an observed vibration at +10 minutes."""
    anchors = split_data.index

    # Rolling over 30 rows checks t-29 through t, including blank minutes on the grid.
    complete_rows = split_data.loc[:, FEATURE_COLUMNS].notna().all(axis=1)
    complete_inputs = (
        complete_rows.astype("int8")
        .rolling(WINDOW_MINUTES, min_periods=WINDOW_MINUTES)
        .sum()
        .eq(WINDOW_MINUTES)
    )

    # Reindex by exact timestamps; the nine intervening minutes are not checked.
    target_times = anchors + pd.Timedelta(minutes=FORECAST_HORIZON_MINUTES)
    target_values = split_data[VIBRATION_COLUMN].reindex(target_times)
    observed_targets = pd.Series(target_values.notna().to_numpy(), index=anchors)
    valid = complete_inputs & observed_targets
    return anchors[valid.to_numpy()]


def get_sulzer_loaders(csv_path, batch_size=64):
    """Read, split, normalize, and return three ordered DataLoaders plus diagnostics.

    Returns train, validation, and test loaders, training means/stds, train/validation
    before-and-after window counts, and a temperature-B alarm review flag.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    # 1. Read the CSV and retain the complete minute timeline.
    raw_data = load_sulzer_frame(csv_path)

    # Keep alarm-range readings, but flag train/validation observations for review.
    temperature_b_alarm_flag = raw_data.loc[
        raw_data.index < TEST_START, TEMPERATURE_B_COLUMN
    ].ge(85.0)

    # 2. Make one prepared copy and treat exact temperature-B zeros as missing.
    prepared_data = raw_data.copy()
    temperature_b_zeros = prepared_data[TEMPERATURE_B_COLUMN].eq(0.0)
    prepared_data.loc[temperature_b_zeros, TEMPERATURE_B_COLUMN] = float("nan")

    # 3. Split both views by time; no window can cross a split boundary.
    raw_train, raw_validation, _ = split_by_date(raw_data)
    train_data, validation_data, test_data = split_by_date(prepared_data)

    # Keep before/after counts in the EDA notebook; valid positions are reused by Dataset.
    positions_before = {
        "train": find_valid_window_positions(raw_train),
        "validation": find_valid_window_positions(raw_validation),
    }
    train_positions = find_valid_window_positions(train_data)
    validation_positions = find_valid_window_positions(validation_data)
    test_positions = find_valid_window_positions(test_data)

    # 4. Fit one mean and population standard deviation per column on training only.
    mean = train_data.mean(skipna=True)
    std = train_data.std(skipna=True, ddof=0)
    if mean.isna().any() or std.isna().any():
        raise ValueError("Every signal needs observed readings in the training split.")
    zero_std = std.eq(0.0)
    if zero_std.any():
        print("Using a scale of 1 for constant training signals:", list(std.index[zero_std]))
        std = std.mask(zero_std, 1.0)

    # Apply the training statistics unchanged to every split; NaNs remain missing.
    train_scaled = (train_data - mean) / std
    validation_scaled = (validation_data - mean) / std
    test_scaled = (test_data - mean) / std

    # 5. Create one Dataset per split from the already-found valid window positions.
    train_dataset = SulzerWindowDataset(train_scaled, train_positions)
    validation_dataset = SulzerWindowDataset(validation_scaled, validation_positions)
    test_dataset = SulzerWindowDataset(test_scaled, test_positions)

    # Match Jena: preserve the order of samples in all three loaders.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Counts are reported for train/validation only; the test loader is not evaluated.
    window_counts = pd.DataFrame(
        {
            "before_temperature_b_zero_mask": [
                len(positions_before["train"]), len(positions_before["validation"])
            ],
            "after_temperature_b_zero_mask": [
                len(train_positions), len(validation_positions)
            ],
        },
        index=["train", "validation"],
    )
    window_counts["windows_removed_by_temperature_b_zero_mask"] = (
        window_counts["before_temperature_b_zero_mask"] - window_counts["after_temperature_b_zero_mask"]
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
        mean,
        std,
        window_counts,
        temperature_b_alarm_flag,
    )
