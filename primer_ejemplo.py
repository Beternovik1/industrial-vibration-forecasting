"""Reproduce una ventana de 30 minutos y su vibración diez minutos después."""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

DATE_COLUMN = "Date"
FEATURE_COLUMNS = (
    "Analogue_Signals.30-VI-3274-A",
    "Analogue_Signals.30-NI-3271-A",
    "Analogue_Signals.30-TI-3273",
    "Analogue_Signals.30-TI-3275",
)
VIBRATION_COLUMN = FEATURE_COLUMNS[0]
DATE_FORMAT = "%m/%d/%Y %H:%M"
WINDOW_MINUTES = 30
HORIZON_MINUTES = 10
DEFAULT_CSV_PATH = Path("data/raw/CFG_42_EDS_Data-_1_.csv")
EXAMPLE_START = datetime(2023, 5, 18, 13, 31)


def load_example_window(
    csv_path: Path,
    start: datetime,
) -> tuple[list[tuple[float, ...]], float, datetime]:
    """Carga 30 lecturas consecutivas y la vibración a diez minutos del final.

    Args:
        csv_path: Ruta al CSV de la centrífuga.
        start: Fecha y hora de la primera lectura de entrada.

    Returns:
        Entrada con forma lógica (30, 4), etiqueta y fecha/hora de la etiqueta.

    Raises:
        FileNotFoundError: Si el CSV no existe.
        ValueError: Si faltan columnas, minutos o valores numéricos requeridos.
    """
    input_end = start + timedelta(minutes=WINDOW_MINUTES - 1)
    target_time = input_end + timedelta(minutes=HORIZON_MINUTES)
    relevant_values: dict[datetime, tuple[float, ...]] = {}

    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        _validate_columns(reader.fieldnames)
        for row in reader:
            if not row[DATE_COLUMN].strip():
                continue
            timestamp = datetime.strptime(row[DATE_COLUMN], DATE_FORMAT)
            if start <= timestamp <= target_time:
                relevant_values[timestamp] = _parse_features(row, timestamp)

    expected_times = [start + timedelta(minutes=i) for i in range(WINDOW_MINUTES)]
    missing_times = [timestamp for timestamp in expected_times if timestamp not in relevant_values]
    if missing_times:
        raise ValueError(f"Faltan minutos en la entrada: {missing_times}")
    if target_time not in relevant_values:
        raise ValueError(f"No hay medición objetivo en {target_time:%Y-%m-%d %H:%M}")

    inputs = [relevant_values[timestamp] for timestamp in expected_times]
    target = relevant_values[target_time][FEATURE_COLUMNS.index(VIBRATION_COLUMN)]
    return inputs, target, target_time


def _validate_columns(fieldnames: list[str] | None) -> None:
    """Valida que el CSV tenga fecha y las cuatro señales requeridas."""
    required_columns = {DATE_COLUMN, *FEATURE_COLUMNS}
    available_columns = set(fieldnames or [])
    missing_columns = required_columns - available_columns
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {sorted(missing_columns)}")


def _parse_features(row: dict[str, str], timestamp: datetime) -> tuple[float, ...]:
    """Convierte las señales de una fila y da contexto si falta un valor."""
    try:
        return tuple(float(row[column]) for column in FEATURE_COLUMNS)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Valor de señal ausente o no numérico en {timestamp:%Y-%m-%d %H:%M}"
        ) from error


def main() -> None:
    """Ejecuta el ejemplo de pronóstico y presenta sus dimensiones y etiqueta."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    args = parser.parse_args()

    inputs, target, target_time = load_example_window(args.csv, EXAMPLE_START)
    logger.info("Entrada: (%d, %d)", len(inputs), len(FEATURE_COLUMNS))
    logger.info("Ventana: %s a %s", EXAMPLE_START, EXAMPLE_START + timedelta(minutes=29))
    logger.info("Etiqueta: vibración de %s = %.2f mm/s RMS", target_time, target)

import pandas as pd
df = pd.read_csv("data/raw/CFG_42_EDS_Data-_1_.csv", encoding="utf-8-sig")
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %H:%M", errors="coerce")
print(len(df), df["Date"].isna().sum(), df["Date"].duplicated().sum())
print(df["Date"].sort_values().diff().value_counts().head(8))
cols = [c for c in df.columns if c.startswith("Analogue_Signals")]
print(df[cols].isna().sum()); print(df[cols].describe().T[["min","mean","std","max"]])


if __name__ == "__main__":
    main()
