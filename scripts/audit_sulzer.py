"""Auditoría de solo lectura del CSV de la centrífuga Sulzer CFG-42.

Responde: ¿cuántas ventanas válidas (30 min de entrada completos y la
vibración exacta a +10 min) quedan en train, val y test al partir por fecha?

Uso:
    python scripts/audit_sulzer.py --csv data/raw/CFG_42_EDS_Data-_1_.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATE_COL = "Date"
DATE_FMT = "%m/%d/%Y %H:%M"
# Orden de las señales = orden de entrada del modelo.
SIGNALS = {
    "vibracion": "Analogue_Signals.30-VI-3274-A",
    "torque": "Analogue_Signals.30-NI-3271-A",
    "temp_A": "Analogue_Signals.30-TI-3273",
    "temp_B": "Analogue_Signals.30-TI-3275",
}


def load_signals(path: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Lee el CSV como texto y separa vacíos, texto no numérico y números."""
    raw = pd.read_csv(
        path, dtype=str, keep_default_na=False, na_values=[], encoding="utf-8-sig"
    )
    missing = [c for c in [DATE_COL, *SIGNALS.values()] if c not in raw.columns]
    if missing:
        raise SystemExit(f"Faltan columnas requeridas: {missing}")

    date = pd.to_datetime(raw[DATE_COL], format=DATE_FMT, errors="coerce")
    bad_dates = raw.loc[date.isna()]

    data, tokens = {}, {}
    for name, col in SIGNALS.items():
        text = raw[col]
        blank = text.str.strip() == ""
        numeric = pd.to_numeric(text, errors="coerce")
        weird = text[~blank & numeric.isna()]
        data[name] = numeric
        tokens[name] = (int(blank.sum()), weird.value_counts().head(8))

    df = pd.DataFrame(data)
    df.index = date
    return df[df.index.notna()], tokens, bad_dates


def invalid_runs(bad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve inicio y fin (exclusivo) de cada racha de minutos inválidos."""
    edges = np.diff(np.concatenate(([0], bad.astype(np.int8), [0])))
    return np.where(edges == 1)[0], np.where(edges == -1)[0]


def build_valid_window_mask(
    grid: pd.DataFrame, window: int, horizon: int, columns: list[str] | None = None
) -> pd.Series:
    """Marca el minuto t0 (última lectura de entrada) de cada ventana válida.

    Válida = las `window` lecturas que terminan en t0 tienen todas las señales
    y existe la vibración exacta en t0 + horizon. Usa la malla de 1 minuto,
    nunca un desplazamiento de filas.
    """
    cols = columns or list(SIGNALS)
    inputs_ok = grid[cols].notna().all(axis=1).astype(int).rolling(window).sum().eq(window)
    target_ok = grid["vibracion"].notna().shift(-horizon, fill_value=False)
    return inputs_ok & target_ok


def split_masks(
    index: pd.DatetimeIndex, window: int, horizon: int
) -> tuple[pd.Timestamp, pd.Timestamp, dict[str, np.ndarray]]:
    """Partición 70/15/15 por fecha, sin traslape entre bloques."""
    span = index.max() - index.min()
    b1 = (index.min() + span * 0.70).floor("D")
    b2 = (index.min() + span * 0.85).floor("D")
    first_input = index - pd.Timedelta(minutes=window - 1)
    target_time = index + pd.Timedelta(minutes=horizon)
    return b1, b2, {
        "train": np.asarray(target_time < b1),
        "val": np.asarray((first_input >= b1) & (target_time < b2)),
        "test": np.asarray(first_input >= b2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/raw/CFG_42_EDS_Data-_1_.csv"))
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--idle-torque", type=float, default=20.0)
    parser.add_argument("--min-windows", type=int, default=20_000)
    args = parser.parse_args()

    df, tokens, bad_dates = load_signals(args.csv)
    print(f"Filas con fecha válida: {len(df):,} | filas con fecha inválida: {len(bad_dates)}")
    if len(bad_dates):
        print(bad_dates.head(3).to_string())
    dups = int(df.index.duplicated().sum())
    if dups:
        print(f"AVISO: {dups} timestamps duplicados (se conserva el último)")
        df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    print("\n== Celdas por señal: vacías y texto no numérico ==")
    any_tokens = False
    for name, (n_blank, weird) in tokens.items():
        print(f"{name:10s} vacías={n_blank:,}  no numéricas={int(weird.sum()):,}")
        if len(weird):
            any_tokens = True
            print(weird.to_string())

    grid = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="1min"))
    print(f"\nRango: {grid.index.min()} -> {grid.index.max()} | minutos en la malla: {len(grid):,}")
    print(f"Minutos de la malla sin fila en el archivo: {len(grid) - len(df):,}")

    print("\n== Valores válidos por señal ==")
    print((grid.notna().mean() * 100).round(2).astype(str).add(" %").to_string())
    print("\n== Rangos (solo valores numéricos) ==")
    print(grid.describe().T[["min", "mean", "std", "max"]].round(2).to_string())

    all_ok = grid.notna().all(axis=1)
    bad = ~all_ok.to_numpy()
    starts, ends = invalid_runs(bad)
    lengths = ends - starts
    print(f"\n== Minutos con alguna señal inválida: {bad.sum():,} ({bad.mean() * 100:.1f} %) ==")
    print(f"Rachas: {len(lengths):,} | rachas de más de 10 min: {(lengths > 10).sum():,} | "
          f"minutos en rachas de más de 10 min: {int(lengths[lengths > 10].sum()):,}")
    print("Las 8 rachas más largas:")
    for i in np.argsort(lengths)[::-1][:8]:
        print(f"  {grid.index[starts[i]]} -> {grid.index[ends[i] - 1]}  "
              f"{lengths[i]:,} min ({lengths[i] / 1440:.1f} días)")
    monthly = pd.Series(bad, index=grid.index).groupby(grid.index.to_period("M")).mean() * 100
    print("% de minutos inválidos por mes:")
    print(monthly.round(1).to_string())

    valid = build_valid_window_mask(grid, args.window, args.horizon)
    valid3 = build_valid_window_mask(grid, args.window, args.horizon, ["vibracion", "torque", "temp_A"])
    frac = valid.mean()
    print(f"\n== Ventanas válidas ({args.window} min -> +{args.horizon} min) ==")
    print(f"Con las 4 señales: {int(valid.sum()):,} ({frac * 100:.1f} % de la malla)")
    print(f"Sin temp_B (3 señales): {int(valid3.sum()):,} ({valid3.mean() * 100:.1f} %)")

    b1, b2, masks = split_masks(grid.index, args.window, args.horizon)
    print(f"\nLímites por fecha: train < {b1.date()} <= val < {b2.date()} <= test")
    vib = grid["vibracion"].to_numpy()
    target = grid["vibracion"].shift(-args.horizon).to_numpy()
    torque = grid["torque"].to_numpy()
    valid_np = valid.to_numpy()
    train_mean = np.nanmean(target[valid_np & masks["train"]])

    counts = {}
    for name, mask in masks.items():
        sel = valid_np & mask
        counts[name] = int(sel.sum())
        if not sel.any():
            print(f"{name:5s} ventanas=0")
            continue
        running = sel & (torque >= args.idle_torque)
        print(
            f"{name:5s} ventanas={counts[name]:,} | operando={running.sum() / sel.sum() * 100:.0f} % | "
            f"vib media={target[sel].mean():.2f} desvío={target[sel].std():.2f}\n"
            f"      MAE repetir último: todas={np.abs(target[sel] - vib[sel]).mean():.3f}  "
            f"operando={np.abs(target[running] - vib[running]).mean():.3f} | "
            f"MAE media de train={np.abs(target[sel] - train_mean).mean():.3f}"
        )

    print("\n== Veredicto ==")
    nogo = [n for n, c in counts.items() if c < args.min_windows]
    caution = []
    if frac < 0.80:
        caution.append(f"solo {frac * 100:.1f} % de ventanas válidas (< 80 %)")
    if any_tokens:
        caution.append("hay celdas con texto no numérico: decide cómo tratarlas")
    if nogo:
        print(f"NO-GO: bloques con menos de {args.min_windows:,} ventanas: {nogo}")
    elif caution:
        print("CAUTION: " + "; ".join(caution))
    else:
        print("GO: sin bloqueos en los criterios revisados.")


if __name__ == "__main__":
    main()