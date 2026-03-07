from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves from JSON/CSV logs.")
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("artifacts/tat/training_history.csv"),
        help="Path to training history (.csv or .json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/tat/loss_curves.svg"),
        help="Output SVG path.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, float]]:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows = []
        for row in data:
            rows.append({k: float(v) if isinstance(v, (int, float)) else v for k, v in row.items()})
        return rows

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                parsed = {}
                for k, v in row.items():
                    if v is None or v == "":
                        parsed[k] = v
                        continue
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
                rows.append(parsed)
        return rows

    raise ValueError(f"Unsupported history format: {path.suffix}")


def write_loss_svg(rows: List[Dict[str, float]], output_path: Path) -> None:
    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_loss = [float(r["eval_loss"]) for r in rows]
    train_rec = [float(r["train_rec"]) for r in rows]
    val_rec = [float(r["eval_rec"]) for r in rows]
    train_con = [float(r["train_con"]) for r in rows]
    val_con = [float(r["eval_con"]) for r in rows]

    panels = [
        ("Total Loss", train_loss, val_loss),
        ("Reconstruction Loss", train_rec, val_rec),
        ("Contrastive Loss", train_con, val_con),
    ]

    width = 1300
    height = 420
    margin_left = 50
    margin_right = 20
    margin_top = 35
    margin_bottom = 40
    panel_gap = 24
    panel_width = (width - margin_left - margin_right - panel_gap * 2) / 3.0
    panel_height = height - margin_top - margin_bottom

    def _polyline(xs, ys, x0, y0, w, h) -> str:
        if len(xs) <= 1:
            return ""
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if y_min == y_max:
            y_min -= 1e-6
            y_max += 1e-6
        if x_min == x_max:
            x_max = x_min + 1
        pts = []
        for x, y in zip(xs, ys):
            px = x0 + (x - x_min) / (x_max - x_min) * w
            py = y0 + h - (y - y_min) / (y_max - y_min) * h
            pts.append(f"{px:.2f},{py:.2f}")
        return " ".join(pts)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white" />',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937} .axis{stroke:#9ca3af;stroke-width:1} .grid{stroke:#e5e7eb;stroke-width:1}</style>',
        '<text x="20" y="22" font-size="14" font-weight="600">TAT Training Curves</text>',
    ]

    for i, (title, ys_train, ys_val) in enumerate(panels):
        x0 = margin_left + i * (panel_width + panel_gap)
        y0 = margin_top
        all_y = ys_train + ys_val
        y_min, y_max = min(all_y), max(all_y)
        if y_min == y_max:
            y_min -= 1e-6
            y_max += 1e-6

        parts.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{panel_width:.2f}" height="{panel_height:.2f}" fill="none" class="axis" />')
        for frac in [0.25, 0.5, 0.75]:
            gy = y0 + panel_height * frac
            parts.append(f'<line x1="{x0:.2f}" y1="{gy:.2f}" x2="{x0 + panel_width:.2f}" y2="{gy:.2f}" class="grid" />')
        parts.append(f'<text x="{x0:.2f}" y="{y0 - 8:.2f}" font-size="12" font-weight="600">{title}</text>')
        parts.append(
            f'<text x="{x0:.2f}" y="{y0 + panel_height + 16:.2f}" font-size="10">y[{y_min:.4f}, {y_max:.4f}]</text>'
        )

        tr_points = _polyline(epochs, ys_train, x0, y0, panel_width, panel_height)
        va_points = _polyline(epochs, ys_val, x0, y0, panel_width, panel_height)
        if tr_points:
            parts.append(f'<polyline fill="none" stroke="#2563eb" stroke-width="2" points="{tr_points}" />')
        if va_points:
            parts.append(f'<polyline fill="none" stroke="#dc2626" stroke-width="2" points="{va_points}" />')

        lx = x0 + 8
        ly = y0 + 14
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 16}" y2="{ly}" stroke="#2563eb" stroke-width="2" />')
        parts.append(f'<text x="{lx + 20}" y="{ly + 3}" font-size="10">train</text>')
        parts.append(f'<line x1="{lx + 60}" y1="{ly}" x2="{lx + 76}" y2="{ly}" stroke="#dc2626" stroke-width="2" />')
        parts.append(f'<text x="{lx + 80}" y="{ly + 3}" font-size="10">val</text>')

    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.history)
    if not rows:
        raise ValueError("History is empty.")

    write_loss_svg(rows, args.output)
    print(f"Wrote plot: {args.output}")


if __name__ == "__main__":
    main()
