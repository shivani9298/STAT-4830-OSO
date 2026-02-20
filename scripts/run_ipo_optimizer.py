#!/usr/bin/env python3
"""
End-to-end: load data, train, export daily weights and summary.
Optional: use policy layer to print a simple rule for the retail trader.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

# Allow running from repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_layer import get_data
from src.train import run_training
from src.export import predict_weights, portfolio_stats, export_weights_csv, export_summary
from src.policy_layer import ipo_tilt_to_position_scale, policy_rule
import torch


def main():
    p = argparse.ArgumentParser(description="Train IPO portfolio optimizer and export weights.")
    p.add_argument("--start", default="2010-01-01", help="Data start date")
    p.add_argument("--end", default=None, help="Data end date")
    p.add_argument("--ipo-csv", default=None, help="Path to IPO index returns CSV (optional)")
    p.add_argument("--window", type=int, default=252, help="Rolling window length")
    p.add_argument("--val-frac", type=float, default=0.15, help="Validation fraction")
    p.add_argument("--epochs", type=int, default=50, help="Max epochs")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    p.add_argument("--checkpoint-dir", default=None, help="Directory to save best checkpoint")
    p.add_argument("--model", default="gru", choices=["mlp", "gru", "lstm", "transformer", "hybrid"],
                   help="Model architecture to train")
    p.add_argument("--weights-csv", default=None, help="Path to export daily weights CSV")
    p.add_argument("--summary-txt", default=None, help="Path to export summary for retail")
    args = p.parse_args()

    print("Loading data...")
    data = get_data(
        start=args.start,
        end=args.end,
        ipo_csv_path=args.ipo_csv,
        window_len=args.window,
        val_frac=args.val_frac,
    )
    print("Train windows: {}, Val windows: {}".format(
        data["X_train"].shape[0], data["X_val"].shape[0]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, history = run_training(
        data,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        model_type=args.model,
    )

    # Inference on validation set (or full dataset) and export
    X_val = data["X_val"]
    R_val = data["R_val"]
    dates_val = data["dates_val"]
    weights = predict_weights(model, X_val, device)
    stats = portfolio_stats(weights, R_val)

    out_dir = (Path(args.weights_csv).parent) if args.weights_csv else (ROOT / "output")
    weights_path = args.weights_csv or (out_dir / "daily_weights.csv")
    summary_path = args.summary_txt or out_dir / "summary.txt"

    export_weights_csv(dates_val, weights, weights_path)
    export_summary(stats, weights, summary_path)
    print(f"Exported weights to {weights_path}")
    print(f"Exported summary to {summary_path}")

    # Policy interpretation (IPO weight is second column when n_assets >= 2)
    avg_ipo = float(weights[:, 1].mean()) if weights.shape[1] >= 2 else 0.0
    scale = ipo_tilt_to_position_scale(avg_ipo)
    print("\n" + policy_rule(avg_ipo))
    print(f"Suggested position scale for next IPO: {scale:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
