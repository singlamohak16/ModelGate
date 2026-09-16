"""Train both Telco baselines from previously prepared partitions."""

import argparse
from pathlib import Path

from modelgate.training import train_baselines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/baseline"))
    args = parser.parse_args()
    try:
        report = train_baselines(args.prepared_dir, args.output_dir)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"Training error: {error}\n")
    for name, result in report["models"].items():
        metrics = result["reference_metrics"]
        print(f"{name}: F1={metrics['f1']:.4f}, PR-AUC={metrics['pr_auc']:.4f}")
    print(f"Metadata: {args.output_dir / 'baseline_metrics.json'}")


if __name__ == "__main__":
    main()
