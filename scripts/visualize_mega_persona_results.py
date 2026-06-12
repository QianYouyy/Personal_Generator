"""Create plots from MegaPersona generation, experiment, or evolution outputs.

Supports two output formats:
  --format png   Static PNG figures via matplotlib (default, backward-compatible)
  --format html  Self-contained interactive HTML report via Plotly.js
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mega_persona.visualization import visualize_result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize MegaPersona result artifacts.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to result directory, summary.json, final_summary.json, or generation JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. For PNG: figures/ subdirectory. For HTML: report.html inside this directory.",
    )
    parser.add_argument(
        "--format",
        choices=["png", "html"],
        default="png",
        help="Output format: png (matplotlib, default) or html (interactive Plotly report).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.format == "html":
        from src.mega_persona.html_viz import generate_html_report

        input_path = Path(args.input)
        if args.output_dir:
            output_path = Path(args.output_dir) / "report.html"
        else:
            output_path = None  # let generate_html_report pick the default
        result = generate_html_report(input_path, output_path)
        print(f"Saved interactive HTML report: {result}")
    else:
        output_dir = Path(args.output_dir) if args.output_dir else None
        written = visualize_result_path(Path(args.input), output_dir)
        print(f"Saved {len(written)} MegaPersona figure(s):")
        for path in written:
            print(f"  {path}")


if __name__ == "__main__":
    main()
