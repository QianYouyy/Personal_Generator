"""Report validation-only multi-objective candidate elites for a MegaPersona run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mega_persona.openevolve_adapter import (
    multi_objective_best_candidates,
    multi_objective_best_candidates_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan mega_eval/candidates and report candidates that are best under "
            "several validation-only objectives."
        )
    )
    parser.add_argument("run_dir", type=Path, help="Run dir, mega_eval dir, or candidates dir.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for JSON/Markdown reports. Defaults to the run directory.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print Markdown to stdout without writing report files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    candidates_dir = _find_candidates_dir(run_dir)
    report = multi_objective_best_candidates(candidates_dir)
    report = {
        **report,
        "source_dir": str(run_dir),
        "candidates_dir": str(candidates_dir),
    }
    markdown = multi_objective_best_candidates_markdown(report)
    if args.no_write:
        print(markdown)
        return

    output_dir = args.output_dir or _default_output_dir(run_dir, candidates_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "multi_objective_best_candidates.json"
    md_path = output_dir / "multi_objective_best_candidates.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def _find_candidates_dir(path: Path) -> Path:
    if path.name == "candidates" and path.exists():
        return path
    if (path / "mega_eval" / "candidates").exists():
        return path / "mega_eval" / "candidates"
    if (path / "candidates").exists():
        return path / "candidates"
    raise FileNotFoundError(
        f"Could not find candidates dir under {path}. Expected mega_eval/candidates or candidates."
    )


def _default_output_dir(run_dir: Path, candidates_dir: Path) -> Path:
    if candidates_dir.name == "candidates" and candidates_dir.parent.name == "mega_eval":
        return candidates_dir.parent.parent
    if run_dir.name == "candidates":
        return run_dir.parent
    return run_dir


if __name__ == "__main__":
    main()
