"""On-demand Form 8949 CSV generator. Run once or twice a year:

    python scripts/generate_form_8949.py --year 2026

Never wired into a scheduled job, dashboard button, or API endpoint --
this is a manual, explicit action. See
docs/superpowers/specs/2026-08-02-tax-lot-tracking-design.md."""

import argparse
from pathlib import Path

from src.tax.form_8949 import generate_form_8949, write_csv


def main(jsonl_dir: Path, year: int, output_path: Path) -> dict:
    result = generate_form_8949(jsonl_dir, year)
    write_csv(result, output_path)
    print(f"Read {result['total_events_read']} trade events; "
          f"{result['paper_events_excluded']} excluded as paper.")
    print(f"{len(result['short_term_rows'])} short-term rows, "
          f"{len(result['long_term_rows'])} long-term rows written to {output_path}.")
    if result["review_rows"]:
        print(f"WARNING: {len(result['review_rows'])} row(s) need manual review "
              f"(no matching buy found) -- see the NEEDS REVIEW section in the CSV.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True, help="Tax year, e.g. 2026")
    parser.add_argument("--jsonl-dir", type=Path, default=Path("data/trade_history"))
    parser.add_argument("--output", type=Path, default=None,
                         help="Defaults to form_8949_<year>.csv in the current directory")
    args = parser.parse_args()
    output_path = args.output or Path(f"form_8949_{args.year}.csv")
    main(jsonl_dir=args.jsonl_dir, year=args.year, output_path=output_path)
