"""Orchestrates the tax package into Form 8949-shaped output: reads all
trade history, excludes paper trades (unconditionally -- see the
paper-exclusion guarantee in docs/superpowers/specs/2026-08-02-tax-lot-
tracking-design.md), reconstructs FIFO lots over the FULL history (not just
the target year -- required so a wash-sale basis adjustment on a
replacement lot carries forward correctly even if that lot is sold in a
later year than the loss that triggered it), applies wash-sale adjustment,
then filters to the requested year's sales for the final CSV."""

import csv
from pathlib import Path

from src.tax.holding_period import is_long_term
from src.tax.lot_matching import match_lots
from src.tax.trade_log_reader import read_tax_events
from src.tax.wash_sale import apply_wash_sale_adjustments


def _row(adjusted) -> dict:
    lot = adjusted.lot
    code = "W" if adjusted.disallowed_loss > 0 else ""
    # Cost Basis reflects adjusted_cost_basis (lot.cost_basis plus any basis increase
    # inherited from an earlier wash sale as this lot's replacement), not the original
    # lot.cost_basis -- fixed 2026-08-08, GitHub #51. Standard Form 8949 practice: a
    # replacement lot's cost basis is shown at its true bumped value directly, with no
    # separate adjustment code/amount on that row (Code W belongs on the loss-sale row
    # that triggered the disallowance). Before this fix, a basis-increase-only row
    # showed the ORIGINAL cost basis alongside an already-reduced Gain/Loss with
    # Adjustment Amount=0 -- Proceeds - Cost Basis didn't reconcile to Gain/Loss with
    # nothing in the row to explain the gap.
    return {
        "Description": lot.ticker,
        "Date Acquired": lot.acquired_date,
        "Date Sold": lot.sold_date,
        "Proceeds": lot.proceeds,
        "Cost Basis": adjusted.adjusted_cost_basis,
        "Adjustment Code": code,
        "Adjustment Amount": adjusted.disallowed_loss,
        "Gain/Loss": adjusted.adjusted_gain_loss,
        # Always present (empty string when None) -- must be in _COLUMNS below or a
        # wash-sale flag (e.g. "multiple qualifying buys", "replacement lot still open")
        # would be computed but silently never make it into the actual CSV output.
        "Note": adjusted.wash_sale_note or "",
    }


def generate_form_8949(jsonl_dir: Path, year: int) -> dict:
    all_events = read_tax_events(jsonl_dir)
    total_events_read = len(all_events)

    real_events = [e for e in all_events if not e.is_paper]
    paper_events_excluded = total_events_read - len(real_events)

    closed_lots, unmatched = match_lots(real_events)
    adjusted_lots = apply_wash_sale_adjustments(closed_lots, real_events)

    short_term_rows, long_term_rows = [], []
    for adjusted in adjusted_lots:
        sold_year = int(adjusted.lot.sold_date[:4])
        if sold_year != year:
            continue
        row = _row(adjusted)
        if is_long_term(adjusted.lot.acquired_date, adjusted.lot.sold_date):
            long_term_rows.append(row)
        else:
            short_term_rows.append(row)

    review_rows = [{
        "Description": u.ticker,
        "Date Sold": u.sold_date,
        "Shares": u.shares,
        "Price": u.price,
        "Note": "No matching buy found in trade history for this quantity -- "
                "verify manually before filing.",
    } for u in unmatched if int(u.sold_date[:4]) == year]

    return {
        "short_term_rows": short_term_rows,
        "long_term_rows": long_term_rows,
        "review_rows": review_rows,
        "total_events_read": total_events_read,
        "paper_events_excluded": paper_events_excluded,
    }


_COLUMNS = ["Description", "Date Acquired", "Date Sold", "Proceeds",
            "Cost Basis", "Adjustment Code", "Adjustment Amount", "Gain/Loss", "Note"]


def write_csv(result: dict, output_path: Path) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Part I -- Short-Term"])
        writer.writerow(_COLUMNS)
        for row in result["short_term_rows"]:
            writer.writerow([row.get(c, "") for c in _COLUMNS])
        writer.writerow([])
        writer.writerow(["Part II -- Long-Term"])
        writer.writerow(_COLUMNS)
        for row in result["long_term_rows"]:
            writer.writerow([row.get(c, "") for c in _COLUMNS])
        if result["review_rows"]:
            writer.writerow([])
            writer.writerow(["NEEDS REVIEW -- unmatched sells, verify manually"])
            writer.writerow(["Description", "Date Sold", "Shares", "Price", "Note"])
            for row in result["review_rows"]:
                writer.writerow([row["Description"], row["Date Sold"], row["Shares"],
                                  row["Price"], row["Note"]])
