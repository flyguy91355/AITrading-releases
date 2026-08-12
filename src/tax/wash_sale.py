"""IRS wash-sale loss disallowance: for every realized loss, checks for a
buy of the same ticker within 30 days before or after the sale. If found,
the loss is disallowed and its amount is added to the cost basis of the
earliest qualifying replacement buy (per IRS ordering guidance) -- if that
replacement lot has itself already been closed in this same run. See the
scoping note in docs/superpowers/plans/2026-08-02-tax-lot-tracking.md
Task 6 for why a still-open replacement lot can't be adjusted yet.

Rewritten 2026-08-08 (GitHub #51) -- the original two-pass design (pass 1:
classify every lot's OWN loss and accumulate basis_increases from raw
gain_loss; pass 2: apply the full accumulated increase to every lot sharing
a (ticker, acquired_date) key) had two real bugs, both traced to the same
root cause: this codebase always exits a position in staged T1/T2/final
tranches (see CLAUDE.md's "Exit Order System"), so one replacement buy
routinely produces MULTIPLE ClosedLots sharing the same acquired_date --

1. The full disallowed amount was applied to EVERY sub-lot independently
   instead of being distributed once across the combined position (a 3-way
   split tranche each ate the SAME $100 disallowed loss, tripling the real
   adjustment).
2. A lot that was itself already fully disallowed (adjusted_gain_loss forced
   to 0 in pass 1) could still get a basis increase subtracted from that
   already-zeroed value in pass 2, producing a phantom negative instead of
   staying $0, and the combined disallowed amount never cascaded forward
   into whatever lot THAT one was itself a replacement for.

Fixed with a single algorithm that iterates to a fixed point instead of a
fixed two passes: each round, every lot's OWN basis increase (this lot's
proportional share, by shares, of whatever total has accumulated so far
for its (ticker, acquired_date) key) is subtracted from its RAW gain_loss
to get an "effective" gain/loss, which is what actually gets tested against
a qualifying replacement buy and, if disallowed, cascades forward. Basis
increases only ever grow round over round (never shrink), and the set of
disallowed lots only ever grows, so this is guaranteed to converge within
at most len(closed_lots) rounds -- handles a wash-sale chain of any depth,
not just one level, and the split-tranche case is exactly the len(chain)==1
case of the same mechanism, no separate code path needed."""

from dataclasses import dataclass
from datetime import datetime, timedelta

from src.tax.lot_matching import ClosedLot
from src.tax.trade_log_reader import TaxTradeEvent

_WASH_SALE_WINDOW_DAYS = 30


@dataclass
class AdjustedLot:
    lot: ClosedLot
    disallowed_loss: float
    adjusted_gain_loss: float
    wash_sale_note: str | None
    # The real, basis-increase-adjusted cost basis for this specific lot's own sale --
    # equals lot.cost_basis plus whatever this lot inherited as a replacement for an
    # earlier wash sale (0 if none). Standard Form 8949 practice keeps a replacement
    # lot's shown cost basis at its true (bumped) value with NO separate adjustment
    # code/amount on that row -- only the loss-sale row that triggered the disallowance
    # gets Code W. Always Proceeds - adjusted_cost_basis + disallowed_loss == adjusted_gain_loss.
    adjusted_cost_basis: float = 0.0


def _qualifying_buys(ticker: str, sold_date: str, events: list[TaxTradeEvent]) -> list[TaxTradeEvent]:
    sold = datetime.fromisoformat(sold_date)
    window_start = sold - timedelta(days=_WASH_SALE_WINDOW_DAYS)
    window_end = sold + timedelta(days=_WASH_SALE_WINDOW_DAYS)
    return sorted(
        (e for e in events
         if e.is_buy and e.ticker == ticker
         and window_start <= datetime.fromisoformat(e.timestamp) <= window_end),
        key=lambda e: e.timestamp,
    )


def apply_wash_sale_adjustments(
    closed_lots: list[ClosedLot], all_events: list[TaxTradeEvent],
) -> list[AdjustedLot]:
    if not closed_lots:
        return []

    # Total shares per (ticker, acquired_date), computed once -- lets a basis increase
    # be distributed proportionally across every split tranche of the same replacement
    # buy instead of being applied in full to each one independently.
    shares_by_key: dict[tuple[str, str], float] = {}
    for lot in closed_lots:
        key = (lot.ticker, lot.acquired_date)
        shares_by_key[key] = shares_by_key.get(key, 0.0) + lot.shares

    basis_increases: dict[tuple[str, str], float] = {}
    results: list[AdjustedLot] = []

    for _ in range(len(closed_lots) + 1):
        new_basis_increases: dict[tuple[str, str], float] = {}
        new_results: list[AdjustedLot] = []

        for lot in closed_lots:
            key = (lot.ticker, lot.acquired_date)
            total_shares = shares_by_key.get(key) or lot.shares
            # Defensive (2026-08-08, caught by adversarial testing, not a live incident):
            # a genuinely malformed 0-share ClosedLot -- shouldn't occur from real trade
            # data (a 0-share sale wouldn't be logged as a trade) -- would otherwise divide
            # by zero here and crash the ENTIRE Form 8949 generation over one bad row.
            # Skip proportional distribution for a degenerate lot rather than let one
            # corrupt row take down the whole (twice-a-year, manually-run) report.
            own_increase = (
                basis_increases.get(key, 0.0) * (lot.shares / total_shares)
                if total_shares else 0.0
            )
            effective_gain_loss = lot.gain_loss - own_increase
            adjusted_cost_basis = lot.cost_basis + own_increase

            if effective_gain_loss >= 0:
                new_results.append(AdjustedLot(
                    lot=lot, disallowed_loss=0.0, adjusted_gain_loss=effective_gain_loss,
                    wash_sale_note=(
                        f"Cost basis increased ${own_increase:.2f} from an earlier wash sale."
                        if own_increase > 0 else None
                    ),
                    adjusted_cost_basis=adjusted_cost_basis,
                ))
                continue

            qualifying = [b for b in _qualifying_buys(lot.ticker, lot.sold_date, all_events)
                          if b.timestamp != lot.acquired_date]
            if not qualifying:
                new_results.append(AdjustedLot(
                    lot=lot, disallowed_loss=0.0, adjusted_gain_loss=effective_gain_loss,
                    wash_sale_note=(
                        f"Cost basis increased ${own_increase:.2f} from an earlier wash sale."
                        if own_increase > 0 else None
                    ),
                    adjusted_cost_basis=adjusted_cost_basis,
                ))
                continue

            replacement = qualifying[0]
            disallowed = abs(effective_gain_loss)
            note = (f"Wash sale: loss disallowed, ${disallowed:.2f} added to cost basis of "
                    f"the {replacement.timestamp} buy.")
            if own_increase > 0:
                note += f" (Includes ${own_increase:.2f} inherited from an earlier wash sale.)"
            if len(qualifying) > 1:
                note += (f" Multiple qualifying buys in the 61-day window ({len(qualifying)} total) "
                         "-- earliest used per IRS ordering guidance; verify with a professional.")

            rkey = (lot.ticker, replacement.timestamp)
            new_basis_increases[rkey] = new_basis_increases.get(rkey, 0.0) + disallowed
            new_results.append(AdjustedLot(
                lot=lot, disallowed_loss=disallowed, adjusted_gain_loss=0.0,
                wash_sale_note=note, adjusted_cost_basis=adjusted_cost_basis,
            ))

        if new_basis_increases == basis_increases:
            return new_results
        basis_increases = new_basis_increases
        results = new_results

    return results  # pragma: no cover -- unreachable given the monotonic-convergence bound above
