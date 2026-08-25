"""Risk-tier slider (2026-08-21) -- a single 0-100 dial per program that computes and
applies conviction gate, R/R gate, position sizing, stop-loss/take-profit widths, and
portfolio circuit breakers together, instead of leaving each as an independent Settings-
page field with no shared concept tying them together. See
docs/superpowers/specs/2026-08-21-risk-tier-design.md for the full design and the
owner's real request behind it.

tier_value=50 ("Medium") is defined as exactly reproducing a program's own real,
already-live settings at the moment this feature's anchors were captured -- not a
generic industry-standard "medium." Low(0)/High(100) are offsets/multipliers off that
anchor. Interpolation is piecewise-linear across two segments (0->50, 50->100) so that
tier=50 always reproduces the anchor exactly regardless of how asymmetric the Low/High
endpoints are (e.g. cash reserve's x3.0 low-side vs. x0.5 high-side)."""

CONVICTION_FLOOR = 3.0
RR_FLOOR = 1.0


def _interp(t: float, low: float, anchor: float, high: float) -> float:
    if t <= 50.0:
        return low + (anchor - low) * (t / 50.0)
    return anchor + (high - anchor) * ((t - 50.0) / 50.0)


def compute_risk_tier_settings(tier_value: float, anchors: dict) -> dict:
    """Returns the 11 real settings values (8 factors + the 3 take-profit ladder
    prices, which scale by the same multiplier as stop_loss_pct to preserve each
    program's existing stop:target ratio) at the given tier. tier_value is clamped to
    [0, 100] first."""
    t = max(0.0, min(100.0, tier_value))

    conviction_high = max(CONVICTION_FLOOR, anchors["min_conviction_score"] - 1.5)
    rr_high = max(RR_FLOOR, anchors["min_risk_reward_ratio"] - 0.5)

    stop_loss_pct = _interp(
        t, anchors["stop_loss_pct"] * 0.6, anchors["stop_loss_pct"],
        anchors["stop_loss_pct"] * 1.6,
    )
    stop_loss_multiplier = stop_loss_pct / anchors["stop_loss_pct"]

    result = {
        "min_conviction_score": _interp(
            t, anchors["min_conviction_score"] + 1.5, anchors["min_conviction_score"],
            conviction_high,
        ),
        "min_risk_reward_ratio": _interp(
            t, anchors["min_risk_reward_ratio"] + 0.5, anchors["min_risk_reward_ratio"],
            rr_high,
        ),
        "starting_position_pct": _interp(
            t, anchors["starting_position_pct"] * 0.5, anchors["starting_position_pct"],
            anchors["starting_position_pct"] * 2.0,
        ),
        "max_loss_per_trade_pct": _interp(
            t, anchors["max_loss_per_trade_pct"] * 0.6, anchors["max_loss_per_trade_pct"],
            anchors["max_loss_per_trade_pct"] * 1.6,
        ),
        "stop_loss_pct": stop_loss_pct,
        "t1_pct": anchors["t1_pct"] * stop_loss_multiplier,
        "t2_pct": anchors["t2_pct"] * stop_loss_multiplier,
        "t3_pct": anchors["t3_pct"] * stop_loss_multiplier,
        "min_cash_reserve_pct": _interp(
            t, anchors["min_cash_reserve_pct"] * 3.0, anchors["min_cash_reserve_pct"],
            anchors["min_cash_reserve_pct"] * 0.5,
        ),
        "drawdown_halt_pct": _interp(
            t, anchors["drawdown_halt_pct"] * 0.6, anchors["drawdown_halt_pct"],
            anchors["drawdown_halt_pct"] * 1.6,
        ),
        "daily_loss_limit_pct": _interp(
            t, anchors["daily_loss_limit_pct"] * 0.6, anchors["daily_loss_limit_pct"],
            anchors["daily_loss_limit_pct"] * 1.6,
        ),
    }
    # Rounded before returning (fixed 2026-08-25, owner report: a live t1_pct/t2_pct/
    # t3_pct in AICryptoTrading's own config/settings.yaml showed ~15 digits after the
    # decimal, e.g. 11.840000000000002) -- every one of these values is either an
    # _interp() result or a chain of division-then-multiplication
    # (stop_loss_multiplier = stop_loss_pct / anchors["stop_loss_pct"], then
    # anchors["tN_pct"] * stop_loss_multiplier), both classic sources of IEEE-754
    # residue. Nothing downstream needs more precision than what's already shown
    # everywhere these values are displayed (Settings page inputs, the risk-tier
    # preview table's own .toFixed(1)/.toFixed(2) formatting) -- min_risk_reward_ratio
    # rounds to 2 decimals (matches its own .toFixed(2) display as an R/R ratio,
    # e.g. "2.10:1"); every other field is a plain percentage, rounded to 1 decimal
    # (matches their .toFixed(1) display). Rounding here, once, means every caller
    # (the live apply-to-settings path, the read-only preview endpoint, and whatever
    # ends up written to config/settings.yaml) gets a clean value for free, instead of
    # each needing its own defensive rounding.
    for key in result:
        result[key] = round(result[key], 2 if key == "min_risk_reward_ratio" else 1)
    return result


def risk_tier_label(tier_value: float) -> str:
    """Buckets a 0-100 tier value into a human-readable label for both the Settings-page
    display and the AI prompt section. Input is clamped to [0, 100] first."""
    t = max(0.0, min(100.0, tier_value))
    if t < 20.0:
        return "Low"
    if t < 40.0:
        return "Medium-Low"
    if t < 60.0:
        return "Medium"
    if t < 80.0:
        return "Medium-High"
    return "High"


_RISK_TIER_POSTURES = {
    "Low": (
        "Operate with a LOW risk tolerance right now. Prioritize capital "
        "preservation over upside -- decline a marginal setup rather than stretch "
        "for it, and favor an earlier, more conservative exit over riding a "
        "position further."
    ),
    "Medium-Low": (
        "Operate with a MEDIUM-LOW risk tolerance right now. Lean toward "
        "selectivity and caution, but a genuinely strong setup is still worth "
        "taking."
    ),
    "Medium": (
        "Operate with a balanced, MEDIUM risk tolerance right now -- weigh upside "
        "and downside evenly, neither reaching for marginal setups nor declining "
        "solid ones out of excess caution."
    ),
    "Medium-High": (
        "Operate with a MEDIUM-HIGH risk tolerance right now. Lean toward "
        "capturing upside -- a solid (not just a perfect) setup is worth taking, "
        "and a bit more room before cutting a loss is acceptable."
    ),
    "High": (
        "Operate with a HIGH risk tolerance right now. Greater risk tolerance is "
        "acceptable -- don't let a merely-good setup pass waiting for a perfect "
        "one, and give a position more room to work before treating a pullback as "
        "a reason to exit."
    ),
}


def build_risk_tier_prompt_section(tier_value: float, mode: str = "auto") -> str:
    """The AI-facing framing half of the risk-tier feature -- tells Claude directly
    what risk posture this portfolio is operating under, alongside the mechanical
    gate/sizing numbers compute_risk_tier_settings already changes. See
    _RISK_TIER_POSTURES for the owner-directed interpretive lean at each bucket
    (mirrors _build_market_context_section's own "give real data + explicit framing,
    trust Claude's judgment" pattern in src/research/engine.py).

    The Manual-mode gate lives here now, not at each caller (fixed 2026-08-24,
    GitHub #86) -- it used to be an independent inline ternary
    (`... if risk_tier_cfg.get("mode", "auto") != "manual" else ""`) written
    separately at each of the 2 real call sites in engine.py, the exact
    duplicated-check shape that already caused a real incident (2026-08-23:
    "AI framing ignored Manual mode entirely until fixed," patched as 2
    independent inline fixes rather than folded into one shared gate). In
    Manual mode the dial is deliberately disconnected from the real settings
    (see apply_risk_tier_to_settings/restore_anchors_to_settings below), so
    presenting its posture as the portfolio's "current risk tier" would hand
    Claude a framing that can actively contradict whatever the owner has
    actually hand-set the real gates to -- an empty string omits the section
    entirely, the same graceful-omission pattern every other optional prompt
    section in this codebase already uses, rather than describing it as
    inactive. Every caller now gets this guard for free instead of having to
    remember to add it -- including any future one."""
    if mode == "manual":
        return ""
    label = risk_tier_label(tier_value)
    t = max(0.0, min(100.0, tier_value))
    return (
        f"\n── RISK TIER ──\n"
        f"This portfolio's current risk tier is {label} ({t:.0f}/100).\n"
        f"{_RISK_TIER_POSTURES[label]}\n"
    )


RISK_TIER_DOTKEYS = {
    "min_conviction_score": "research.min_conviction_score",
    "min_risk_reward_ratio": "research.min_risk_reward_ratio",
    "starting_position_pct": "risk_management.starting_position_pct",
    "max_loss_per_trade_pct": "risk_management.max_loss_per_trade_pct",
    "stop_loss_pct": "take_profit.stop_loss_pct",
    "t1_pct": "take_profit.t1_pct",
    "t2_pct": "take_profit.t2_pct",
    "t3_pct": "take_profit.t3_pct",
    "min_cash_reserve_pct": "risk_management.min_cash_reserve_pct",
    "drawdown_halt_pct": "risk_management.drawdown_halt_pct",
    "daily_loss_limit_pct": "risk_management.daily_loss_limit_pct",
}


def apply_risk_tier_to_settings(
    coerced_values: dict, anchors: dict, current_tier_value: float | None = None,
) -> dict:
    """If a /api/settings payload's already-coerced values include "risk_tier.value"
    AND that value genuinely differs from current_tier_value (the tier value
    already stored, before this save), computes the 11 real factor values at that
    tier and merges them into a NEW dict (the input is never mutated), overwriting
    any of the same dotkeys the payload already had -- the slider move is the
    explicit action being applied, so it wins over whatever the browser's
    individual fields happened to hold at submit time.

    Real live incident, 2026-08-23 -- the risk-tier slider is one of every
    settings page's <input class="setting-input"> elements, so its current value
    is included in literally EVERY settings save, whether or not the caller
    touched it. Before this current_tier_value comparison existed,
    "risk_tier.value" being present in coerced_values (which it always was) was
    treated as "the slider was moved," so any unrelated save silently recomputed
    and overwrote 8 real trading fields back to the tier's own values every
    time -- confirmed live: this program's own real conviction gate drifted from
    the owner's actual 6.2 to the tier-40-computed 6.5 this way after an
    unrelated settings save. current_tier_value defaults to None (never equal to
    a real submitted float) so every existing caller that doesn't pass it keeps
    the original always-apply behavior.

    Returns the input dict unchanged (same contents) when "risk_tier.value" isn't
    present, or hasn't genuinely changed, so a plain settings save that never
    touches the slider is a true no-op here."""
    if "risk_tier.value" not in coerced_values:
        return coerced_values
    if coerced_values["risk_tier.value"] == current_tier_value:
        return coerced_values
    computed = compute_risk_tier_settings(coerced_values["risk_tier.value"], anchors)
    result = dict(coerced_values)
    for factor_key, dotkey in RISK_TIER_DOTKEYS.items():
        result[dotkey] = computed[factor_key]
    return result


def restore_anchors_to_settings(coerced_values: dict, anchors: dict) -> dict:
    """Restores the 11 real factor dotkeys to their anchor values, merged into a NEW
    dict (the input is never mutated) -- used when risk_tier.mode switches from
    "auto" to "manual". Per explicit owner direction, leaving auto mode must restore
    the program's real original settings, never a hardcoded factory default; the
    anchors ARE that original baseline (captured once, at the moment this program's
    risk-tier feature was first seeded -- see compute_risk_tier_settings's own
    docstring), so restoring to them is restoring the owner's own prior values."""
    result = dict(coerced_values)
    for factor_key, dotkey in RISK_TIER_DOTKEYS.items():
        result[dotkey] = anchors[factor_key]
    return result
