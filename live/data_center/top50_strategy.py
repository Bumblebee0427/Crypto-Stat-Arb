"""Fixed 25-day TOP50 preselection overlay; separate version from old base."""
from data_center.base_strategy import select as base_select

STRATEGY_VERSION='bigultra_smallqty_btc30_qv25d_top50_20260917_v1'
LAGGED_STRATEGY_VERSION='bigultra_smallqty_btc30_qv599h_preclose_top50_previous_hour_20260927_v4'
LAGGED_FORMULA_VERSION='ref_notional_previous_utc_hour_closed_both_v3'
VOLUME_HOURS=599
VOLUME_TOP_PCT=.5


def select(rows,btc_trend):
    return base_select(rows,btc_trend,volume_top_pct=VOLUME_TOP_PCT)
