"""Universe ready for a complete UTC day using prior-hour trade thresholds."""
from datetime import date, timedelta
import json
from pathlib import Path
from data_center.base_strategy import symbol_name


def new_symbols(universe, day):
    """Listing date falls inside the twelve required daily observations."""
    first=(date.fromisoformat(day)-timedelta(days=11)).isoformat()
    return {symbol_name(s) for s,d in universe.get('symbol_onboard_dates',{}).items()
            if first <= d <= day}


def readiness(universe, day, work=None):
    symbols = {symbol_name(s) for s in universe['tradable_symbols']} - {'BTCUSDT', 'ETHUSDT'}
    onboard = {symbol_name(s): d for s, d in universe.get('symbol_onboard_dates', {}).items()}
    # A listing day has neither the previous UTC day ID nor a 23:00 seed.
    # Defer this partial first day explicitly; mature-symbol gaps still fail.
    deferred = {s:['listing_day_has_no_prior_day_seed'] for s in symbols if onboard.get(s, '') == day}
    eligible = {s for s in symbols if onboard.get(s, '') <= day} - set(deferred)
    if work is not None:
        work=Path(work)
        state_path=work/'stage_state.json'
        state=json.loads(state_path.read_text()) if state_path.exists() else {}
        prior=(date.fromisoformat(day)-timedelta(days=1)).isoformat()
        for symbol in sorted(eligible & new_symbols(universe,day)):
            reasons=[]
            if symbol not in state.get('previous',{}):reasons.append('previous_day_id_anchor_missing')
            if not (work.parent/prior/'hours/23'/f'{symbol}.csv').exists():
                reasons.append('previous_hour_quantile_seed_missing')
            missing=[]
            for hour in range(24):
                path=work/'hours'/f'{hour:02d}'/f'{symbol}.csv'
                if not path.exists() or path.stat().st_size==0:missing.append(hour)
            if missing:reasons.append('raw_hours_missing:'+','.join(map(str,missing)))
            if reasons:deferred[symbol]=reasons
        eligible -= set(deferred)
    return eligible, deferred


def split(universe, day, work=None):
    eligible,deferred=readiness(universe,day,work)
    return eligible,set(deferred)
