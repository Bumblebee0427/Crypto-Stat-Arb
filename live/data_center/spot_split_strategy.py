"""B: global TOP50/small filters, then spot groups40/30%, minimum10 per leg.

Allocation ratios are BEFORE the existing2x account multiplier. Preserve the
original BTC protection. Groups each get half of each leg; empty groups cash.
"""
from data_center.base_strategy import finite,percentile,symbol_name

STRATEGY_VERSION='bigultra_v3_global_top50_spot_split_l40_s30_min10_20261115_v1'
ALLOCATION_METHOD='spot_split_l40_s30_min10_equal_v1'
RECIPE={'long_select_pct':.40,'short_select_pct':.30,'minimum_per_group_leg':10,
        'spot_budget_fraction':.50,'no_spot_budget_fraction':.50,
        'filter_scope':'global_before_spot_split','optimizer_blend':0.0}


def select(rows,btc_trend,membership):
    normalized={}
    for raw,row in rows.items():
        s=symbol_name(raw)
        if s in normalized:raise ValueError('duplicate normalized symbol')
        if s not in ('BTCUSDT','ETHUSDT') and row.get('tradable',True):
            if type(membership.get(s)) is not bool:raise ValueError('missing spot membership: '+s)
            normalized[s]=row
    allocations={};legs={};groups={};pool_counts={}
    for label,field,cut,high,fraction,budget,extra in [
        ('longs','small_long',.40,True,.40,.50,False),
        ('shorts','small_short',.35,False,.30,-.225,False),
        ('extra_shorts','small_short',.35,False,.30,-.275,True)]:
        available={s:r for s,r in normalized.items() if finite(r.get('main')) and finite(r.get(field))
                   and finite(r.get('quote_volume_mean')) and r['quote_volume_mean']>=0
                   and (not extra or finite(r.get('btc',btc_trend)))}
        small=percentile({s:r[field] for s,r in available.items()})
        volume=percentile({s:-r['quote_volume_mean'] for s,r in available.items()})
        pool=[s for s,r in available.items() if (small[s]>=cut if high else small[s]<=cut)
              and volume[s]<=.50 and (not extra or r.get('btc',btc_trend)<=.20)]
        pool_counts[label]=len(pool);legs[label]=[];groups[label]={}
        for name,has_spot in [('spot',True),('no_spot',False)]:
            candidates=[s for s in pool if membership[s]==has_spot]
            ordered=sorted(candidates,key=lambda s:available[s]['main'],reverse=high)
            k=min(len(ordered),max(int(len(ordered)*fraction),10))
            selected=[];previous=None;rank=0
            for i,s in enumerate(ordered,1):
                value=available[s]['main']
                if previous is None or value!=previous:rank,previous=i,value
                if rank<=k:selected.append(s)
            groups[label][name]={'pool_count':len(candidates),'selected':sorted(selected),
                                 'allocation_budget':budget*.50 if selected else 0.}
            for s in selected:allocations[s]=allocations.get(s,0.)+budget*.50/len(selected)
            legs[label].extend(selected)
        legs[label]=sorted(legs[label])
    return {'allocations':dict(sorted(allocations.items())),**legs,'groups':groups,
        'long_pool':pool_counts['longs'],'short_pool':pool_counts['shorts'],
        'btc_trend':btc_trend if finite(btc_trend) else None,
        'short_scale':1. if finite(btc_trend) and btc_trend<=.20 else .45,
        'recipe':RECIPE,'allocation_method':ALLOCATION_METHOD}
