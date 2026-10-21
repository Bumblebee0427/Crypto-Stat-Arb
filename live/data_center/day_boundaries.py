"""Day-edge evidence from retained parquet and already-verified archive metadata."""
from datetime import date,timedelta
import json
from pathlib import Path
import pyarrow.parquet as pq

from data_center.quality_tolerance import archive_reference,integer


def observed_edge(directory,symbol,day,previous):
    edge=None;candidates=[]
    for path in sorted(Path(directory).glob('*.parquet')):
        meta=pq.read_metadata(path)
        column=meta.schema.names.index('id')
        for group in range(meta.num_row_groups):
            stats=meta.row_group(group).column(column).statistics
            if stats is None or not stats.has_min_max:raise ValueError('missing ID statistics for boundary')
            ident=int(stats.max if previous else stats.min)
            if edge is None or (ident>edge if previous else ident<edge):edge=ident;candidates=[(path,group)]
            elif ident==edge:candidates.append((path,group))
    if edge is None:return None
    found=[]
    for path,group in candidates:
        table=pq.ParquetFile(path).read_row_group(group,columns=['id','timestamp_ms'])
        ids=table['id'].to_pylist();times=table['timestamp_ms'].to_pylist()
        found.extend((int(t),path.name.split('-')[0]) for i,t in zip(ids,times) if i==edge)
    if not found or len({t for t,h in found})!=1:raise ValueError('conflicting boundary ID times')
    return {'kind':'observed_neighbor_trade','symbol':symbol,'day':day,'id':edge,
            'timestamp_ms':found[0][0],'raw_segment_sha256':found[0][1]}


def evidence(root,symbol,day):
    root=Path(root);d=date.fromisoformat(day);result={}
    for label,offset in (('previous',-1),('next',1)):
        target=(d+timedelta(days=offset)).isoformat()
        path=root/'reference_daily'/target/(symbol+'.json')
        row=json.loads(path.read_text()) if path.exists() else {}
        key='last_trade_id' if offset<0 else 'first_trade_id'
        if archive_reference(row,symbol,target) and integer(row.get(key)):
            fields=('symbol','day','archive_url','archive_sha256','partition_basis','first_trade_id','last_trade_id')
            result[label]={'kind':'checksummed_archive_edge','symbol':symbol,'day':target,'id':row[key],
                           'archive_evidence':{k:row[k] for k in fields}}
        else:
            result[label]=observed_edge(root/'staged'/target/symbol,symbol,target,offset<0)
    return result
