import re
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.utils.log_kit import logger


def load_coin_cap(file_path: str, candle_df: pd.DataFrame, save_cols: list, symbol: str) -> pd.DataFrame | None:
    """
    加载coinmarketcap数据
    :param file_path: 文件路径
    :param candle_df: 原始k线数据
    :param save_cols: 需要保存的列
    :param symbol: 币种
    :return: 一个 df 数据
    """
    data_path = Path(file_path)
    # 合并选币名称并剔除空字符串
    symbols = (
        {symbol} if symbol else
        {s for s in (list(candle_df['symbol_spot'].unique()) + list(candle_df['symbol_swap'].unique())) if s}
    )
    # 兼容 回测 & 实盘。
    # 回测币种带 -USDT，实盘币种带 USDT
    all_file_paths = [data_path / f'{symbol.replace("-", "")[:-4]}-USDT.csv' for symbol in symbols]

    # 读取数据
    extra_df_list = []
    for file_path in all_file_paths:
        try:
            extra_df = pd.read_csv(file_path, encoding='gbk', skiprows=1, parse_dates=['candle_begin_time'])
            extra_df_list.append(extra_df)
        except Exception as e:
            continue

    if not extra_df_list:
        logger.warning(f'coin_cap 没有找到任何数据，symbol={symbols}')
        return None

    if len(candle_df) == 1:
        logger.warning(f'{candle_df["symbol"].iloc[0]}币种K线数据不足，跳过')
        return None

    # 预处理
    extra_df = pd.concat(extra_df_list, ignore_index=True, copy=False)
    time_diff = candle_df['candle_begin_time'].iloc[1] - candle_df['candle_begin_time'].iloc[0]
    hour_diff = time_diff.components.days * 24 + time_diff.components.hours
    hold_period_type = 'D' if hour_diff == 24 else 'H'

    if hold_period_type == 'H':
        extra_df['candle_begin_time'] += pd.Timedelta('23H')

    # 合并数据
    merge_df = pd.merge(candle_df[['candle_begin_time', 'close']], extra_df, how='left', on='candle_begin_time')

    # 处理 cmc 数据中的 usd_price与close 数据的不一致
    # 例：
    # 1000SATS usd_price = 0.0000001， 但是close = 0.0001
    times = merge_df['close'] / merge_df['usd_price']
    times = times.map(lambda x: 10 ** np.log10(x).round(0))
    times = [times.mode().iloc[0] if times.notna().sum() > 0 else np.nan] * len(merge_df)

    # 筛选指定列
    columns = save_cols if save_cols else merge_df.columns
    for col in columns:
        if 'supply' in col:
            merge_df[col] = merge_df[col] / times
        merge_df[col] = merge_df[col].ffill()

    return merge_df[columns]


_TIME_COL_CANDIDATES = (
    'candle_begin_time', 'open_time', 'timestamp', 'ts', 'datetime', 'dt', 'time'
)
_SYMBOL_COL_CANDIDATES = (
    'symbol', 'symbol_swap', 'symbol_spot', 'pair', 'ticker', 'instrument', 'asset'
)
_FACTOR_DATE_RANGE_CACHE: dict[tuple[str, str], tuple[pd.Timestamp, pd.Timestamp] | None] = {}
_FACTOR_TIME_ALIGNMENT_CACHE: dict[tuple[str, str], int] = {}
_FACTOR_TIME_ALIGNMENT_LOGGED: set[tuple[str, str]] = set()
_PARQUET_SCHEMA_CACHE: dict[str, tuple[str, ...] | None] = {}


def _normalize_symbol(value: str) -> str:
    # Preserve Unicode asset names; only remove separators/punctuation.
    # Stripping non-ASCII letters collapses distinct Chinese symbols to USDT.
    return re.sub(r'[\W_]', '', str(value).upper(), flags=re.UNICODE)


def _pick_existing_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None


def _parse_datetime_series(series: pd.Series) -> pd.Series:
    # 兼容常见整型时间戳单位，并统一为 timezone-naive UTC 时间
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors='coerce')
        max_abs = numeric.abs().max()
        if pd.isna(max_abs):
            dt = pd.to_datetime(numeric, errors='coerce', utc=True)
            return dt.dt.tz_convert(None)

        if max_abs >= 1e17:
            unit = 'ns'
        elif max_abs >= 1e14:
            unit = 'us'
        elif max_abs >= 1e11:
            unit = 'ms'
        else:
            unit = 's'

        dt = pd.to_datetime(numeric, unit=unit, errors='coerce', utc=True)
        return dt.dt.tz_convert(None)

    dt = pd.to_datetime(series, errors='coerce', utc=True)
    return dt.dt.tz_convert(None)


def _resolve_value_column(
    df: pd.DataFrame,
    save_cols: list[str],
    fallback_name: str,
    time_col: str | None,
    symbol_col: str | None,
) -> str | None:
    for col in save_cols:
        if col in df.columns:
            return col

    if fallback_name in df.columns:
        return fallback_name

    excluded_cols = {time_col, symbol_col, 'year', 'month', 'day'}
    excluded_cols = {c for c in excluded_cols if c}
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excluded_cols]
    if len(numeric_cols) == 1:
        return numeric_cols[0]

    return None


def _resolve_value_column_from_columns(
    columns: list[str],
    save_cols: list[str],
    fallback_name: str,
    time_col: str | None,
    symbol_col: str | None,
) -> str | None:
    for col in save_cols:
        if col in columns:
            return col

    if fallback_name in columns:
        return fallback_name

    if 'value' in columns:
        return 'value'

    excluded_cols = {time_col, symbol_col, 'year', 'month', 'day'}
    excluded_cols = {c for c in excluded_cols if c}
    value_cols = [c for c in columns if c not in excluded_cols]
    if len(value_cols) == 1:
        return value_cols[0]

    return None


def _read_parquet_with_retry(parquet_file: Path, **kwargs) -> pd.DataFrame:
    last_error = None
    for attempt in range(5):
        try:
            return pd.read_parquet(parquet_file, **kwargs)
        except PermissionError as e:
            last_error = e
        except OSError as e:
            last_error = e
            if getattr(e, 'errno', None) != 13:
                raise
        except Exception as e:
            message = str(e)
            if 'Permission denied' not in message:
                raise
            last_error = e

        time.sleep(0.05 * (attempt + 1))

    raise last_error


def _path_op_with_retry(path: Path, op):
    last_error = None
    for attempt in range(5):
        try:
            return op(path)
        except PermissionError as e:
            last_error = e
        except OSError as e:
            last_error = e
            if getattr(e, 'errno', None) != 13:
                raise

        time.sleep(0.05 * (attempt + 1))

    if last_error is not None:
        return None
    return None


def _path_exists_with_retry(path: Path) -> bool | None:
    return _path_op_with_retry(path, lambda p: p.exists())


def _path_is_file_with_retry(path: Path) -> bool | None:
    return _path_op_with_retry(path, lambda p: p.is_file())


def _path_is_dir_with_retry(path: Path) -> bool | None:
    return _path_op_with_retry(path, lambda p: p.is_dir())


def _path_glob_with_retry(path: Path, pattern: str) -> list[Path] | None:
    return _path_op_with_retry(path, lambda p: sorted(p.glob(pattern)))


def _get_parquet_columns(parquet_file: Path) -> tuple[str, ...] | None:
    key = str(parquet_file)
    if key in _PARQUET_SCHEMA_CACHE:
        return _PARQUET_SCHEMA_CACHE[key]

    try:
        import pyarrow.parquet as pq

        columns = tuple(pq.read_schema(parquet_file).names)
    except Exception:
        columns = None

    _PARQUET_SCHEMA_CACHE[key] = columns
    return columns


def _symbol_filter_values(symbol_values: set[str]) -> set[str]:
    values = set()
    for value in symbol_values:
        raw = str(value)
        if not raw or raw.lower() == 'nan':
            continue

        upper = raw.upper()
        norm = _normalize_symbol(raw)
        values.update({raw, upper, norm})

        if norm.endswith('USDT') and len(norm) > 4:
            base = norm[:-4]
            values.update({
                f'{base}-USDT',
                f'{base}/USDT',
                f'{base}_USDT',
            })

    values.update({str(v).lower() for v in values})
    return {v for v in values if v}


def _read_factor_parquet_file(
    parquet_file: Path,
    save_cols: list[str],
    factor_name: str,
    symbol_filter_values: set[str],
    use_symbol_filter: bool,
) -> tuple[pd.DataFrame | None, str | None, str | None, str | None]:
    columns = _get_parquet_columns(parquet_file)
    if columns is None:
        try:
            raw = _read_parquet_with_retry(parquet_file)
        except Exception as e:
            logger.warning(f'读取外部因子 parquet 失败: {parquet_file}, err={e}')
            return None, None, None, None

        cols = raw.columns.tolist()
        time_col = _pick_existing_column(cols, _TIME_COL_CANDIDATES)
        symbol_col = _pick_existing_column(cols, _SYMBOL_COL_CANDIDATES)
        value_col = _resolve_value_column(raw, save_cols, factor_name, time_col, symbol_col)
        return raw, time_col, symbol_col, value_col

    cols = list(columns)
    time_col = _pick_existing_column(cols, _TIME_COL_CANDIDATES)
    symbol_col = _pick_existing_column(cols, _SYMBOL_COL_CANDIDATES)
    value_col = _resolve_value_column_from_columns(cols, save_cols, factor_name, time_col, symbol_col)
    if time_col is None or value_col is None:
        return None, time_col, symbol_col, value_col

    read_cols = [time_col, value_col]
    if symbol_col:
        read_cols.append(symbol_col)
    for col in save_cols:
        if col in cols:
            read_cols.append(col)
    read_cols = list(dict.fromkeys(read_cols))

    filters = None
    if use_symbol_filter and symbol_col and symbol_filter_values:
        filters = [(symbol_col, 'in', sorted(symbol_filter_values))]

    try:
        raw = _read_parquet_with_retry(parquet_file, columns=read_cols, filters=filters)
    except Exception as e:
        if filters is not None:
            try:
                raw = _read_parquet_with_retry(parquet_file, columns=read_cols)
            except Exception as e2:
                logger.warning(f'读取外部因子 parquet 失败: {parquet_file}, err={e2}')
                return None, time_col, symbol_col, value_col
        else:
            logger.warning(f'读取外部因子 parquet 失败: {parquet_file}, err={e}')
            return None, time_col, symbol_col, value_col

    return raw, time_col, symbol_col, value_col


def _discover_factor_date_range(base_path: Path, factor_name: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    key = (str(base_path), factor_name)
    if key in _FACTOR_DATE_RANGE_CACHE:
        return _FACTOR_DATE_RANGE_CACHE[key]

    dates = []
    year_dirs = _path_glob_with_retry(base_path, "[0-9][0-9][0-9][0-9]")
    if year_dirs is None:
        _FACTOR_DATE_RANGE_CACHE[key] = None
        return None

    for year_dir in year_dirs:
        if not _path_is_dir_with_retry(year_dir):
            continue
        day_dirs = _path_glob_with_retry(year_dir, "[0-9][0-9][0-9][0-9]_[0-9][0-9]_[0-9][0-9]")
        if day_dirs is None:
            continue

        for day_dir in day_dirs:
            factor_file = day_dir / f"{factor_name}.parquet"
            if _path_exists_with_retry(factor_file):
                dates.append(pd.Timestamp(day_dir.name.replace("_", "-")))

    if not dates:
        _FACTOR_DATE_RANGE_CACHE[key] = None
        return None

    result = (min(dates), max(dates))
    _FACTOR_DATE_RANGE_CACHE[key] = result
    return result


def load_partitioned_factor_parquet(
    file_path: str,
    candle_df: pd.DataFrame,
    save_cols: list,
    symbol: str,
) -> pd.DataFrame | None:
    """
    加载按 年/日期 分区存储的因子 parquet 数据。
    目录结构示例：
    {root}/2026/2026_01_01/concave_middle_trade_imbalance_log.parquet
    """
    if candle_df.empty or 'candle_begin_time' not in candle_df.columns:
        return None

    base_path = Path(file_path)
    factor_name = save_cols[0] if save_cols else 'external_factor'
    save_cols = save_cols if save_cols else [factor_name]

    path_exists = _path_exists_with_retry(base_path)
    if path_exists is None:
        logger.warning(f'外部因子目录无权限访问: {base_path}')
        return None

    if not path_exists:
        logger.warning(f'外部因子目录不可用或无权限访问: {base_path}')
        return None

    candle_time = pd.to_datetime(candle_df['candle_begin_time'], errors='coerce', utc=True).dt.tz_convert(None)
    if candle_time.notna().sum() == 0:
        return None

    start_date = candle_time.min().normalize()
    end_date = candle_time.max().normalize()

    symbol_values = set()
    if symbol:
        symbol_values.add(symbol)
    for col in ('symbol', 'symbol_spot', 'symbol_swap'):
        if col in candle_df.columns:
            symbol_values.update(candle_df[col].dropna().astype(str).tolist())
    symbol_values = {s for s in symbol_values if s and s.lower() != 'nan'}
    symbol_norm_set = {_normalize_symbol(s) for s in symbol_values}
    symbol_filter_values = _symbol_filter_values(symbol_values)

    parquet_files: list[Path] = []
    is_file = _path_is_file_with_retry(base_path)
    if is_file is None:
        logger.warning(f'外部因子路径无权限访问: {base_path}')
        return None

    if is_file and base_path.suffix == '.parquet':
        parquet_files = [base_path]
    else:
        available_range = _discover_factor_date_range(base_path, factor_name)
        if available_range is None:
            return None

        available_start, available_end = available_range
        query_start = max(start_date, available_start)
        query_end = min(end_date, available_end)
        if query_start > query_end:
            return None

        for day in pd.date_range(query_start, query_end, freq='D'):
            day_file = base_path / f'{day.year}' / day.strftime('%Y_%m_%d') / f'{factor_name}.parquet'
            day_file_exists = _path_exists_with_retry(day_file)
            if day_file_exists is None:
                return None
            if day_file_exists:
                parquet_files.append(day_file)

    if not parquet_files:
        return None

    def _load_pieces(use_symbol_filter: bool) -> list[pd.DataFrame]:
        out_pieces = []
        for parquet_file in parquet_files:
            raw, time_col, symbol_col, value_col = _read_factor_parquet_file(
                parquet_file,
                save_cols,
                factor_name,
                symbol_filter_values,
                use_symbol_filter=use_symbol_filter,
            )

            if raw is None or raw.empty:
                continue
            if time_col is None or value_col is None:
                continue

            raw = raw.copy()
            raw['candle_begin_time'] = _parse_datetime_series(raw[time_col])
            raw = raw[raw['candle_begin_time'].notna()]
            if raw.empty:
                continue

            if symbol_col and symbol_norm_set:
                sym_norm = raw[symbol_col].astype(str).map(_normalize_symbol)
                raw = raw[sym_norm.isin(symbol_norm_set)]
                if raw.empty:
                    continue

            piece = pd.DataFrame({'candle_begin_time': raw['candle_begin_time']})
            for idx, col in enumerate(save_cols):
                if col in raw.columns:
                    piece[col] = pd.to_numeric(raw[col], errors='coerce')
                elif idx == 0:
                    piece[col] = pd.to_numeric(raw[value_col], errors='coerce')
                else:
                    piece[col] = np.nan
            out_pieces.append(piece)
        return out_pieces

    use_symbol_filter = os.environ.get('SELECT_COIN_PARQUET_SYMBOL_FILTER', '1') != '0'
    pieces = _load_pieces(use_symbol_filter=use_symbol_filter)

    # Safety fallback: if symbol-level parquet filtering produced no data at all,
    # retry the exact old broad-read path.  This preserves correctness for any
    # external dataset whose symbol encoding is not covered by the filter values.
    if not pieces and use_symbol_filter and os.environ.get('SELECT_COIN_PARQUET_FILTER_FALLBACK', '1') != '0':
        pieces = _load_pieces(use_symbol_filter=False)

    if not pieces:
        return None

    extra_df = pd.concat(pieces, ignore_index=True, copy=False)
    extra_df.sort_values('candle_begin_time', inplace=True)
    extra_df = extra_df.groupby('candle_begin_time', as_index=False).last()

    merge_df = pd.DataFrame({'candle_begin_time': candle_time})
    merged_df = pd.merge(merge_df, extra_df, how='left', on='candle_begin_time')

    # 兼容时间锚点错位：例如行情在 xx:50，外部因子在 xx:00。
    # 当直接按时间戳合并后完全无值时，自动尝试“按小时内偏移秒数”做二次对齐。
    first_col = save_cols[0]
    direct_non_na = int(merged_df[first_col].notna().sum()) if first_col in merged_df.columns else 0
    if direct_non_na == 0 and not extra_df.empty:
        align_key = (str(base_path), factor_name)

        # 目标时间的主模态（秒级），例如 xx:50:00 -> 3000
        target_sec = (
            candle_time.dt.minute.astype(int) * 60
            + candle_time.dt.second.astype(int)
        )
        target_sec_mode = int(target_sec.mode().iloc[0]) if not target_sec.empty else 0

        # 目标数据频率（秒级），用于在 1H 场景下优先固定对齐到整点锚点
        candle_unique_time = candle_time.dropna().sort_values().drop_duplicates()
        if len(candle_unique_time) >= 2:
            target_step_sec = int(
                candle_unique_time.diff().dropna().dt.total_seconds().median()
            )
        else:
            target_step_sec = 0

        extra_time_index = extra_df['candle_begin_time']

        # 同一数据源+因子复用偏移，避免每个symbol重复搜索，保证一致性与性能
        if align_key in _FACTOR_TIME_ALIGNMENT_CACHE:
            best_delta_sec = _FACTOR_TIME_ALIGNMENT_CACHE[align_key]
            shifted_time = candle_time - pd.to_timedelta(best_delta_sec, unit='s')
            best_overlap = int(shifted_time.isin(extra_time_index).sum())
        else:
            # 外部因子的可选锚点（秒级），例如 [0, 900, 1800, 2700]
            extra_sec = (
                extra_df['candle_begin_time'].dt.minute.astype(int) * 60
                + extra_df['candle_begin_time'].dt.second.astype(int)
            )
            candidate_sec = sorted(set(extra_sec.dropna().astype(int).tolist()))

            best_overlap = -1
            best_delta_sec = 0
            cache_this_delta = False

            # 1H及以上频率：优先对齐到整点锚点，避免 15m 因子出现 +300/+1200/+3000s 漂移
            prefer_hour_anchor = target_step_sec >= 3600 and (0 in candidate_sec)
            if prefer_hour_anchor:
                preferred_delta = target_sec_mode
                shifted_time = candle_time - pd.to_timedelta(preferred_delta, unit='s')
                preferred_overlap = int(shifted_time.isin(extra_time_index).sum())
                if preferred_overlap > 0:
                    best_delta_sec = preferred_delta
                    best_overlap = preferred_overlap
                    cache_this_delta = True

            # 如果优先锚点不可用，再按重叠度搜索
            if best_overlap <= 0:
                for sec in candidate_sec:
                    delta_sec = target_sec_mode - sec
                    shifted_time = candle_time - pd.to_timedelta(delta_sec, unit='s')
                    overlap = int(shifted_time.isin(extra_time_index).sum())
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_delta_sec = delta_sec

                # 仅当重叠足够时缓存，避免被短历史币种“污染”
                min_cache_overlap = max(32, int(len(candle_time) * 0.2))
                cache_this_delta = best_overlap >= min_cache_overlap

            if cache_this_delta:
                _FACTOR_TIME_ALIGNMENT_CACHE[align_key] = best_delta_sec

        if best_overlap > 0:
            shifted_time = candle_time - pd.to_timedelta(best_delta_sec, unit='s')
            left_df = pd.DataFrame({
                '__row_id': np.arange(len(candle_time), dtype=np.int64),
                '__match_time': shifted_time,
            })
            right_df = extra_df.rename(columns={'candle_begin_time': '__match_time'})
            shifted_merged = pd.merge(left_df, right_df, how='left', on='__match_time')
            shifted_merged.sort_values('__row_id', inplace=True)
            shifted_merged.reset_index(drop=True, inplace=True)

            for col in save_cols:
                if col in shifted_merged.columns:
                    merged_df[col] = shifted_merged[col].values

            if os.environ.get('SELECT_COIN_LOG_FACTOR_TIME_ALIGN', '0') == '1':
                log_key = (str(base_path), factor_name)
                if log_key not in _FACTOR_TIME_ALIGNMENT_LOGGED:
                    logger.debug(
                        f'外部因子时间锚点自动偏移对齐: factor={factor_name}, '
                        f'delta={best_delta_sec:+d}s'
                    )
                    _FACTOR_TIME_ALIGNMENT_LOGGED.add(log_key)

    return merged_df[save_cols]
