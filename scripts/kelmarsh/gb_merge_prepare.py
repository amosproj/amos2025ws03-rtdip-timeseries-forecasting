import os, re, csv, argparse
from pathlib import Path
import pandas as pd

def detect_header_and_delim(path: Path):
    # detect delimiter from a small sample
    with open(path, 'r', errors='replace', newline='') as f:
        sample = f.read(4096)
        try:
            delim = csv.Sniffer().sniff(sample).delimiter
        except Exception:
            delim = ','
    # find the header line that starts with '# ' and contains column names
    header_idx, header = None, None
    with open(path, 'r', errors='replace', newline='') as f:
        for i, line in enumerate(f):
            if line.startswith('# ') and ('Date' in line or 'Timestamp' in line):
                header_idx = i
                header = [h.strip() for h in line[2:].strip().split(delim)]
                break
            if (not line.startswith('#')) and ('Timestamp' in line or 'Date' in line) and (delim in line):
                header_idx = i - 1
                header = [h.strip() for h in line.strip().split(delim)]
                break
    if header_idx is None or header is None:
        raise RuntimeError(f"Could not find header in {path}")
    return header_idx, header, delim

def clean_cols(cols):
    out = []
    for c in cols:
        c = c.replace('Â°', '°').replace('"', '')
        c = re.sub(r'\s+', ' ', c).strip()
        out.append(c)
    return out

def make_unique(cols):
    seen = {}
    out = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
    return out

def turbine_id_from_path(p: Path) -> int:
    m = re.search(r'Turbine_Data_Kelmarsh_(\d)_', p.name)
    return int(m.group(1)) if m else -1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base dir with Kelmarsh_SCADA_20xx_* folders")
    ap.add_argument("--year", type=int, help="Filter to this calendar year (optional)")
    args = ap.parse_args()

    base = Path(args.base)
    files = sorted(base.glob("Kelmarsh_SCADA_*_*/Turbine_Data_Kelmarsh_*_*.csv"))
    print(f"Found {len(files)} files")
    frames = []

    for f in files:
        hi, cols, sep = detect_header_and_delim(f)
        cols = clean_cols(cols)
        cols = make_unique(cols)

        df = pd.read_csv(
            f, sep=sep, skiprows=hi+1, names=cols, engine='python', on_bad_lines='skip'
        )
        # --- Coerce numeric-like columns ---
        for c in df.columns:
            if c == 'Date and time':
                continue
            if df[c].dtype == 'object':
                s = df[c].astype(str).str.strip()
                # normalize unicode minus/dash and thousands separators
                s = s.replace({'−': '-', '–': '-'}, regex=False)
                s = s.str.replace(',', '', regex=False)
                # try numeric; keep only if at least half the column converts
                s_num = pd.to_numeric(s, errors='coerce')
                if s_num.notna().sum() >= max(10, int(0.5 * len(s_num))):
                    df[c] = s_num
                else:
                    df[c] = s
        # -----------------------------------
        if 'Date and time' in df.columns:
            df['Date and time'] = pd.to_datetime(df['Date and time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df['item_id'] = turbine_id_from_path(f)
        frames.append(df)

    if not frames:
        print("No data frames were created. Exiting.")
        return

    df = pd.concat(frames, ignore_index=True)

    # drop rows where target is missing (Power is our target)
    target_col = 'Power (kW)'
    if target_col in df.columns:
        before = len(df)
        df = df.dropna(subset=[target_col])
        print(f"Dropped {before - len(df)} rows with missing '{target_col}'")

    # optional: filter by calendar year
    if args.year and 'Date and time' in df.columns:
        df = df[df['Date and time'].dt.year == args.year]
        print(f"Filtered to year {args.year}: {len(df)} rows")

    # sort by timestamp then turbine id
    if 'Date and time' in df.columns:
        df = df.sort_values(['Date and time', 'item_id'], kind='mergesort').reset_index(drop=True)

    # remove empty or constant columns
    all_null = [c for c in df.columns if df[c].isna().all()]
    constant = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    drop_1 = sorted(set(all_null + constant))
    df = df.drop(columns=drop_1, errors='ignore')

    # remove sparse (>=70% NaN)
    sparse = [c for c in df.columns if df[c].isna().mean() >= 0.70]
    df = df.drop(columns=sparse, errors='ignore')

    # time features
    if 'Date and time' in df.columns:
        ts = df['Date and time']
        df['year'] = ts.dt.year
        df['month'] = ts.dt.month
        df['dayofyear'] = ts.dt.day_of_year
        df['week'] = ts.dt.isocalendar().week
        df['dow'] = ts.dt.weekday
        df['hour'] = ts.dt.hour
        df['minute'] = ts.dt.minute

    # save outputs
    # --- Final dtype normalization before Parquet ---
    # Ensure no remaining 'object' columns with mixed types (strings + floats) confuse pyarrow
    obj_cols = df.select_dtypes(include=['object']).columns
    for c in obj_cols:
        s = df[c]
        # try to convert to numeric aggressively; if most entries convert, keep numeric
        s_num = pd.to_numeric(s, errors='coerce')
        if s_num.notna().sum() >= max(10, int(0.8 * len(s_num))):
            df[c] = s_num
        else:
            # keep as proper pandas string dtype (preserves NA), avoids mixed object
            df[c] = s.astype('string')

    # Ensure timestamp dtype is datetime (in case it drifted)
    if 'Date and time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date and time']):
        df['Date and time'] = pd.to_datetime(df['Date and time'], errors='coerce')
    # ------------------------------------------------
    out_dir = Path(os.environ['PROJ_WORK']) / "data/kelmarsh/processed"
    rep_dir = out_dir / "_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)
    tag = args.year if args.year else "all"

    out_parquet = out_dir / f"first_prepared_{tag}_scada.parquet"
    try:
        df.to_parquet(out_parquet, index=False)
    except Exception as e:
        print("Parquet write failed (likely missing pyarrow). Install and re-run:\n"
              "  conda install -y pyarrow  # or: pip install pyarrow")
        raise

    # small text report (good to commit)
    rep = rep_dir / f"first_prepared_{tag}_summary.txt"
    with open(rep, "w") as fh:
        fh.write(
            f"Rows: {len(df)}\n"
            f"Columns: {df.shape[1]}\n"
            f"item_id turbines: {df['item_id'].nunique()}\n"
            f"Time span: {df['Date and time'].min()} — {df['Date and time'].max()}\n"
            f"Dropped empty/constant: {len(drop_1)}\n"
            f"Dropped sparse(>=70% NaN): {len(sparse)}\n"
            f"Kept columns sample: {df.columns[:20].tolist()}\n"
        )
    print(f"Wrote {out_parquet}")
    print(f"Wrote report {rep}")

if __name__ == "__main__":
    main()
