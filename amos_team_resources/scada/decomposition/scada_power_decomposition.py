"""
SCADA Power Output Decomposition Script

Performs MSTL decomposition on wind turbine power output data.
Can analyze individual turbines or average across all turbines.

Usage:
    python scada_power_decomposition.py --turbine_id all
    python scada_power_decomposition.py --turbine_id 1 --patterns daily weekly
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add SDK to path
SDK_PATH = Path(__file__).resolve().parents[3] / "src" / "sdk" / "python"
sys.path.insert(0, str(SDK_PATH))

from rtdip_sdk.pipelines.decomposition.pandas import MSTLDecomposition
from rtdip_sdk.pipelines.visualization.matplotlib.decomposition import MSTLDecompositionPlot


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MSTL decomposition for SCADA turbine power data"
    )

    # Data selection
    parser.add_argument(
        "--turbine_id",
        type=str,
        default="all",
        help="Turbine ID to analyze ('all' for averaged, or '1'-'6' for individual turbine). Default: 'all'"
    )

    # Decomposition parameters
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["daily", "weekly", "monthly"],
        choices=["daily", "weekly", "monthly", "yearly"],
        help="Seasonality patterns to extract. Default: daily weekly monthly"
    )

    parser.add_argument(
        "--use_robust",
        action="store_true",
        help="Use robust MSTL mode (resistant to outliers). Default: False"
    )

    # Preprocessing parameters
    parser.add_argument(
        "--cap_outliers",
        action="store_true",
        help="Cap extreme outliers using std deviation + percentile method. Default: False"
    )

    parser.add_argument(
        "--outlier_threshold",
        type=float,
        default=20.0,
        help="Number of standard deviations for outlier detection. Default: 20.0"
    )

    parser.add_argument(
        "--interpolation_limit",
        type=int,
        default=None,
        help="Maximum gap size to interpolate (in minutes). Default: None (interpolate all gaps)"
    )

    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_images",
        help="Directory to save output images and data. Default: 'output_images'"
    )

    parser.add_argument(
        "--use_preprocessed",
        action="store_true",
        default=True,
        help="Use preprocessed data from preprocessing folder. Default: True"
    )

    return parser.parse_args()


def load_data(use_preprocessed=True):
    """Load SCADA data from preprocessed file or raw CSV files."""
    if use_preprocessed:
        print("Loading preprocessed data...")
        preprocessed_file = Path(__file__).parent.parent / "preprocessing" / "scada_prepro.parquet"

        if not preprocessed_file.exists():
            raise FileNotFoundError(
                f"Preprocessed file not found: {preprocessed_file}\n"
                "Run the preprocessing notebook first or use --no-use_preprocessed"
            )

        df = pd.read_parquet(preprocessed_file, columns=['item_id', 'timestamp', 'target'])

        print(f"Loaded {len(df):,} rows")
        print(f"Turbines: {sorted(df['item_id'].unique())}")

    else:
        print("Loading raw CSV files...")
        data_dir = Path(__file__).parent.parent / "data" / "raw" / "scada_2020"

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        csv_files = sorted(data_dir.glob("Turbine_Data_Kelmarsh_*.csv"))
        print(f"Found {len(csv_files)} turbine data files")

        dfs = []
        for i, file in enumerate(csv_files, 1):
            print(f"Reading: {file.name}")
            df_temp = pd.read_csv(
                file,
                skiprows=9,
                header=0,
                low_memory=False,
                on_bad_lines='skip'
            )
            df_temp.columns = [c.lstrip('# ').strip() for c in df_temp.columns]
            df_temp = df_temp.rename(columns={'Date and time': 'timestamp', 'Power (kW)': 'target'})
            df_temp = df_temp[['timestamp', 'target']]
            df_temp['item_id'] = f'{i}_Kelmarsh'
            dfs.append(df_temp)

        df = pd.concat(dfs, ignore_index=True)
        print(f"\nLoaded {len(df):,} rows")
        print(f"Turbines: {sorted(df['item_id'].unique())}")

    return df


def select_and_aggregate(df, turbine_id):
    """Select turbine(s) and aggregate if needed."""
    if turbine_id.lower() == 'all':
        print("\nAveraging all turbines")
        df_selected = df.groupby('timestamp', as_index=False)['target'].mean()
        print(f"Averaged data: {len(df_selected):,} timestamps from {df['item_id'].nunique()} turbines")
    else:
        print(f"\nFiltering to turbine: {turbine_id}")
        item_id_to_select = f"{turbine_id}_Kelmarsh"
        df_selected = df[df['item_id'] == item_id_to_select].copy()
        df_selected = df_selected[['timestamp', 'target']]
        print(f"Selected turbine has {len(df_selected):,} rows")

    return df_selected


def preprocess_data(df):
    """Preprocess data: validate timestamps, remove nulls, sort."""
    print("\nPreprocessing...")

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    invalid_times = df['timestamp'].isna().sum()
    if invalid_times > 0:
        df = df.dropna(subset=['timestamp'])
        print(f"Dropped {invalid_times:,} invalid timestamps")

    df['target'] = pd.to_numeric(df['target'], errors='coerce')

    before_len = len(df)
    df = df[df['target'].notna()].copy()
    removed = before_len - len(df)
    print(f"Removed {removed:,} null target values ({removed/before_len*100:.2f}%)")
    print(f"Remaining: {len(df):,} rows")

    df = df.sort_values('timestamp').reset_index(drop=True)

    duration = df['timestamp'].max() - df['timestamp'].min()
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Duration: {duration.days} days ({duration.days/365.25:.2f} years)")

    return df


def resample_data(df, resample_freq='10min'):
    """Resample data to regular intervals."""
    print(f"\nResampling to {resample_freq} intervals...")

    df_resampled = df.set_index('timestamp').resample(resample_freq)['target'].mean().reset_index()

    missing_count = df_resampled['target'].isna().sum()
    print(f"Original points: {len(df):,}")
    print(f"Resampled to: {len(df_resampled):,} intervals")
    print(f"Missing values: {missing_count:,} ({missing_count/len(df_resampled)*100:.2f}%)")

    return df_resampled


def interpolate_data(df, interpolation_limit=None):
    """Interpolate missing values."""
    print("\nInterpolating missing values...")

    missing_before = df['target'].isna().sum()
    print(f"Missing values before interpolation: {missing_before:,}")

    if interpolation_limit:
        limit_intervals = int(interpolation_limit / 10)
        print(f"Interpolation limit: {interpolation_limit} minutes ({limit_intervals} intervals)")
    else:
        limit_intervals = None
        print("Interpolation limit: None (all gaps)")

    df_interpolated = df.set_index('timestamp')
    df_interpolated['target'] = df_interpolated['target'].interpolate(
        method='time',
        limit=limit_intervals,
        limit_direction='both'
    )
    df_interpolated = df_interpolated.reset_index()

    missing_after = df_interpolated['target'].isna().sum()
    interpolated_count = missing_before - missing_after

    print(f"Interpolated: {interpolated_count:,} values ({interpolated_count/len(df_interpolated)*100:.2f}%)")

    if missing_after > 0:
        print(f"Removing remaining {missing_after:,} NaN values...")
        df_interpolated = df_interpolated.dropna(subset=['target']).reset_index(drop=True)

    return df_interpolated, interpolated_count


def cap_outliers(df, cap_outliers, outlier_threshold):
    """Cap extreme outliers if enabled."""
    outlier_metadata = {'method': 'none', 'outliers_capped': 0}

    if cap_outliers:
        print("\nChecking for extreme outliers...")

        mean = df['target'].mean()
        std = df['target'].std()

        lower_extreme = mean - outlier_threshold * std
        upper_extreme = mean + outlier_threshold * std

        extreme_mask = (df['target'] < lower_extreme) | (df['target'] > upper_extreme)
        extreme_count = extreme_mask.sum()

        print(f"Extreme outlier threshold ({outlier_threshold} std):")
        print(f"  Lower: {lower_extreme:.4f}")
        print(f"  Upper: {upper_extreme:.4f}")
        print(f"Found {extreme_count} extreme outliers ({extreme_count/len(df)*100:.3f}%)")

        if extreme_count > 0:
            q99_9 = df['target'].quantile(0.999)
            q00_1 = df['target'].quantile(0.001)

            cap_lower = (df['target'] < lower_extreme) & (df['target'] < q00_1)
            cap_upper = (df['target'] > upper_extreme) & (df['target'] > q99_9)
            outliers_total = cap_lower.sum() + cap_upper.sum()

            print(f"Capping thresholds:")
            print(f"  Lower (0.1st percentile): {q00_1:.4f}")
            print(f"  Upper (99.9th percentile): {q99_9:.4f}")
            print(f"Values to cap: {outliers_total}")

            if outliers_total > 0:
                df['target'] = df['target'].clip(lower=q00_1, upper=q99_9)
                print(f"After capping: Min={df['target'].min():.4f}, Max={df['target'].max():.4f}")

                outlier_metadata = {
                    'method': 'std_deviation + percentile',
                    'std_threshold': outlier_threshold,
                    'lower_percentile': 0.001,
                    'upper_percentile': 0.999,
                    'outliers_capped': int(outliers_total)
                }

    return df, outlier_metadata


def decompose_data(df, patterns, use_robust):
    """Perform MSTL decomposition."""
    print("\nApplying MSTL Decomposition...")
    print("="*80)

    # Calculate periods (10-minute intervals)
    daily_period = 6 * 24  # 144 intervals per day
    weekly_period = 6 * 24 * 7  # 1008 intervals per week
    monthly_period = 6 * 24 * 30  # ~4320 intervals per month
    yearly_period = 6 * 24 * 365  # ~52560 intervals per year

    pattern_map = {
        'daily': daily_period,
        'weekly': weekly_period,
        'monthly': monthly_period,
        'yearly': yearly_period
    }

    periods = []
    pattern_names = []

    for pattern in patterns:
        if pattern not in pattern_map:
            raise ValueError(f"Unknown pattern '{pattern}'. Valid: {list(pattern_map.keys())}")

        period = pattern_map[pattern]
        min_required = period * 2

        if len(df) < min_required:
            print(f"Warning: Insufficient data for {pattern} pattern")
            print(f"  Need: {min_required:,} points, Have: {len(df):,} points")
            print(f"  Skipping {pattern} pattern")
            continue

        periods.append(period)
        pattern_names.append(pattern)

    if len(periods) == 0:
        raise ValueError("No valid patterns available for decomposition")

    print(f"Using patterns: {', '.join(pattern_names)}")
    print(f"Data points: {len(df):,}")
    print(f"Periods: {periods}")
    print(f"Robust mode: {use_robust}")

    try:
        print("\nCreating MSTL decomposer...")

        mstl = MSTLDecomposition(
            df=df,
            value_column='target',
            timestamp_column='timestamp',
            periods=periods,
            iterate=2,
            stl_kwargs={'robust': use_robust}
        )

        print("Decomposing...")
        df_decomposed = mstl.decompose()

        print("\nDecomposition successful")
        print(f"Shape: {df_decomposed.shape}")
        print(f"Components: {[col for col in df_decomposed.columns if col not in ['timestamp', 'target']]}")

        return df_decomposed, pattern_names, pattern_map

    except Exception as e:
        print(f"\nDecomposition failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def visualize_before_decomposition(df, turbine_id, output_dir):
    """Visualize data before decomposition."""
    print("\nCreating before-decomposition visualization...")

    fig, ax = plt.subplots(figsize=(16, 5))

    ax.plot(df['timestamp'], df['target'], linewidth=0.5, alpha=0.7)
    if turbine_id.lower() == 'all':
        title = "Average Power Output Across All Turbines"
    else:
        title = f"Turbine {turbine_id} - Power Output"
    ax.set_title(f"{title} (10-minute resampled, interpolated)")
    ax.set_xlabel('Time')
    ax.set_ylabel('Power (kW)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'data_before_decomposition.png'
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"Saved: {output_file}")


def visualize_decomposition(df_decomposed, turbine_id, output_dir):
    """Visualize decomposition results using SDK component."""
    print("\nCreating decomposition visualization...")

    if turbine_id.lower() == 'all':
        file_suffix = "all_turbines_average"
        sensor_id = "Average of All Turbines"
    else:
        file_suffix = f"turbine_{turbine_id}"
        sensor_id = f"Turbine {turbine_id}"

    plot = MSTLDecompositionPlot(
        decomposition_data=df_decomposed,
        timestamp_column='timestamp',
        value_column='target',
        sensor_id=sensor_id
    )

    fig = plot.plot()

    filename = output_dir / f"decomposition_{file_suffix}.png"
    plot.save(filename, dpi=150)
    plt.close()

    print(f"Saved: {filename}")


def visualize_zoom(df_decomposed, turbine_id, output_dir, zoom_days=14):
    """Visualize zoomed view of first N days."""
    print(f"\nCreating {zoom_days}-day zoom visualization...")

    start = df_decomposed['timestamp'].min()
    end = start + pd.Timedelta(days=zoom_days)

    df_zoom = df_decomposed[df_decomposed['timestamp'] <= end].copy()

    if turbine_id.lower() == 'all':
        sensor_id_zoom = f"Average of All Turbines (First {zoom_days} Days)"
    else:
        sensor_id_zoom = f"Turbine {turbine_id} (First {zoom_days} Days)"

    plot_zoom = MSTLDecompositionPlot(
        decomposition_data=df_zoom,
        timestamp_column='timestamp',
        value_column='target',
        sensor_id=sensor_id_zoom
    )

    fig = plot_zoom.plot()
    output_file = output_dir / 'patterns_14days.png'
    plot_zoom.save(output_file, dpi=150)
    plt.close()

    print(f"Saved: {output_file}")


def calculate_variance_analysis(df_decomposed, pattern_names, pattern_map):
    """Calculate variance explained by each component."""
    print("\nVariance Analysis")
    print("="*80)

    seasonal_cols = [col for col in df_decomposed.columns if col.startswith('seasonal_')]

    total_var = df_decomposed['target'].var()
    trend_var = df_decomposed['trend'].dropna().var()
    residual_var = df_decomposed['residual'].dropna().var()

    print(f"\nTotal variance: {total_var:.2f}")
    print()
    print(f"Breakdown:")
    print(f"  Trend:          {trend_var:>12.2f} ({trend_var/total_var*100:>6.2f}%)")

    variance_dict = {
        'trend_pct': float(trend_var / total_var * 100),
        'residual_pct': float(residual_var / total_var * 100)
    }

    combined_seasonal = 0
    for col, pattern_name in zip(seasonal_cols, pattern_names):
        seasonal_var = df_decomposed[col].dropna().var()
        combined_seasonal += seasonal_var
        period = int(col.split('_')[1])
        variance_dict[f'seasonal_{period}_pct'] = float(seasonal_var / total_var * 100)
        print(f"  {pattern_name.capitalize()} Seasonal: {seasonal_var:>12.2f} ({seasonal_var/total_var*100:>6.2f}%)")

    print(f"  Residual:       {residual_var:>12.2f} ({residual_var/total_var*100:>6.2f}%)")
    print()
    print(f"  Combined Seasonal: {combined_seasonal:>12.2f} ({combined_seasonal/total_var*100:>6.2f}%)")

    variance_dict['combined_seasonal_pct'] = float(combined_seasonal / total_var * 100)

    return variance_dict


def save_results(df_decomposed, turbine_id, pattern_names, pattern_map,
                interpolated_count, interpolation_limit, outlier_metadata,
                variance_dict, output_dir):
    """Save decomposition results and metadata."""
    print("\nSaving results...")
    print("="*80)

    if turbine_id.lower() == 'all':
        file_suffix = "all_turbines_average"
    else:
        file_suffix = f"turbine_{turbine_id}"

    parquet_file = output_dir / f"decomposition_{file_suffix}.parquet"
    df_decomposed.to_parquet(parquet_file, index=False)
    print(f"Saved: {parquet_file}")

    seasonal_cols = [col for col in df_decomposed.columns if col.startswith('seasonal_')]
    periods = [int(col.split('_')[1]) for col in seasonal_cols]

    metadata = {
        'turbine_id': turbine_id,
        'analysis_type': 'averaged' if turbine_id.lower() == 'all' else 'single_turbine',
        'duration_days': float((df_decomposed['timestamp'].max() - df_decomposed['timestamp'].min()).days),
        'data_points': len(df_decomposed),
        'resampling': {
            'frequency': '10 minutes',
            'aggregation': 'mean'
        },
        'interpolation': {
            'method': 'time',
            'limit': interpolation_limit if interpolation_limit else 'none',
            'values_interpolated': int(interpolated_count)
        },
        'outlier_handling': outlier_metadata,
        'patterns': pattern_names,
        'periods': {f'period_{p}': p for p in periods},
        'time_range': {
            'start': str(df_decomposed['timestamp'].min()),
            'end': str(df_decomposed['timestamp'].max())
        },
        'variance_explained': variance_dict
    }

    metadata_file = output_dir / f"decomposition_{file_suffix}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {metadata_file}")


def main():
    """Main execution function."""
    args = parse_arguments()

    print("SCADA Power Decomposition Script")
    print("="*80)
    print(f"Configuration:")
    print(f"  Turbine ID: {args.turbine_id}")
    print(f"  Patterns: {', '.join(args.patterns)}")
    print(f"  Robust mode: {args.use_robust}")
    print(f"  Cap outliers: {args.cap_outliers}")
    print(f"  Output directory: {args.output_dir}")
    print("="*80)

    # Create output directory relative to script location
    script_dir = Path(__file__).parent
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir.absolute()}")

    # Load and preprocess data
    df = load_data(args.use_preprocessed)
    df_selected = select_and_aggregate(df, args.turbine_id)
    df_preprocessed = preprocess_data(df_selected)
    df_resampled = resample_data(df_preprocessed)
    df_interpolated, interpolated_count = interpolate_data(df_resampled, args.interpolation_limit)
    df_final, outlier_metadata = cap_outliers(df_interpolated, args.cap_outliers, args.outlier_threshold)

    # Visualize before decomposition
    visualize_before_decomposition(df_final, args.turbine_id, output_dir)

    # Decompose
    df_decomposed, pattern_names, pattern_map = decompose_data(
        df_final,
        args.patterns,
        args.use_robust
    )

    if df_decomposed is None:
        print("\nDecomposition failed. Exiting.")
        return 1

    # Variance analysis
    variance_dict = calculate_variance_analysis(df_decomposed, pattern_names, pattern_map)

    # Visualize results
    visualize_decomposition(df_decomposed, args.turbine_id, output_dir)
    visualize_zoom(df_decomposed, args.turbine_id, output_dir)

    # Save results
    save_results(
        df_decomposed,
        args.turbine_id,
        pattern_names,
        pattern_map,
        interpolated_count,
        args.interpolation_limit,
        outlier_metadata,
        variance_dict,
        output_dir
    )

    print("\nDecomposition complete!")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
