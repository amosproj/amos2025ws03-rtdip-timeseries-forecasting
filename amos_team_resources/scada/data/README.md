# SCADA Data Directory

## Download Data

From this directory, run:
```bash
bash download_ds.sh
```

Or download manually and place here.

## Manual Download

1. Download from: https://zenodo.org/records/5841834/files/Kelmarsh_SCADA_2020_3086.zip?download=1
2. Extract to `raw/scada_2020/`

For multiple years, download additional datasets from the same Zenodo repository.

## Expected Structure

```
data/
└── raw/
    └── scada_2020/
        ├── Turbine_Data_Kelmarsh_*.csv (6 files)
        └── Status_Kelmarsh_*.csv (6 files)
```
