#!/bin/bash

# Download SCADA data to raw directory
# Run from: amos_team_resources/scada/data/

# Target directory
DATA_DIR="raw"
mkdir -p "$DATA_DIR"

# Download 2020 data
echo "Downloading SCADA 2020 data..."
curl -L -o "$DATA_DIR/scada_2020.zip" "https://zenodo.org/records/5841834/files/Kelmarsh_SCADA_2020_3086.zip?download=1"

# Unzip
echo "Extracting..."
unzip -q "$DATA_DIR/scada_2020.zip" -d "$DATA_DIR/scada_2020"

echo "Done! Data is in raw/scada_2020/"

# For additional years, download from:
# 2021: https://zenodo.org/records/5841834/files/Kelmarsh_SCADA_2021_3087.zip?download=1
# 2022: https://zenodo.org/records/5841834/files/Kelmarsh_SCADA_2022_3088.zip?download=1
