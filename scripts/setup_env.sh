#!/usr/bin/env bash
# One-shot environment setup for GeoDisaster-FM.
# Creates a fresh conda env and installs all runtime + dev deps.
set -euo pipefail

ENV_NAME="${ENV_NAME:-geodisaster}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

echo "[1/4] Creating conda env: $ENV_NAME (python=$PYTHON_VERSION)"
conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "[2/4] Installing geospatial system deps via conda-forge"
conda install -y -c conda-forge \
    gdal proj geos cartopy rasterio rioxarray geopandas shapely fiona pyproj \
    rtree networkx osmnx

echo "[3/4] Installing project (editable) + dev/fm extras"
pip install -e ".[dev,fm]"

echo "[4/4] Earth Engine auth (one-time, opens browser)"
echo "    earthengine authenticate"
echo "    earthengine set_project YOUR_GCP_PROJECT"

cat <<EOF

Done. Next steps:
  conda activate $ENV_NAME
  python -m geodisaster.cli list-events
  python -m geodisaster.cli download-gee --event jp_typhoon_hagibis_2019  # (P1)

EOF
