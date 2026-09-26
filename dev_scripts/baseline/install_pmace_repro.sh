#!/bin/bash
# Build the conda environment that runs the OLD ptycho_pmace code, so its
# results can be captured as the numerical baseline for xptycho.
#
# The old setup.py pins numpy 1.22 and scipy 1.8, which do not install on
# Apple silicon with Python 3.11, so the package is installed with
# --no-deps on top of pins that do.  scipy stays below 1.13 because the
# old code calls scipy.signal.tukey, removed in 1.13.
#
# Usage:  bash install_pmace_repro.sh [path to the ptycho_pmace checkout]

set -eo pipefail

NAME="pmace_repro"
PYTHON_VERSION="3.11"
PMACE_DIR="${1:-$HOME/Documents/GitHub/ptycho_pmace}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base
conda env remove -y -n "$NAME" 2>/dev/null || true
conda create -y -n "$NAME" python="$PYTHON_VERSION"
conda activate "$NAME"

pip install "numpy==1.26.4" "scipy==1.10.1" "pandas==2.2.*" "tifffile==2024.*" \
            imagecodecs "h5py==3.11.*" "pyfftw==0.14.0" "matplotlib==3.8.*" \
            "pyyaml==6.*" tqdm
pip install "bm4d==4.2.5" || echo "bm4d did not install; the capture scripts use the stub"
pip install --no-deps -e "$PMACE_DIR"

# The old module imports bm4d at import time.  When bm4d is absent or
# cannot import on this machine, the stub beside this script stands in;
# every baseline run uses add_reg=False, so bm4d is never called.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python - <<EOF
import sys
try:
    import bm4d
    print('bm4d imports')
except Exception as e:
    sys.path.insert(0, '$SCRIPT_DIR/bm4d_stub')
    import bm4d
    print('bm4d stub in use:', e)
import pmace.pmace, pmace.utils
import numpy, scipy, pyfftw
print('pmace imports; numpy', numpy.__version__, 'scipy', scipy.__version__,
      'pyfftw', pyfftw.__version__)
EOF

echo "Use:  conda activate $NAME"
