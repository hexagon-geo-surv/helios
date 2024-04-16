#/bin/bash

# Activate the conda environment
source "$PREFIX/etc/profile.d/conda.sh"
conda activate "$PREFIX"

python -c "from menuinst.api import install; install('menu.json')"
