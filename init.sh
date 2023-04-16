#!/bin/bash

if [[ ! -f .triton_env ]]; then
  echo "The file .triton_env was not found!"
  echo "Please run setup.sh first!"
else
  source "$PWD/.triton_env"
fi

case $MODE in
IMAGE)
  export JUPYTER_PATH=/srv/.jupyter
  export JUPYTER_RUNTIME_DIR=/srv/.local/share/jupyter/runtime
  export JUPYTER_DATA_DIR=/srv/.local/share/jupyter
  export IPYTHONDIR=/srv/.ipython
  export PYTHONPATH=/srv
  export BASE=/srv
  
  # Setting the condor configuration
  grep -v '^include' /etc/condor/config.d/01_cmslpc_interactive > .condor_config

  # Running the singularity image
  singularity exec -p -B ${PWD}:/srv -B /uscmst1b_scratch --pwd /srv \
    /cvmfs/unpacked.cern.ch/registry.hub.docker.com/${COFFEA_IMAGE} \
    /bin/bash --rcfile /srv/.install.sh
  ;;
LOCAL)
  source ${VENV_NAME}/bin/activate
  ;;
esac
