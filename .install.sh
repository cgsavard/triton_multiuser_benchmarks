#!/bin/bash

###############################################################################
#
# As singularity images need to run the installation scripts only after the
# images have completed spin-up. Installation will have to be handled by the
# .bashrc file for one-time setup. This mimics the bootstrap code used in the
# lpcjobsqueue setup with more options added for a variety of different setups.
#
###############################################################################
case $(uname) in
Linux) ECHO="echo -e" ;;
*) ECHO="echo" ;;
esac

if [[ ! -f .triton_env ]]; then
  $ECHO "The file .triton_env was not found!"
  $ECHO "Please run setup.sh first!"
  exit 1
else
  source "$PWD/.triton_env"
fi

make_virtual_env() {
  set -e # Exit immediately if any commands fail
  $ECHO "Installing shallow virtual environment in $PWD/${VENV_NAME}..."
  python3 -m venv --system-site-packages ${VENV_NAME}
  unlink ${VENV_NAME}/lib64
  set +e
}

make_kernel() {
  ## Creating the kernel with specified python kernel.
  # HTCondor can't transfer symlink to directory and it appears optional
  # work around issues copying CVMFS xattr when copying to tmpdir
  if [[ -z $(jupyter kernelspec list | grep coffea-triton) ]]; then
    $ECHO "installing jupyter kernel"
    export TMPDIR=$(mktemp -d -p .)
    $INSTALL_PYTHON -m ipykernel install \
      --user                             \
      --name coffea-triton               \
      --display-name "coffea for triton" \
      --env PYTHONPATH $PYTHONPATH:$PWD  \
      --env BASE $PWD                 \
      --env PYTHONNOUSERSITE 1
    rm -rf $TMPDIR && unset TMPDIR
  fi
}

install_pip_packages() {
  if [[ -z $($INSTALL_PYTHON -m pip list | grep lpcjobqueue) ]]; then
    $ECHO "Installing python packages specified in ${PIP_FILE}"
    $INSTALL_PYTHON -m pip install -r ${PIP_FILE} --no-cache-dir
  fi
}

# For IMAGE/LOCAL with a virtual evnrionment
if [[ $MODE == LOCAL ]]; then
  # Setting jupyter paths for local kernel install
  storage_dir=$(readlink -f $PWD)
  export JUPYTER_PATH=${storage_dir}/.jupyter
  export JUPYTER_RUNTIME_DIR=${storage_dir}/.local/share/jupyter/runtime
  export JUPYTER_DATA_DIR=${storage_dir}/.local/share/jupyter
  export IPYTHONDIR=${storage_dir}/.ipython
fi

if [[ ! -d $VENV_NAME ]]; then
  $ECHO "Installing virtual environment"
  make_virtual_env
fi
INSTALL_PYTHON=$VENV_NAME/bin/python
source $VENV_NAME/bin/activate
install_pip_packages
make_kernel
unset INSTALL_PYTHON

