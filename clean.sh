#!/usr/bin/env bash

case $(uname) in
Linux) ECHO="echo -e" ;;
*) ECHO="echo" ;;
esac

if [[ ! -f $PWD/.triton_env ]]; then
  $ECHO "Already cleaned!"
  exit 0
else
  source $PWD/.triton_env
fi

# Removing the kernel.
kernel_dir=$(jupyter kernelspec list | grep coffea-triton | awk '{print $2}')
rm -rf ${kernel_dir}

storage_dir=$(readlink -f $PWD)

$ECHO "Removing the virtual environment ... "
rm -rf ${storage_dir}/${VENV_NAME}

$ECHO "Removing local ipython/jupyter files..."
rm -rf ${storage_dir}/.jupyter
rm -rf ${storage_dir}/.local/share/jupyter
rm -rf ${storage_dir}/.ipython
deactivate

# Unsetting environment variables
rm $PWD/.triton_env
unset VENV_NAME
unset COFFEA_IMAGE
unset PIP_FILE
unset MODE
