#!/usr/bin/env bash

# Common setups
case $(uname) in
Linux) ECHO="echo -e" ;;
*) ECHO="echo" ;;
esac

# Setting up the options.
PIP_FILE_IMAGE="requirements.txt"
PIP_FILE_LOCAL="requirements_noimage.txt"

VENV_NAME="tritonenv"
COFFEA_IMAGE="coffeateam/coffea-dask:0.7.21-fastjet-3.4.0.1-gc3d707c"
PIP_FILE=$PIP_FILE_IMAGE
#PIP_FILE=$PIP_FILE_LOCAL
LCG_PATH="/cvmfs/sft.cern.ch/lcg/views/LCG_98python3/x86_64-centos7-gcc9-opt"
MODE="IMAGE"

usage() {
  EXIT=$1
  $ECHO "setup.sh [options]"
  $ECHO
  $ECHO "Options:"
  $ECHO "-m  Mode to setup (default: ${MODE}, valid: IMAGE, LCG, CASA, LOCAL)"
  $ECHO "-v  Name of the virtual environment (default=$VENV_NAME})"
  $ECHO "-r  The requirements.txt file used"
  $ECHO "    (default for IMAGE: ${PIP_FILE_IMAGE})"
  $ECHO "    (default for LOCAL:  ${PIP_FILE_LOCAL})"
  $ECHO "-i  cvmfs coffea-image file to use (only for IMAGE mode). "
  $ECHO "-h  Print message and exit"
  exit $EXIT
}

# check arguments
while getopts "hm:v:r:i:" opt; do
  case "$opt" in
  m)
    MODE=$OPTARG
    case $MODE in
    IMAGE)
      [[ -z $PIP_FILE ]] && PIP_FILE=$PIP_FILE_IMAGE
      ;;
    LOCAL)
      [[ -z $PIP_FILE ]] && PIP_FILE=$PIP_FILE_LOCAL
      ;;
    *)
      printf "Unsupported mode: %s\n" "$MODE" >&2
      usage -2
      ;;
    esac
    ;;
  v)
    VENV_NAME=$OPTARG
    ;;
  r)
    PIP_FILE=$OPTARG
    ;;
  i)
    COFFEA_IMAGE=$OPTARG
    ;;
  h)
    usage 0
    ;;
  :)
    printf "missing argument for -%s\n" "$OPTARG" >&2
    usage -1
    ;;
  \?)
    printf "illegal option: -%s\n" "$OPTARG" >&2
    usage -2
    ;;
  esac
done

# Making the environment variable file.
if [[ -f ".triton_env" ]]; then
  $ECHO "Warning overriding existing settings in .triton_env"
  printf "" >.triton_env
else
  $ECHO "Creating the .triton_env file"
fi

printf "export MODE=${MODE}\n" >>.triton_env
printf "export PIP_FILE=${PIP_FILE}\n" >>.triton_env
printf "export COFFEA_IMAGE=${COFFEA_IMAGE}\n" >>.triton_env
printf "export VENV_NAME=${VENV_NAME}\n" >>.triton_env

## Setting up the different modes
case $MODE in
IMAGE)
  $ECHO "Completed IMAGE based setup"
  $ECHO "To start the singularity container, run:"
  $ECHO ">  source ./init.sh"
  $ECHO "The first setup would include package installation and will take "
  $ECHO "a little longer"
  ;;
LOCAL)
  $ECHO "Running Local installation setup"
  ./.install.sh
  $ECHO "Completed the local setup"
  $ECHO "Start the virtual environment using the command: "
  $ECHO "> source ./init.sh"
  ;;
esac
