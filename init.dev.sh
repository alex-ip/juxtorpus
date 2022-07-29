#!/bin/zsh

set -e

VENV_DIR="./.venv"
REQ_FILE="./requirements.dev.txt"

echo "++ Initialising dev environment..."
if [[ -d $VENV_DIR ]]; then
  printf "-- Virtual environment $VENV_DIR already exists. Replace(y/n)? "
  read x
  [[ $x != 'y' ]] && echo "Exited." && exit 0
  rm -rf $VENV_DIR
fi

echo "++ Creating virtual env at $VENV_DIR..."
python3 -m venv $VENV_DIR

echo "++ Activating virtual env..."
source $VENV_DIR/bin/activate

set +e
echo "++ Installing dependencies..."
if [[ -f $REQ_FILE ]]; then
  pip install --upgrade pip
  pip install -r requirements.dev.txt
else
  echo "++ Missing $REQ_FILE! No dependencies installed."
fi

echo "Done."
