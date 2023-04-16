#!/bin/bash

PORT=$1
if [ -z "$PORT" ]; then
	PORT=8888
fi

# generate 32 bit random hex string, # Solution updated for singularity
# https://stackoverflow.com/questions/34328759/how-to-get-a-random-string-of-32-hexadecimal-digits-through-command-line/34329057
TOKEN=$(od -vN "16" -An -tx1 /dev/urandom | tr -d " \n")

echo "Server url: http://127.0.0.1:${PORT}/?token=${TOKEN}"
echo ""

jupyter nbclassic --ip 0.0.0.0 --port ${PORT} \
	--no-browser --notebook-dir . \
	--NotebookApp.token=$TOKEN
