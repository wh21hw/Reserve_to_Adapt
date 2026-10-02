#!/bin/sh
export PATH="/home/wanghao21/.local/opt/osda-node/bin:$PATH"
exec node --use-env-proxy /home/wanghao21/.local/opt/murphy-colab-cli/dist/index.js "$@"
