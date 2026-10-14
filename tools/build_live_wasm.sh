#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .tools/emsdk/emsdk_env.sh ]]; then
  source .tools/emsdk/emsdk_env.sh >/dev/null 2>&1
fi
em++ --version | head -1 | grep -F '6.0.12' >/dev/null || { echo 'Emscripten 6.0.12 required' >&2; exit 1; }
mkdir -p miniprogram/wasm
em++ core/live_api.cpp -std=c++17 -O3 -fwasm-exceptions --no-entry \
 -sSTANDALONE_WASM=1 -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=67108864 \
 -sMAXIMUM_MEMORY=134217728 -sSTACK_SIZE=1048576 \
 -sEXPORTED_FUNCTIONS='["_fc_create","_fc_destroy","_fc_input","_fc_result","_fc_row","_fc_reset","_fc_calibrate","_fc_start","_fc_arm","_fc_monitor","_fc_begin_measurement","_fc_stop_measurement","_fc_push","_fc_finish","_fc_interrupt","_fc_calibration_end"]' \
 -o miniprogram/wasm/fringe.wasm

chmod 644 miniprogram/wasm/fringe.wasm
