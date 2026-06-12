#!/usr/bin/env bash

# Run any command with nohup in the background.
# Usage:
#   source run_nohup.sh python my_script.py arg1 arg2
#   bash run_nohup.sh python my_script.py arg1 arg2

_is_sourced() {
  [[ "${BASH_SOURCE[0]}" != "$0" ]]
}

run_nohup() {
  if [[ $# -eq 0 ]]; then
    echo "Usage: source run_nohup.sh <command> [args ...]"
    return 1
  fi

  local timestamp cmd_name safe_cmd log_file pid
  timestamp="$(date +%Y%m%d_%H%M%S)"
  cmd_name="$(basename "$1")"
  safe_cmd="${cmd_name//[^a-zA-Z0-9._-]/_}"
  log_file="${NOHUP_LOG_FILE:-nohup_${safe_cmd}_${timestamp}.log}"

  nohup "$@" >"$log_file" 2>&1 &
  pid=$!

  echo "Started PID: $pid"
  echo "Log file: $log_file"
  echo "Tail logs: tail -f $log_file"
}

run_nohup "$@"
status=$?

if _is_sourced; then
  return "$status"
else
  exit "$status"
fi
