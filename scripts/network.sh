#!/usr/bin/env bash
set -euo pipefail
# These names and interfaces are exclusively owned by this POC.
case "${1:-up}" in
up)
  for ns in split-enterprise split-cloud; do
    if ! ip netns list | awk '{print $1}' | rg -qx "$ns"; then ip netns add "$ns"; fi
    ip -n "$ns" link set lo up
  done
  if ! ip link show split-host >/dev/null 2>&1; then
    ip link add split-host type veth peer name split-api
    ip link set split-api netns split-enterprise
    ip addr add 10.204.0.1/30 dev split-host
    ip link set split-host up
    ip -n split-enterprise addr add 10.204.0.2/30 dev split-api
    ip -n split-enterprise link set split-api up
  fi
  if ! ip -n split-enterprise link show split-e >/dev/null 2>&1; then
    ip link add split-e type veth peer name split-c
    ip link set split-e netns split-enterprise
    ip link set split-c netns split-cloud
    ip -n split-enterprise addr add 10.205.0.1/30 dev split-e
    ip -n split-cloud addr add 10.205.0.2/30 dev split-c
    ip -n split-enterprise link set split-e up
    ip -n split-cloud link set split-c up
  fi
  ;;
wan|local|check)
  exec python3 "$(dirname -- "${BASH_SOURCE[0]}")/network_state.py" "$@"
  ;;
down)
  # Caller must stop only POC processes first. Never touch the management NIC.
  for ns in split-cloud split-enterprise; do
    if [[ -n "$(ip netns pids "$ns" 2>/dev/null)" ]]; then echo "$ns still has processes"; exit 1; fi
    ip netns del "$ns" 2>/dev/null || true
  done
  ;;
*) echo 'network.sh up|wan [one-way-ms] [--delay-ms 5 --bandwidth-gbps 10]|local|check [--output PATH]|down'; exit 1;;
esac
