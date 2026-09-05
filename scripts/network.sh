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
wan)
  delay="${2:-5}"
  [[ "$delay" =~ ^[0-9]+$ ]] && (( delay <= 100 )) || { echo 'Delay must be 0..100 ms'; exit 1; }
  for pair in 'split-enterprise split-e' 'split-cloud split-c'; do
    read -r ns dev <<< "$pair"
    ip netns exec "$ns" tc qdisc replace dev "$dev" root handle 1: tbf rate 10gbit burst 2mb latency 100ms
    ip netns exec "$ns" tc qdisc replace dev "$dev" parent 1:1 handle 10: netem delay "${delay}ms" limit 100000
  done
  ;;
local)
  ip netns exec split-enterprise tc qdisc del dev split-e root 2>/dev/null || true
  ip netns exec split-cloud tc qdisc del dev split-c root 2>/dev/null || true
  ;;
check)
  ip netns exec split-enterprise ping -c 5 10.205.0.2
  ip netns exec split-cloud iperf3 -s -1 -D --logfile /tmp/split-poc-iperf.log
  ip netns exec split-enterprise iperf3 -c 10.205.0.2 -t 5 -P 4 -J
  ;;
down)
  # Caller must stop only POC processes first. Never touch the management NIC.
  for ns in split-cloud split-enterprise; do
    if [[ -n "$(ip netns pids "$ns" 2>/dev/null)" ]]; then echo "$ns still has processes"; exit 1; fi
    ip netns del "$ns" 2>/dev/null || true
  done
  ;;
*) echo 'network.sh up|wan [one-way-ms]|local|check|down'; exit 1;;
esac
