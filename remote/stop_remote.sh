#!/bin/bash
# Stop the dashboard and the tunnel. Uses pid files, never pkill -f (that can kill your own ssh).
cd "$(dirname "$0")"
for p in logs/tunnel.pid logs/gateway.pid; do
  [ -f "$p" ] && kill "$(cat $p)" 2>/dev/null
  rm -f "$p"
done
sleep 1
rm -f current_link.txt
echo "remote dashboard stopped"
