#!/bin/bash
# Start the DHWANI remote dashboard and a Cloudflare quick tunnel; print the link, the PIN and a QR.
#
#   ./start_remote.sh             LIVE: drive and mission buttons really move the rover
#   ./start_remote.sh --dry-run   TEST: every motion command goes to a *_dryrun topic instead
#
# The quick tunnel needs no Cloudflare account. Its address is random and changes every start,
# so the QR code is regenerated every time. Stop everything with ./stop_remote.sh.
# (no set -u: the ROS setup scripts reference unset variables)
cd "$(dirname "$0")"
MODE="${1:-}"
source /opt/ros/melodic/setup.bash
export ROS_MASTER_URI=http://localhost:11311 ROS_IP=127.0.0.1
./stop_remote.sh >/dev/null 2>&1
mkdir -p logs
rm -f current_link.txt qr.png qr_control.png

nohup env PYTHONPATH=/opt/ros/melodic/lib/python2.7/dist-packages python3 gateway.py $MODE \
  > logs/gateway.log 2>&1 &
echo $! > logs/gateway.pid
for i in $(seq 1 40); do curl -s -o /dev/null http://127.0.0.1:8088/api/info && break; sleep 1; done
curl -s -o /dev/null http://127.0.0.1:8088/api/info || { echo "gateway did not start; see logs/gateway.log"; tail -20 logs/gateway.log; exit 1; }

# --protocol http2: phone hotspots often block QUIC (UDP 7844); HTTP/2 goes over TCP 443 (28 Sep 2026)
nohup bin/cloudflared tunnel --url http://127.0.0.1:8088 --no-autoupdate --edge-ip-version 4 --protocol http2 \
  > logs/tunnel.log 2>&1 &
echo $! > logs/tunnel.pid
# The quick tunnel needs TCP/UDP 7844 to Cloudflare's edge. Some phone hotspots (c4, 28 Sep 2026)
# block it: then do not wait a minute for a link that cannot come up; cloudflared keeps retrying
# in the background and connects by itself if the rover moves to a network that allows it.
EDGE_OK=0
timeout 4 bash -c "</dev/tcp/region1.v2.argotunnel.com/7844" 2>/dev/null && EDGE_OK=1
URL=""
for i in $(seq 1 20); do
  URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' logs/tunnel.log | head -1)
  [ -n "$URL" ] && break
  sleep 1
done
if [ -n "$URL" ] && [ "$EDGE_OK" = 1 ]; then
  for i in $(seq 1 40); do grep -q "Registered tunnel connection" logs/tunnel.log && break; sleep 1; done
fi
PIN=${DHWANI_PIN:-$(cat pin.txt 2>/dev/null)}
LOCALS=$(hostname -I | tr ' ' '\n' | grep -v '^172\.17\.' | grep -v '^$' | sed 's#.*#http://&:8088/#' | tr '\n' ' ')
if [ -z "$URL" ] || ! grep -q "Registered tunnel connection" logs/tunnel.log; then
  [ "$EDGE_OK" = 1 ] || echo "Public link unavailable: this network blocks port 7844 (Cloudflare tunnel)."
  echo "  DHWANI remote is up on the local network$( [ "$MODE" = "--dry-run" ] && echo ' in TEST MODE (nothing moves)')"
  echo "  Local:     $LOCALS"
  [ -n "$URL" ] && echo "  Public (connects when the network allows): $URL/" && echo "$URL/" > current_link.txt
  echo "PIN: $PIN"
  exit 0
fi
echo "$URL/" > current_link.txt
PYTHONPATH=pylib python3 - "$URL/" "$PIN" <<'PY'
import sys, segno
url, pin = sys.argv[1], sys.argv[2]
q = segno.make(url, error="m")
q.save("qr.png", scale=10, border=2)
segno.make(url + "#pin=" + pin, error="m").save("qr_control.png", scale=10, border=2)
q.terminal(compact=True)
PY
echo
echo "  DHWANI remote is up$( [ "$MODE" = "--dry-run" ] && echo ' in TEST MODE (nothing moves)')"
echo "  Anywhere:  $URL/"
echo "  Local:     $LOCALS"
echo "  PIN:       $PIN      (driving and missions need it; STOP never does)"
echo "  QR codes:  $(pwd)/qr.png (view)   $(pwd)/qr_control.png (includes the PIN, share carefully)"
echo "  Someone must stay next to the rover whenever it is driven remotely."
