#!/bin/bash
# DHWANI: find the rover, start its dashboard, and say whether it is ready to go wireless.
#
#   Double-click this file in Finder, or run it in Terminal:
#     ./dhwani_start.command              find the rover (USB first), start the dashboard, open it
#     ./dhwani_start.command --restart    restart the dashboard even if it is already running
#     ./dhwani_start.command --dry-run    (re)start it in TEST MODE: every motion goes to a test topic
#     ./dhwani_start.command --status     only report; start and change nothing
#     ./dhwani_start.command --no-open    do not open the browser
#     ./dhwani_start.command --hide-pin   do not print the PIN (for screen recordings)
#     ./dhwani_start.command --install-key  one time: log in with this Mac's SSH key instead of the password
#
# How it finds the rover: the USB cable first (192.168.55.1), then the Wi-Fi address it saw last
# time (saved in ~/.dhwani_rover_ip), then jetracer.local, then the rover's own Wi-Fi DHWANI
# (10.42.0.1). "Ready for wireless" means this Mac can open the dashboard WITHOUT the cable.
# Starting the dashboard never moves the rover: driving and missions still need the PIN, and the
# on-board AI only drives after someone presses ARM.

USB_IP=192.168.55.1
AP_IP=10.42.0.1                 # the rover's own Wi-Fi network (SSID and password: set your own on the rover)
AP_SSID=${DHWANI_AP_SSID:-DHWANI}; AP_PASS=${DHWANI_AP_PASS:-<your AP password>}
CACHE="$HOME/.dhwani_rover_ip"
PORT=8088
USER_NAME=${DHWANI_SSH_USER:-jetson}; PASSWORD=${DHWANI_SSH_PASS:-}   # only needed until an SSH key is installed

RESTART=0; MODE=""; STATUS_ONLY=0; OPEN=1; INSTALL_KEY=0; HIDE_PIN=0
for a in "$@"; do
  case "$a" in
    --restart) RESTART=1 ;;
    --dry-run) MODE="--dry-run"; RESTART=1 ;;
    --status)  STATUS_ONLY=1 ;;
    --no-open) OPEN=0 ;;
    --hide-pin) HIDE_PIN=1 ;;
    --install-key) INSTALL_KEY=1 ;;
    -h|--help) sed -n 2,16p "$0"; exit 0 ;;
    *) echo "unknown option $a (try --help)"; exit 2 ;;
  esac
done

G=$'\033[32m'; R=$'\033[31m'; Y=$'\033[33m'; B=$'\033[1m'; N=$'\033[0m'
ok()   { echo "  ${G}OK${N}   $*"; }
bad()  { echo "  ${R}NO${N}   $*"; }
warn() { echo "  ${Y}!!${N}   $*"; }
info() { echo "       $*"; }

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=6"
# Run a command on the rover. Tries an SSH key first, then the password (the Nano sometimes refuses a
# password login while it is busy, so that is retried a few times).
rover() {
  local host="$1"; shift
  local rc
  ssh $SSH_OPTS -o BatchMode=yes "$USER_NAME@$host" "$@" 2>/dev/null; rc=$?
  [ $rc -ne 255 ] && return $rc          # 255 = could not log in with a key: fall back to the password
  if [ -z "$PASSWORD" ]; then
    echo "no SSH key login: set DHWANI_SSH_PASS once and run with --install-key" >&2
    return 255
  fi
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "sshpass is missing: brew install hudochenkov/sshpass/sshpass (or set up an SSH key)" >&2
    return 255
  fi
  local i
  for i in 1 2 3 4 5; do
    sshpass -p "$PASSWORD" ssh $SSH_OPTS "$USER_NAME@$host" "$@" 2>/dev/null; rc=$?
    [ $rc -ne 255 ] && return $rc
    sleep 3
  done
  return 255
}
pingable() { ping -c 1 -t 2 "$1" >/dev/null 2>&1; }
ssh_port()  { nc -z -G 3 "$1" 22 >/dev/null 2>&1; }
dash_up()   { [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 "http://$1:$PORT/api/info")" = "200" ]; }

echo
echo "${B}DHWANI rover${N}  $(date '+%d %b %Y %H:%M')"
echo

# ---------------------------------------------------------------- 1. find the rover
USB=0
if [ -n "$DHWANI_FORCE_WIFI" ]; then :       # testing: behave as if the cable were out
elif ifconfig 2>/dev/null | grep -q "inet 192\.168\.55\." && pingable "$USB_IP"; then
  USB=1; ok "USB cable connected (rover at $USB_IP)"
else
  info "USB cable: not connected"
fi

HOST=""; VIA=""
if [ $USB = 1 ]; then
  HOST=$USB_IP; VIA="USB"
else
  for cand in "$(cat "$CACHE" 2>/dev/null)" jetracer.local "$AP_IP"; do
    [ -z "$cand" ] && continue
    if ssh_port "$cand"; then HOST=$cand; VIA="Wi-Fi"; break; fi
  done
  if [ -n "$HOST" ]; then ok "found the rover over Wi-Fi at $HOST"; fi
fi
if [ -z "$HOST" ]; then
  bad "rover not found"
  info "Plug in the micro-USB cable (Jetson J28 port to this Mac), or join the rover's own"
  info "Wi-Fi \"$AP_SSID\" (password $AP_PASS), then run this again. Is the rover switched on?"
  exit 1
fi

if [ $INSTALL_KEY = 1 ]; then
  KEY=$(ls "$HOME"/.ssh/id_ed25519.pub "$HOME"/.ssh/id_rsa.pub 2>/dev/null | head -1)
  [ -z "$KEY" ] && { ssh-keygen -t ed25519 -N "" -f "$HOME/.ssh/id_ed25519" -q; KEY="$HOME/.ssh/id_ed25519.pub"; }
  if rover "$HOST" "mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && grep -qxF '$(cat "$KEY")' ~/.ssh/authorized_keys || echo '$(cat "$KEY")' >> ~/.ssh/authorized_keys"; then
    ok "SSH key installed on the rover ($KEY): no more password logins"
  else
    bad "could not install the SSH key"
  fi
fi

# ---------------------------------------------------------------- 2. is it booted?
printf "       waiting for the rover's software (ROS)"
READY=0
for i in $(seq 1 40); do
  if rover "$HOST" 'source /opt/ros/melodic/setup.bash; timeout 4 rostopic list >/dev/null 2>&1' ; then READY=1; break; fi
  printf "."; sleep 3
done
echo
if [ $READY = 0 ]; then
  bad "ROS is not up after 2 minutes. On the rover: screen -ls (the boot service starts it)."
  exit 1
fi
HEALTH=$(rover "$HOST" 'source /opt/ros/melodic/setup.bash
  echo "SCREENS=$(screen -ls 2>/dev/null | grep -c Detached)"
  echo "UPTIME=$(uptime -p 2>/dev/null)"
  echo "OAK=$(timeout 6 rostopic echo -n 1 /oak/status 2>/dev/null | sed -n "s/^data: //p" | tr -d \")"
  echo "LIVE=$(python -c "import json,os; print(json.load(open(os.path.expanduser(\"~/dhwani_executive.json\"))).get(\"live\", False))" 2>/dev/null)"
  echo "WLAN=$(ip -4 -o addr show wlan1 2>/dev/null | awk "{print \$4}" | cut -d/ -f1)"
  echo "WLAN_NET=$(iw dev wlan1 link 2>/dev/null | sed -n "s/^[[:space:]]*SSID: //p")"
  echo "AP=$(iw dev wlan0 info 2>/dev/null | sed -n "s/.*ssid //p")"
  echo "NET=$(ping -c 1 -W 2 1.1.1.1 >/dev/null 2>&1 && echo up || echo down)"
  echo "EDGE=$(timeout 4 bash -c "</dev/tcp/region1.v2.argotunnel.com/7844" 2>/dev/null && echo open || echo blocked)"
  echo "PIN=$(cat ~/dhwani_remote/pin.txt 2>/dev/null)"')
val() { echo "$HEALTH" | sed -n "s/^$1=//p" | head -1; }
SCREENS=$(val SCREENS); OAK=$(val OAK); LIVE=$(val LIVE); WLAN=$(val WLAN); WLAN_NET=$(val WLAN_NET)
AP=$(val AP); NET=$(val NET); EDGE=$(val EDGE); PIN=$(val PIN)
ok "rover is up ($(val UPTIME)), $SCREENS services running"
case "$OAK" in OK*) ok "OAK-D camera: $OAK" ;; "") warn "OAK-D camera: no status (still starting?)" ;; *) warn "OAK-D camera: $OAK" ;; esac
[ "$LIVE" = "True" ] && warn "on-board AI is in LIVE mode (it drives only after ARM on the dashboard)" \
                     || info "on-board AI: test mode (decides, never moves)"

# ---------------------------------------------------------------- 3. dashboard
echo
if [ $STATUS_ONLY = 1 ]; then
  dash_up "$HOST" && ok "dashboard is running" || bad "dashboard is not running (run without --status to start it)"
elif dash_up "$HOST" && [ $RESTART = 0 ]; then
  ok "dashboard already running"
else
  info "starting the dashboard${MODE:+ in TEST MODE}..."
  OUT=$(rover "$HOST" "cd ~/dhwani_remote && ./start_remote.sh $MODE 2>&1")
  if dash_up "$HOST"; then ok "dashboard started${MODE:+ (TEST MODE: nothing moves)}"
  else bad "dashboard did not start:"; echo "$OUT" | tail -8 | sed 's/^/       /'; exit 1; fi
fi
PUBLIC=$(rover "$HOST" 'cat ~/dhwani_remote/current_link.txt 2>/dev/null')

# ---------------------------------------------------------------- 4. ready for wireless?
echo
[ -n "$WLAN" ] && echo "$WLAN" > "$CACHE"
WIRELESS_URL=""
if [ -n "$WLAN" ] && pingable "$WLAN" && dash_up "$WLAN"; then
  WIRELESS_URL="http://$WLAN:$PORT/"
  ok "${B}READY FOR WIRELESS${N}: this Mac reaches the rover over Wi-Fi \"${WLAN_NET:-hotspot}\""
  [ $USB = 1 ] && info "You can unplug the USB cable now."
elif pingable "$AP_IP" && dash_up "$AP_IP"; then
  WIRELESS_URL="http://$AP_IP:$PORT/"
  ok "${B}READY FOR WIRELESS${N}: this Mac is on the rover's own Wi-Fi \"$AP_SSID\""
  [ $USB = 1 ] && info "You can unplug the USB cable now."
else
  bad "${B}NOT ready for wireless${N}: this Mac cannot reach the dashboard without the cable"
  if [ -n "$WLAN" ]; then
    info "The rover is on Wi-Fi \"${WLAN_NET:-?}\" at $WLAN. Connect this Mac to \"${WLAN_NET:-that network}\" too,"
  else
    info "The rover is not on any hotspot right now (wlan1 has no address)."
  fi
  [ -n "$AP" ] && info "or join the rover's own Wi-Fi \"$AP\" (password $AP_PASS). Then run this again." \
               || warn "the rover's own Wi-Fi \"$AP_SSID\" is not up either"
  [ $USB = 1 ] && info "Keep the USB cable in until it says READY."
fi
if [ "$NET" = up ] && [ "$EDGE" = open ] && [ -n "$PUBLIC" ]; then
  ok "public link (anywhere): $PUBLIC"
elif [ "$NET" = up ]; then
  info "public link: not available here (this network blocks Cloudflare's port 7844); local links work"
else
  info "public link: the rover has no internet right now; local links work"
fi

# ---------------------------------------------------------------- 5. open it
URL=${WIRELESS_URL:-http://$HOST:$PORT/}
echo
echo "  ${B}Dashboard:${N} $URL"
[ -n "$WIRELESS_URL" ] && [ $USB = 1 ] && [ "$URL" != "http://$USB_IP:$PORT/" ] && echo "  (over the cable: http://$USB_IP:$PORT/)"
if [ -n "$PIN" ] && [ $HIDE_PIN = 0 ]; then echo "  ${B}PIN:${N}       $PIN   (needed to drive or start a mission; STOP never needs it)"
elif [ -n "$PIN" ]; then echo "  ${B}PIN:${N}       hidden (--hide-pin); needed to drive or start a mission, STOP never needs it"; fi
echo "  Someone must stay next to the rover whenever it drives."
echo
if [ $OPEN = 1 ] && [ $STATUS_ONLY = 0 ]; then open "$URL"; fi
