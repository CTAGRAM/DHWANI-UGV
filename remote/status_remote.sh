#!/bin/bash
cd "$(dirname "$0")"
alive() { [ -f "$1" ] && kill -0 "$(cat $1)" 2>/dev/null && echo up || echo down; }
echo "gateway: $(alive logs/gateway.pid)   tunnel: $(alive logs/tunnel.pid)"
echo "link:    $(cat current_link.txt 2>/dev/null || echo none)"
echo "PIN:     $(cat pin.txt 2>/dev/null)"
grep -o '"dry_run": [a-z]*' <(curl -s http://127.0.0.1:8088/api/info) 2>/dev/null
