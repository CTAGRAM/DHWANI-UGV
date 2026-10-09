/* DHWANI Remote: browser side. No libraries, so it loads on any phone even when the rover's
   internet is slow. Everything arrives over one WebSocket (/ws) from gateway.py on the rover. */
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var S = {
    ws: null, open: false, retry: 0, rtt: null, token: null, name: null,
    controller: null, amController: false, halted: false, dry: false, limits: { v: 0.3, w: 1.2 },
    pose: null, trail: [], plan: [], scan: [], junctions: [], events: [], exec: null, missionState: "?",
    map: null, mapImg: null, publicUrl: null, lastParsed: null, reqId: 0, execAlive: false
  };
  try { S.token = sessionStorage.getItem("dh_token"); S.name = sessionStorage.getItem("dh_name"); } catch (e) {}

  /* ------------------------------------------------------------------ toasts */
  function toast(msg, level) {
    var t = document.createElement("div");
    t.className = "toast " + (level || "");
    t.textContent = msg;
    $("toasts").appendChild(t);
    setTimeout(function () { t.remove(); }, level === "stop" ? 6000 : 3800);
  }

  /* ------------------------------------------------------------------ socket */
  function connect() {
    var proto = location.protocol === "https:" ? "wss://" : "ws://";
    var ws = new WebSocket(proto + location.host + "/ws");
    ws.binaryType = "arraybuffer";
    S.ws = ws;
    setLink("warn", "Connecting");
    ws.onopen = function () {
      S.open = true; S.retry = 0;
      setLink("ok", "Live");
      sendCamera();
      sendDepth();
      var h = location.hash.match(/pin=(\d{4,8})/);
      if (h && !S.token) auth(S.name || "operator", h[1]);
    };
    ws.onclose = function () {
      S.open = false; S.amController = false; releasePad();
      setLink("bad", "Reconnecting");
      var d = Math.min(8000, 800 * Math.pow(2, S.retry++));
      setTimeout(connect, d);
    };
    ws.onerror = function () { try { ws.close(); } catch (e) {} };
    ws.onmessage = function (ev) {
      if (typeof ev.data !== "string") { onCamera(ev.data); return; }
      var d; try { d = JSON.parse(ev.data); } catch (e) { return; }
      var h = HANDLERS[d.t]; if (h) h(d);
    };
  }
  function send(o) {
    if (!S.open) return false;
    if (S.token && !o.token) o.token = S.token;
    try { S.ws.send(JSON.stringify(o)); return true; } catch (e) { return false; }
  }
  function setLink(level, txt) {
    var c = $("chip-link");
    c.className = "chip " + level;
    $("link-txt").textContent = txt + (S.rtt != null && level === "ok" ? " " + Math.round(S.rtt) + " ms" : "");
  }
  setInterval(function () { if (S.open) send({ t: "ping", ts: performance.now(), rtt: S.rtt }); }, 2000);

  /* ------------------------------------------------------------------ handlers */
  var HANDLERS = {
    hello: function (d) {
      S.dry = !!d.dry_run; S.limits = d.limits || S.limits; S.publicUrl = d.public_url; S.myId = d.you;
      $("testbar").hidden = !S.dry;
      $("vmax").textContent = S.limits.v.toFixed(2);
      setParserKind(d.parser);
    },
    pong: function (d) { S.rtt = performance.now() - d.ts; setLink("ok", "Live"); G.push("rtt", S.rtt); },
    map: function (d) {
      var img = new Image();
      img.onload = function () { S.map = d; S.mapImg = img; if (!V.fitted) fitView(); dirty(); };
      img.src = "data:image/png;base64," + d.png;
    },
    pose: function (d) {
      S.pose = d;
      var last = S.trail[S.trail.length - 1];
      if (!last || Math.hypot(d.x - last[0], d.y - last[1]) > 0.08) { S.trail.push([d.x, d.y]); if (S.trail.length > 3000) S.trail.shift(); }
      $("spd").textContent = d.v.toFixed(2) + " m/s";
      $("trn").textContent = d.w.toFixed(2) + " rad/s";
      $("readout").textContent = "x " + d.x.toFixed(2) + " m   y " + d.y.toFixed(2) + " m   " + Math.round(d.yaw * 180 / Math.PI) + "°";
      if (V.follow) { V.cx = d.x; V.cy = d.y; }
      dirty();
    },
    trail: function (d) { S.trail = d.pts || []; dirty(); },
    scan: function (d) { S.scan = d.pts || []; dirty(); },
    plan: function (d) { S.plan = d.pts || []; dirty(); },
    junctions: function (d) { S.junctions = d.pts || []; dirty(); },
    events: function (d) { S.events = d.items || []; renderEvents(); dirty(); },
    event: function (d) {
      S.events.push(d.event); if (S.events.length > 60) S.events.shift(); renderEvents(); dirty();
      var k = d.event.kind;
      if (k === "PERSON") toast("Person detected: " + (d.event.text || ""), "stop");
      else if (k === "HALT") toast("Rover halted: " + (d.event.text || ""), "warn");
    },
    mission_state: function (d) { S.missionState = d.state; $("mstate").textContent = d.state; },
    exec: function (d) { S.exec = d.state; renderExec(); dirty(); },
    oak: function (d) { $("oakst").textContent = d.status; },
    dets: function (d) { D.objects = d.objects || []; D.at = performance.now(); drawDets(); listDets(); },
    stats: function (d) {
      S.execAlive = !!d.exec_alive;
      if (d.cpu != null) G.push("cpu", d.cpu);
      if (d.mem) G.push("ram", 100 * d.mem.used_mb / d.mem.total_mb, d.mem.used_mb + " MB");
      if (d.temp != null) G.push("temp", d.temp);
      G.push("lidar", d.scan_hz);
      if (S.pose) G.push("speed", Math.abs(S.pose.v));
      setParserKind(d.parser);
      if (!S.execAlive) { $("xphase").textContent = "not running"; }
      updateArmBtn();
    },
    control: function (d) {
      S.controller = d.controller; S.halted = !!d.halted;
      S.amController = !!(S.token && d.controller_client && d.controller_client === S.myId);
      $("ctlname").textContent = d.controller || "none";
      refreshDriveUI();
    },
    auth: function (d) {
      if (d.ok) {
        S.token = d.token; S.name = d.name;
        try { sessionStorage.setItem("dh_token", d.token); sessionStorage.setItem("dh_name", d.name); } catch (e) {}
        closeModal("m-pin"); toast("Unlocked as " + d.name, "ok");
        $("btn-pin").textContent = d.name;
        refreshDriveUI();
      } else { $("pin-err").textContent = d.msg || "Wrong PIN."; }
    },
    parsed: function (d) { if (d.req === S.reqId) renderParsed(d.mission); },
    ack: function (d) {
      if (d.ok) {
        toast(d.msg || "Done", "ok");
        if (d.need_arm) openModal("m-arm");
      } else {
        if (d.busy) {
          if (confirm(d.msg)) send({ t: "take_control", force: true });
          return;
        }
        if (/Enter the PIN/.test(d.msg || "")) { S.token = null; try { sessionStorage.removeItem("dh_token"); } catch (e) {} openModal("m-pin"); }
        if (d.halted) { S.halted = true; refreshDriveUI(); }
        if (d.cmd !== "drive" || !S._driveWarned) toast(d.msg || "Refused", "warn");
        if (d.cmd === "drive") { S._driveWarned = true; setTimeout(function () { S._driveWarned = false; }, 2500); }
      }
    },
    toast: function (d) { toast(d.msg, d.level === "stop" ? "stop" : d.level === "info" ? "" : d.level); }
  };

  function setParserKind(k) {
    if (!k) return;
    $("parserkind").textContent = k === "tinybert4" ? "TinyBERT on the rover" : "keyword fallback";
  }

  /* ------------------------------------------------------------------ map */
  var cv = $("map"), cx2 = cv.getContext("2d");
  var V = { cx: 0, cy: 0, scale: 40, follow: true, fitted: false, dpr: 1, w: 0, h: 0 };
  var needDraw = true;
  function dirty() { needDraw = true; }
  function resize() {
    var r = cv.getBoundingClientRect(), d = window.devicePixelRatio || 1;
    V.dpr = d; V.w = r.width; V.h = r.height;
    cv.width = Math.round(r.width * d); cv.height = Math.round(r.height * d);
    dirty();
  }
  window.addEventListener("resize", resize);
  function w2s(x, y) { return [(x - V.cx) * V.scale + V.w / 2, V.h / 2 - (y - V.cy) * V.scale]; }
  function s2w(px, py) { return [(px - V.w / 2) / V.scale + V.cx, (V.h / 2 - py) / V.scale + V.cy]; }
  function fitView() {
    var m = S.map; if (!m || !V.w) return;
    var W = m.w * m.res, H = m.h * m.res;
    V.scale = Math.max(4, Math.min(120, 0.9 * Math.min(V.w / W, V.h / H)));
    if (!V.follow || !S.pose) { V.cx = m.ox + W / 2; V.cy = m.oy + H / 2; }
    V.fitted = true; dirty();
  }
  $("fit").onclick = function () { V.follow = false; $("follow").checked = false; fitView(); };
  $("follow").onchange = function (e) { V.follow = e.target.checked; if (V.follow && S.pose) { V.cx = S.pose.x; V.cy = S.pose.y; } dirty(); };
  $("showscan").onchange = dirty;

  function draw() {
    requestAnimationFrame(draw);
    if (!needDraw || !V.w) return;
    needDraw = false;
    var c = cx2, d = V.dpr;
    c.setTransform(d, 0, 0, d, 0, 0);
    c.clearRect(0, 0, V.w, V.h);
    // grid
    var step = V.scale > 60 ? 1 : V.scale > 14 ? 5 : 10;
    c.strokeStyle = "rgba(120,140,165,0.08)"; c.lineWidth = 1; c.beginPath();
    var a = s2w(0, V.h), b = s2w(V.w, 0);
    for (var gx = Math.floor(a[0] / step) * step; gx <= b[0]; gx += step) { var p = w2s(gx, 0); c.moveTo(p[0], 0); c.lineTo(p[0], V.h); }
    for (var gy = Math.floor(a[1] / step) * step; gy <= b[1]; gy += step) { var q = w2s(0, gy); c.moveTo(0, q[1]); c.lineTo(V.w, q[1]); }
    c.stroke();
    // occupancy map
    if (S.mapImg && S.map) {
      var m = S.map, tl = w2s(m.ox, m.oy + m.h * m.res);
      c.imageSmoothingEnabled = V.scale < 12;
      c.drawImage(S.mapImg, tl[0], tl[1], m.w * m.res * V.scale, m.h * m.res * V.scale);
    }
    // lidar
    if ($("showscan").checked && S.scan.length) {
      c.fillStyle = "rgba(255,107,139,0.85)";
      var r = Math.max(1.2, Math.min(2.6, V.scale / 30));
      for (var i = 0; i < S.scan.length; i++) { var sp = w2s(S.scan[i][0], S.scan[i][1]); c.fillRect(sp[0] - r / 2, sp[1] - r / 2, r, r); }
    }
    line(S.trail, "#4cc9f0", 2.2, null);
    line(S.plan, "#f0a23a", 2, [6, 5]);
    // junctions
    c.fillStyle = "#58a6ff";
    S.junctions.forEach(function (j) { var p = w2s(j[0], j[1]); c.save(); c.translate(p[0], p[1]); c.rotate(Math.PI / 4); c.fillRect(-4, -4, 8, 8); c.restore(); });
    // executive candidates and goal
    var ex = S.exec;
    if (ex && ex.candidates) {
      ex.candidates.forEach(function (k) {
        if (!k.goal) return;
        var p = w2s(k.goal[0], k.goal[1]);
        c.strokeStyle = "#a78bfa"; c.lineWidth = 2; c.beginPath(); c.arc(p[0], p[1], 7, 0, 7); c.stroke();
        if (k.score != null) { c.fillStyle = "#cfc3ff"; c.font = "11px ui-monospace,monospace"; c.fillText(Number(k.score).toFixed(2), p[0] + 9, p[1] - 6); }
      });
    }
    if (ex && ex.decision && ex.decision.path && ex.decision.path.length > 1) line(ex.decision.path, "#a78bfa", 2, [3, 4]);
    if (ex && ex.decision && ex.decision.goal) {
      var g = w2s(ex.decision.goal[0], ex.decision.goal[1]);
      c.fillStyle = "#a78bfa"; c.beginPath(); c.arc(g[0], g[1], 6, 0, 7); c.fill();
      c.strokeStyle = "#a78bfa"; c.lineWidth = 2; c.beginPath(); c.moveTo(g[0], g[1]); c.lineTo(g[0], g[1] - 18); c.stroke();
      c.fillRect(g[0], g[1] - 18, 10, 7);
    }
    // home
    var home = (ex && ex.home) || (S.trail.length ? S.trail[0] : null);
    if (home) {
      var hp = w2s(home[0], home[1]);
      c.strokeStyle = "#37b37a"; c.lineWidth = 2; c.beginPath();
      c.moveTo(hp[0] - 7, hp[1] + 6); c.lineTo(hp[0] - 7, hp[1] - 2); c.lineTo(hp[0], hp[1] - 8); c.lineTo(hp[0] + 7, hp[1] - 2); c.lineTo(hp[0] + 7, hp[1] + 6); c.closePath(); c.stroke();
      c.fillStyle = "#8fe3b9"; c.font = "11px ui-sans-serif,system-ui"; c.fillText("HOME", hp[0] + 10, hp[1] + 4);
    }
    // events (persons prominent)
    S.events.forEach(function (e) {
      if (e.x == null) return;
      var p = w2s(e.x, e.y);
      if (e.kind === "PERSON") {
        c.fillStyle = "rgba(229,72,77,0.25)"; c.beginPath(); c.arc(p[0], p[1], 14, 0, 7); c.fill();
        c.fillStyle = "#e5484d"; c.beginPath(); c.arc(p[0], p[1], 6, 0, 7); c.fill();
        c.fillStyle = "#ffb3b5"; c.font = "bold 12px ui-sans-serif,system-ui"; c.fillText(e.text || "Person", p[0] + 12, p[1] - 8);
      } else {
        c.fillStyle = e.kind === "ARRIVED" ? "#37b37a" : e.kind === "HALT" ? "#e5484d" : "#8a97a9";
        c.beginPath(); c.arc(p[0], p[1], 3.5, 0, 7); c.fill();
      }
    });
    // rover
    if (S.pose) {
      var rp = w2s(S.pose.x, S.pose.y), s = Math.max(9, 0.32 * V.scale);
      c.save(); c.translate(rp[0], rp[1]); c.rotate(-S.pose.yaw);
      c.fillStyle = "rgba(240,162,58,0.18)"; c.beginPath(); c.arc(0, 0, s * 1.6, 0, 7); c.fill();
      c.fillStyle = "#f0a23a"; c.strokeStyle = "#1b1204"; c.lineWidth = 1.5;
      c.beginPath(); c.moveTo(s, 0); c.lineTo(-s * 0.7, s * 0.62); c.lineTo(-s * 0.35, 0); c.lineTo(-s * 0.7, -s * 0.62); c.closePath(); c.fill(); c.stroke();
      c.restore();
    }
    // scale bar
    var mpx = 1 / V.scale, len = [0.5, 1, 2, 5, 10, 20].filter(function (x) { return x / mpx < 140; }).pop() || 0.5;
    var bx = V.w - 20 - len * V.scale, by = V.h - 18;
    c.strokeStyle = "#c4cfdc"; c.lineWidth = 2; c.beginPath(); c.moveTo(bx, by); c.lineTo(bx + len * V.scale, by); c.stroke();
    c.fillStyle = "#c4cfdc"; c.font = "11px ui-monospace,monospace"; c.fillText(len + " m", bx, by - 6);
    function line(pts, color, w, dash) {
      if (!pts || pts.length < 2) return;
      c.strokeStyle = color; c.lineWidth = w; c.setLineDash(dash || []); c.lineJoin = "round"; c.beginPath();
      for (var k = 0; k < pts.length; k++) { var p = w2s(pts[k][0], pts[k][1]); if (k) c.lineTo(p[0], p[1]); else c.moveTo(p[0], p[1]); }
      c.stroke(); c.setLineDash([]);
    }
  }
  // pan and zoom: mouse, touch, pinch
  var ptrs = {}, pinch0 = null;
  cv.addEventListener("pointerdown", function (e) { try { cv.setPointerCapture(e.pointerId); } catch (_) {} ptrs[e.pointerId] = [e.offsetX, e.offsetY]; });
  cv.addEventListener("pointermove", function (e) {
    if (!ptrs[e.pointerId]) return;
    var ids = Object.keys(ptrs);
    if (ids.length === 1) {
      var p = ptrs[e.pointerId];
      V.cx -= (e.offsetX - p[0]) / V.scale; V.cy += (e.offsetY - p[1]) / V.scale;
      if (V.follow) { V.follow = false; $("follow").checked = false; }
    } else if (ids.length === 2) {
      ptrs[e.pointerId] = [e.offsetX, e.offsetY];
      var a = ptrs[ids[0]], b = ptrs[ids[1]], dist = Math.hypot(a[0] - b[0], a[1] - b[1]);
      if (pinch0) { zoomAt((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, dist / pinch0); }
      pinch0 = dist; dirty(); return;
    }
    ptrs[e.pointerId] = [e.offsetX, e.offsetY]; dirty();
  });
  function up(e) { delete ptrs[e.pointerId]; pinch0 = null; }
  cv.addEventListener("pointerup", up); cv.addEventListener("pointercancel", up);
  cv.addEventListener("wheel", function (e) { e.preventDefault(); zoomAt(e.offsetX, e.offsetY, Math.exp(-e.deltaY * 0.0015)); }, { passive: false });
  cv.addEventListener("dblclick", function () { V.follow = true; $("follow").checked = true; if (S.pose) { V.cx = S.pose.x; V.cy = S.pose.y; } dirty(); });
  function zoomAt(px, py, f) {
    var w = s2w(px, py);
    V.scale = Math.max(3, Math.min(200, V.scale * f));
    if (!V.follow) { var w2 = s2w(px, py); V.cx += w[0] - w2[0]; V.cy += w[1] - w2[1]; }
    dirty();
  }

  /* ------------------------------------------------------------------ camera */
  var camUrl = null, camN = 0, camT = performance.now();
  function onCamera(buf) {
    var u8 = new Uint8Array(buf);
    if (u8[0] !== 74) return;
    var src = u8[1];
    if (src === 2) { onDepth(u8); return; }
    var blob = new Blob([u8.subarray(2)], { type: "image/jpeg" });
    var url = URL.createObjectURL(blob), img = $("cam");
    img.onload = function () { if (camUrl && camUrl !== url) URL.revokeObjectURL(camUrl); camUrl = url; };
    img.src = url;
    $("camempty").hidden = true;
    $("camsrc").textContent = src === 1 ? "OAK-D depth camera" : "front CSI camera";
    D.src = src;
    img.onload = function () { if (camUrl && camUrl !== url) URL.revokeObjectURL(camUrl); camUrl = url; drawDets(); };
    camN++;
    var now = performance.now();
    if (now - camT > 2000) { $("camfps").textContent = (camN * 1000 / (now - camT)).toFixed(1) + " fps"; camN = 0; camT = now; }
  }
  function sendCamera() { send({ t: "camera", on: $("camon").checked }); if (!$("camon").checked) { $("cam").removeAttribute("src"); $("camempty").hidden = false; } }

  /* ------------------------------------------------------------------ objects (OAK-D detections) */
  // Boxes come from the OAK-D's own detector (YOLOv6-nano, 80 COCO classes, ~10 Hz) as fractions of
  // the colour image, so they are drawn only over OAK frames (src 1), never over the CSI camera.
  var D = { objects: [], at: 0, src: -1, seen: {} };
  var DET_STALE_MS = 700;
  function hue(label) { var h = 0; for (var i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) % 360; return h; }
  function detColor(label) { return label === "person" ? "#ff4d57" : "hsl(" + hue(label) + ",85%,58%)"; }
  function drawDets() {
    var cv = $("camdets"), img = $("cam"), ctx = cv.getContext("2d");
    var W = cv.clientWidth, H = cv.clientHeight, dpr = window.devicePixelRatio || 1;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    var fresh = performance.now() - D.at < DET_STALE_MS;
    if (!$("detson").checked || !fresh || D.src !== 1 || !img.naturalWidth || !img.getAttribute("src")) return;
    var k = Math.min(W / img.naturalWidth, H / img.naturalHeight);       // object-fit: contain
    var dw = img.naturalWidth * k, dh = img.naturalHeight * k, ox = (W - dw) / 2, oy = (H - dh) / 2;
    ctx.lineWidth = 2; ctx.font = "600 12px ui-sans-serif,system-ui,sans-serif"; ctx.textBaseline = "top";
    D.objects.forEach(function (o) {
      var b = o.box || [0, 0, 0, 0];
      var x = ox + b[0] * dw, y = oy + b[1] * dh, w = (b[2] - b[0]) * dw, h = (b[3] - b[1]) * dh;
      var col = detColor(o.label);
      ctx.strokeStyle = col; ctx.strokeRect(x, y, w, h);
      var txt = o.label + " " + Math.round(100 * o.conf) + "%" + (o.z != null ? "  " + o.z.toFixed(1) + " m" : "");
      var tw = ctx.measureText(txt).width + 8, ty = y - 17 < oy ? y + 1 : y - 17;
      ctx.fillStyle = col; ctx.fillRect(x, ty, tw, 16);
      ctx.fillStyle = "#0b0e13"; ctx.fillText(txt, x + 4, ty + 2);
    });
  }
  function listDets() {
    var el = $("detslist");
    if (!$("detson").checked) { el.textContent = "Objects: overlay off"; return; }
    var count = {};
    D.objects.forEach(function (o) { count[o.label] = (count[o.label] || 0) + 1; D.seen[o.label] = Math.max(D.seen[o.label] || 0, 1); });
    var now = Object.keys(count).sort();
    var seen = Object.keys(D.seen).sort();
    el.innerHTML = "";
    var head = document.createElement("b"); head.textContent = now.length ? "Seeing now: " : "Seeing now: nothing ";
    el.appendChild(head);
    now.forEach(function (l) {
      var s = document.createElement("span"); s.className = "obj";
      var sw = document.createElement("i"); sw.style.background = detColor(l); s.appendChild(sw);
      s.appendChild(document.createTextNode(l + " " + count[l])); el.appendChild(s);
    });
    var all = document.createElement("div"); all.textContent = "Seen this session: " + (seen.length ? seen.join(", ") : "none");
    el.appendChild(all);
  }
  $("detson").onchange = function () { drawDets(); listDets(); };
  window.addEventListener("resize", drawDets);
  setInterval(function () { if (performance.now() - D.at > DET_STALE_MS) drawDets(); }, 500);   // old boxes never linger
  $("camon").onchange = sendCamera;

  /* ------------------------------------------------------------------ depth */
  var depUrl = null, depN = 0, depT = performance.now(), depLast = 0;
  function onDepth(u8) {
    var url = URL.createObjectURL(new Blob([u8.subarray(2)], { type: "image/jpeg" })), img = $("depth");
    img.onload = function () { if (depUrl && depUrl !== url) URL.revokeObjectURL(depUrl); depUrl = url; };
    img.src = url;
    $("depthempty").hidden = true;
    depLast = performance.now(); depN++;
    if (depLast - depT > 2000) { $("depthfps").textContent = (depN * 1000 / (depLast - depT)).toFixed(1) + " fps"; depN = 0; depT = depLast; }
  }
  function depthNote(txt) {
    $("depth").removeAttribute("src"); $("depthfps").textContent = "";
    $("depthempty").textContent = txt; $("depthempty").hidden = false;
  }
  function sendDepth() {
    var on = $("depthon").checked;
    send({ t: "depth", on: on });
    depLast = 0;
    depthNote(on ? "Waiting for depth" : "Depth off");
  }
  $("depthon").onchange = sendDepth;
  setInterval(function () {            // an old depth picture must never look live
    if ($("depthon").checked && depLast && performance.now() - depLast > 3000) {
      depLast = 0;
      depthNote("Depth paused: no frames for 3 s");
    }
  }, 1000);

  /* ------------------------------------------------------------------ graphs */
  var G = (function () {
    var defs = [
      ["speed", "Speed", "m/s", "#f0a23a", 0, 0.4, 2],
      ["cpu", "CPU", "%", "#58a6ff", 0, 100, 0],
      ["ram", "RAM", "%", "#a78bfa", 0, 100, 0],
      ["temp", "Temp", "°C", "#ff8a5b", 20, 80, 1],
      ["lidar", "Lidar", "Hz", "#37b37a", 0, 12, 1],
      ["rtt", "Latency", "ms", "#4cc9f0", 0, 400, 0]
    ];
    var box = $("graphs"), series = {};
    defs.forEach(function (d) {
      var el = document.createElement("div");
      el.className = "gcard";
      el.innerHTML = '<div class="gtop"><span>' + d[1] + '</span><span class="gval">–</span></div><canvas></canvas>';
      box.appendChild(el);
      series[d[0]] = { def: d, el: el, cv: el.querySelector("canvas"), val: el.querySelector(".gval"), pts: [] };
    });
    function push(k, v, label) {
      var s = series[k]; if (!s || v == null || isNaN(v)) return;
      s.pts.push(v); if (s.pts.length > 120) s.pts.shift();
      s.val.textContent = (label || v.toFixed(s.def[6])) + (label ? "" : " " + s.def[2]);
      paint(s);
    }
    function paint(s) {
      var c = s.cv, r = c.getBoundingClientRect(), d = window.devicePixelRatio || 1;
      if (!r.width) return;
      c.width = r.width * d; c.height = r.height * d;
      var x = c.getContext("2d"); x.setTransform(d, 0, 0, d, 0, 0);
      var W = r.width, H = r.height, lo = s.def[4], hi = Math.max(s.def[5], Math.max.apply(null, s.pts) * 1.1);
      x.clearRect(0, 0, W, H);
      x.strokeStyle = "rgba(120,140,165,0.12)"; x.beginPath(); x.moveTo(0, H - 0.5); x.lineTo(W, H - 0.5); x.moveTo(0, 0.5); x.lineTo(W, 0.5); x.stroke();
      if (s.pts.length < 2) return;
      var n = 120, dx = W / (n - 1), off = n - s.pts.length;
      x.beginPath();
      s.pts.forEach(function (v, i) { var px = (off + i) * dx, py = H - (Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo) * (H - 4) - 2; if (i) x.lineTo(px, py); else x.moveTo(px, py); });
      x.strokeStyle = s.def[3]; x.lineWidth = 1.8; x.stroke();
      x.lineTo(W, H); x.lineTo(off * dx, H); x.closePath();
      x.globalAlpha = 0.14; x.fillStyle = s.def[3]; x.fill(); x.globalAlpha = 1;
      x.fillStyle = "#7d8a9c"; x.font = "10px ui-monospace,monospace"; x.fillText(String(Math.round(hi)), 3, 11);
    }
    return { push: push };
  })();

  /* ------------------------------------------------------------------ drive */
  var pad = $("pad"), knob = $("knob"), drv = { on: false, v: 0, w: 0, timer: null, keys: {} };
  function refreshDriveUI() {
    var hint = $("drivehint"), take = $("btn-take");
    $("btn-resume").hidden = !(S.amController && S.halted);
    if (!S.token) { hint.textContent = "View only"; take.textContent = "Enter PIN to drive"; }
    else if (S.amController) { hint.textContent = S.halted ? "Stopped. Press Resume" : "You are driving"; take.textContent = "Release control"; }
    else if (S.controller) { hint.textContent = S.controller + " is driving"; take.textContent = "Take over"; }
    else { hint.textContent = "Nobody is driving"; take.textContent = "Take control"; }
    pad.classList.toggle("disabled", !S.amController);
    updateArmBtn();
  }
  $("btn-take").onclick = function () {
    if (!S.token) { openModal("m-pin"); return; }
    if (S.amController) { send({ t: "release" }); return; }
    send({ t: "take_control" });
  };
  $("btn-resume").onclick = function () { send({ t: "resume" }); };
  $("btn-home").onclick = function () {
    if (!S.token) { openModal("m-pin"); return; }
    if (!S.amController) { toast("Take control first.", "warn"); return; }
    send({ t: "home" });
  };
  function padVector(e) {
    var r = pad.getBoundingClientRect(), R = r.width / 2;
    var dx = e.clientX - (r.left + R), dy = e.clientY - (r.top + R), L = Math.hypot(dx, dy), max = R - 38;
    if (L > max) { dx *= max / L; dy *= max / L; }
    knob.style.transform = "translate(" + dx + "px," + dy + "px)";
    var nx = dx / max, ny = dy / max;
    drv.v = Math.abs(ny) < 0.08 ? 0 : -ny;
    drv.w = Math.abs(nx) < 0.08 ? 0 : -nx;
  }
  pad.addEventListener("pointerdown", function (e) {
    if (!S.amController) { toast(S.token ? "Take control first." : "Enter the PIN to drive.", "warn"); if (!S.token) openModal("m-pin"); return; }
    if (S.halted) { toast("Stopped. Press Resume first.", "warn"); return; }
    try { pad.setPointerCapture(e.pointerId); } catch (_) {} pad.classList.add("active"); drv.on = true; padVector(e); startDrive();
  });
  pad.addEventListener("pointermove", function (e) { if (drv.on) padVector(e); });
  function padUp() { if (drv.on) releasePad(); }
  pad.addEventListener("pointerup", padUp); pad.addEventListener("pointercancel", padUp); pad.addEventListener("lostpointercapture", padUp);
  function startDrive() {
    if (drv.timer) return;
    var tick = function () { send({ t: "drive", v: drv.v, w: drv.w }); };
    tick(); drv.timer = setInterval(tick, 100);
  }
  function releasePad() {
    drv.on = false; drv.v = 0; drv.w = 0; pad.classList.remove("active");
    knob.style.transform = "translate(0,0)";
    if (drv.timer) { clearInterval(drv.timer); drv.timer = null; send({ t: "drive", v: 0, w: 0 }); }
  }
  var KEYS = { ArrowUp: [1, 0], KeyW: [1, 0], ArrowDown: [-1, 0], KeyS: [-1, 0], ArrowLeft: [0, 1], KeyA: [0, 1], ArrowRight: [0, -1], KeyD: [0, -1] };
  function keyVec() {
    var v = 0, w = 0;
    Object.keys(drv.keys).forEach(function (k) { if (drv.keys[k]) { v += KEYS[k][0]; w += KEYS[k][1]; } });
    drv.v = Math.max(-1, Math.min(1, v)) * 0.7; drv.w = Math.max(-1, Math.min(1, w)) * 0.8;
    var any = Object.keys(drv.keys).some(function (k) { return drv.keys[k]; });
    if (any && !drv.timer) startDrive(); if (!any && drv.timer && !drv.on) releasePad();
  }
  document.addEventListener("keydown", function (e) {
    if (!KEYS[e.code] || /INPUT|TEXTAREA/.test(document.activeElement.tagName) || !S.amController || S.halted) return;
    e.preventDefault(); drv.keys[e.code] = true; keyVec();
  });
  document.addEventListener("keyup", function (e) { if (KEYS[e.code]) { drv.keys[e.code] = false; keyVec(); } });
  document.addEventListener("visibilitychange", function () { if (document.hidden) { drv.keys = {}; releasePad(); } });
  window.addEventListener("blur", function () { drv.keys = {}; releasePad(); });

  /* ------------------------------------------------------------------ STOP */
  function stop() {
    releasePad(); drv.keys = {};
    if (!send({ t: "stop", why: "STOP pressed by " + (S.name || "a viewer") })) toast("Not connected. Use the local STOP on the rover.", "stop");
  }
  $("btn-stop").onclick = stop; $("btn-stop2").onclick = stop;
  document.addEventListener("keydown", function (e) { if (e.code === "Space" && !/INPUT|TEXTAREA/.test(document.activeElement.tagName)) { e.preventDefault(); stop(); } });

  /* ------------------------------------------------------------------ PIN, share, arm */
  function openModal(id) { $(id).hidden = false; var f = $(id).querySelector("input"); if (f) setTimeout(function () { f.focus(); }, 30); }
  function closeModal(id) { $(id).hidden = true; }
  document.querySelectorAll("[data-close]").forEach(function (b) { b.onclick = function () { b.closest(".modal").hidden = true; }; });
  document.querySelectorAll(".modal").forEach(function (m) { m.addEventListener("click", function (e) { if (e.target === m) m.hidden = true; }); });
  $("btn-pin").onclick = function () { $("pin-err").textContent = ""; $("pin-name").value = S.name || ""; openModal("m-pin"); };
  function auth(name, pin) { send({ t: "auth", name: name, pin: pin }); }
  $("pinform").onsubmit = function (e) { e.preventDefault(); auth($("pin-name").value.trim() || "operator", $("pin-val").value.trim()); };
  $("btn-share").onclick = function () {
    fetch("/api/info", { cache: "no-store" }).then(function (r) { return r.json(); }).then(function (i) {
      var url = i.public_url || location.origin + "/";
      $("qr").src = "/qr.svg?ts=" + Date.now(); $("shareurl").textContent = url; $("shareurl").href = url;
      openModal("m-share");
    });
  };
  $("btn-copy").onclick = function () {
    var u = $("shareurl").textContent;
    if (navigator.clipboard) navigator.clipboard.writeText(u).then(function () { toast("Link copied", "ok"); });
  };
  function updateArmBtn() {
    var b = $("btn-arm"), ex = S.exec || {};
    b.hidden = !(S.execAlive && S.amController);
    $("batform").hidden = !(S.execAlive && S.amController);
    b.classList.toggle("on", !!ex.armed);
    b.textContent = ex.armed ? "DISARM" : "ARM";
  }
  $("btn-arm").onclick = function () { if ((S.exec || {}).armed) send({ t: "arm", on: false }); else openModal("m-arm"); };
  $("btn-arm-ok").onclick = function () { closeModal("m-arm"); send({ t: "arm", on: true }); };
  $("batform").onsubmit = function (e) {
    e.preventDefault();
    var v = parseFloat($("batin").value);
    if (!(v >= 5 && v <= 100)) { toast("Battery must be 5 to 100 %", "warn"); return; }
    send({ t: "battery", pct: v });
  };

  /* ------------------------------------------------------------------ mission */
  var TASK = { SEARCH_SURVIVORS: "Search for survivors", MAP_AREA: "Map the area", GOTO_WAYPOINT: "Go to waypoint",
    RETURN_HOME: "Return home", HALT: "Stop now", UNSUPPORTED: "Not a rover mission" };
  var PERSON = { REPORT_AND_CONTINUE: "report it and keep searching", REPORT_AND_RETURN: "report it and come back", STAY: "stay with them" };
  var LINK = { CONTINUE_THEN_RETURN: "keep going, then come back", RETURN_NOW: "come back at once", CONTINUE: "keep going", HOLD: "stop and wait" };
  var yes = function (b) { return b ? "yes" : "no"; };
  $("mform").onsubmit = function (e) {
    e.preventDefault();
    var t = $("mtext").value.trim(); if (!t) { $("mtext").focus(); return; }
    S.reqId++; if (!send({ t: "parse", text: t, req: S.reqId })) toast("Not connected.", "warn");
  };
  document.querySelectorAll(".qchip").forEach(function (b) { b.onclick = function () { $("mtext").value = b.dataset.t; $("mform").requestSubmit ? $("mform").requestSubmit() : $("mform").onsubmit(new Event("submit")); }; });
  function renderParsed(m) {
    S.lastParsed = m;
    var box = $("parsed"); box.hidden = false; box.classList.toggle("bad", !m.understood);
    $("ptask").textContent = m.understood ? (TASK[m.task] || m.task) + (m.task === "GOTO_WAYPOINT" && m.waypoint ? " " + m.waypoint : "") : "Not understood";
    $("pconfbar").style.width = Math.round(100 * (m.confidence || 0)) + "%";
    $("pconfbar").style.background = m.confidence > 0.85 ? "var(--ok)" : m.confidence > 0.6 ? "var(--amber)" : "var(--bad)";
    $("pmeta").textContent = Math.round(100 * (m.confidence || 0)) + "% sure, " + (m.parser === "tinybert4" ? "TinyBERT on rover" : "keyword fallback") + (m.parse_ms != null ? ", " + m.parse_ms + " ms" : "");
    var rows = [];
    if (!m.understood) rows.push(["Why", m.task === "UNSUPPORTED" ? "This does not look like a rover mission." : "Not sure enough. Please rephrase."]);
    else {
      if (m.task === "SEARCH_SURVIVORS" || m.task === "MAP_AREA") {
        rows.push(["If a person is found", PERSON[m.on_person_found]]);
        rows.push(["Go closer to check", yes(m.approach_person)]);
        rows.push(["Report their location", yes(m.report_person)]);
        rows.push(["First branch", (m.first_branch || "ANY").toLowerCase()]);
      }
      if (m.task !== "HALT") rows.push(["If the link drops", LINK[m.on_link_loss]]);
      if (m.task === "SEARCH_SURVIVORS" || m.task === "MAP_AREA") rows.push(["Come back when done", yes(m.return_when_done)]);
      rows.push(["Speed", (m.speed || "NORMAL").toLowerCase()]);
      if (m.max_duration_min != null) rows.push(["Time limit", m.max_duration_min + " min"]);
      if (m.max_distance_m != null) rows.push(["Distance limit", m.max_distance_m + " m"]);
      if (m.min_battery_pct != null) rows.push(["Return below battery", m.min_battery_pct + " %"]);
    }
    var dl = $("pfields"); dl.innerHTML = "";
    rows.forEach(function (r) { var dt = document.createElement("dt"), dd = document.createElement("dd"); dt.textContent = r[0]; dd.textContent = r[1]; dl.appendChild(dt); dl.appendChild(dd); });
    $("pjson").textContent = JSON.stringify(m, null, 2);
    $("btn-start").disabled = !m.understood;
    $("btn-start").textContent = m.task === "HALT" ? "Stop now" : "Start mission";
  }
  $("btn-json").onclick = function () { $("pjson").hidden = !$("pjson").hidden; };
  $("btn-start").onclick = function () {
    var m = S.lastParsed; if (!m) return;
    if (m.task === "HALT") { stop(); return; }
    if (!S.token) { openModal("m-pin"); return; }
    if (!S.amController) { toast("Take control first.", "warn"); return; }
    send({ t: "start_mission", mission: m });
  };

  /* ------------------------------------------------------------------ AI + events */
  function renderExec() {
    var ex = S.exec || {};
    $("xphase").textContent = (ex.phase || "?") + (ex.armed ? "  (armed)" : "  (not armed)") + (ex.dry_run ? "  test mode" : "");
    var d = ex.decision || {};
    $("xdec").textContent = d.kind ? d.kind + (d.sensor && d.sensor !== "MOVE" ? " + " + d.sensor : "") : "none";
    $("xreason").textContent = d.reason || ex.reason || "Waiting for a mission.";
    $("xsrc").textContent = d.source ? d.source + (d.tier ? ", " + d.tier + " rule" : "") + (d.source === "learned" && d.confidence != null ? " (" + Math.round(100 * d.confidence) + "%)" : "") : "none";
    $("xms").textContent = ex.decide_ms != null ? ex.decide_ms + " ms on the rover" : "-";
    $("xlink").textContent = ex.link_ok == null ? "-" : ex.link_ok ? "up" : "lost (rover decides alone)";
    $("xbat").textContent = ex.battery_pct == null ? "-" : ex.battery_pct + " %" +
      (ex.battery_v != null ? " (" + ex.battery_v.toFixed(2) + " V measured)" : " (estimate)");
    $("xodo").textContent = ex.odo_m != null ? ex.odo_m + " m" : "-";
    var ppl = ex.persons || [];
    $("xppl").textContent = ppl.length ? ppl.filter(function (p) { return p.confirmed; }).length + " confirmed, " + ppl.filter(function (p) { return p.reported; }).length + " reported" : "0";
    updateArmBtn();
  }
  function renderEvents() {
    var ol = $("events"); ol.innerHTML = "";
    var items = S.events.slice().reverse();
    if (!items.length) { ol.innerHTML = '<li class="muted">No events yet</li>'; return; }
    items.forEach(function (e) {
      var li = document.createElement("li"); li.className = e.kind || "";
      var t = new Date((e.rx || Date.now() / 1000) * 1000);
      var hh = ("0" + t.getHours()).slice(-2) + ":" + ("0" + t.getMinutes()).slice(-2) + ":" + ("0" + t.getSeconds()).slice(-2);
      li.innerHTML = "<time></time><span class=kind></span><span class=txt></span>";
      li.children[0].textContent = hh; li.children[1].textContent = e.kind || "INFO"; li.children[2].textContent = e.text || "";
      ol.appendChild(li);
    });
  }

  /* ------------------------------------------------------------------ boot */
  if (S.name) $("btn-pin").textContent = S.name;
  resize(); requestAnimationFrame(draw); connect(); refreshDriveUI();
})();
