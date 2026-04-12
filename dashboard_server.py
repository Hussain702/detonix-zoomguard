"""
Detonix ZoomGuard — Live Dashboard Server
Uses Python's built-in http.server + polling (no Flask needed).

main.py calls update_state() to push live data.
Browser polls /state every 800ms to update the dashboard.

Usage (with --dashboard flag):
    python main.py --dashboard
    
Or run separately:
    Terminal 1: python dashboard_server.py
    Terminal 2: python main.py
    Browser:    http://localhost:5050
"""

import http.server
import json
import os
import threading
import socketserver
from datetime import datetime

PORT = 5050
LIVE_FILE = "logs/live_state.json"

# ── Shared state written by orchestrator ─────────────────────────────────────
_state = {
    "running": False,
    "frame": 0,
    "total_frames": 0,
    "fps": 0.0,
    "duration": 0,
    "session_id": "",
    "video_name": "",
    "resolution": "",
    "threshold": 0.65,
    "persons": [],
    "alerts": [],
    "videos_done": 0,
    "status": "idle",
}
_lock = threading.Lock()


def push_event(event_type, data):
    """Called by orchestrator to update live state."""
    with _lock:
        if event_type == "session_start":
            _state.update({
                "running": True,
                "status": "running",
                "session_id": data.get("session_id", ""),
                "video_name": data.get("video_name", ""),
                "total_frames": data.get("total_frames", 0),
                "fps": data.get("fps", 0),
                "resolution": data.get("resolution", ""),
                "threshold": data.get("threshold", 0.65),
                "frame": 0,
                "persons": [],
                "alerts": [],
            })

        elif event_type == "frame":
            _state["frame"] = data.get("frame", 0)
            _state["fps"] = data.get("fps", 0)
            _state["duration"] = data.get("duration", 0)

        elif event_type == "person_update":
            persons = _state["persons"]
            tid = data.get("id")
            for p in persons:
                if p["id"] == tid:
                    p.update(data)
                    break
            else:
                _state["persons"].append(dict(data))

        elif event_type == "alert":
            _state["alerts"].insert(0, data)

        elif event_type == "session_end":
            _state["running"] = False
            _state["status"] = "done"
            _state["videos_done"] = _state.get("videos_done", 0) + 1

    # Write to file so browser can poll it
    os.makedirs("logs", exist_ok=True)
    try:
        with open(LIVE_FILE, "w") as f:
            json.dump(_state, f)
    except Exception:
        pass


def update_state(**kwargs):
    """Legacy helper — just calls push_event internally."""
    push_event("frame", kwargs)


def _get_state():
    try:
        with open(LIVE_FILE) as f:
            return json.load(f)
    except Exception:
        return _state


# ── HTTP handler ──────────────────────────────────────────────────────────────
class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # suppress request logs

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = _DASHBOARD_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/state":
            data = json.dumps(_get_state()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        else:
            self.send_response(404)
            self.end_headers()


_server_instance = None


def start_server(port=5050):
    """Start dashboard HTTP server in a background daemon thread."""
    global PORT, _server_instance
    PORT = port

    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("", port), _Handler)
    _server_instance = httpd

    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()

    import time
    time.sleep(0.3)
    print(f"\n  Dashboard live at: http://localhost:{port}\n")
    return t


# ── Dashboard HTML ────────────────────────────────────────────────────────────
_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Detonix ZoomGuard — Live</title>
<style>
:root {
  --bg:#0d0f10; --bg2:#121416; --bg3:#1a1d20;
  --border:rgba(255,255,255,0.07); --border2:rgba(255,255,255,0.13);
  --text:#e4e7eb; --muted:#636b74; --faint:#2a2e33;
  --green:#22c47a; --green-d:rgba(34,196,122,0.12);
  --red:#ef4444;   --red-d:rgba(239,68,68,0.12);
  --amber:#f59e0b; --blue:#60a5fa; --purple:#a78bfa;
  --radius:10px; --radius-sm:6px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:-apple-system,'Segoe UI',system-ui,sans-serif;
     font-size:13px;line-height:1.5;height:100vh;overflow:hidden;display:flex;flex-direction:column}

/* topbar */
.topbar{display:flex;align-items:center;justify-content:space-between;
        padding:0 20px;height:50px;flex-shrink:0;
        border-bottom:1px solid var(--border);background:var(--bg2)}
.brand{display:flex;align-items:center;gap:10px}
.brand-icon{width:30px;height:30px;border-radius:8px;background:#0a3d2a;
            display:flex;align-items:center;justify-content:center}
.brand-name{font-size:14px;font-weight:600;letter-spacing:-.02em}
.brand-sub{font-size:10px;color:var(--muted)}
.tr{display:flex;align-items:center;gap:12px}
.dot{width:7px;height:7px;border-radius:50%;background:var(--faint);transition:all .4s}
.dot.live{background:var(--green);box-shadow:0 0 7px var(--green)}
.dot.done{background:var(--blue)}
.lbl{font-size:11px;color:var(--muted)}
.badge{font-size:11px;color:var(--muted);background:var(--bg3);
       border:1px solid var(--border);border-radius:var(--radius-sm);padding:3px 9px}

/* layout */
.body{display:grid;grid-template-columns:240px 1fr 260px;flex:1;overflow:hidden}

/* panels */
.left{border-right:1px solid var(--border);overflow-y:auto;display:flex;flex-direction:column}
.right{border-left:1px solid var(--border);overflow-y:auto;display:flex;flex-direction:column}
.center{display:flex;flex-direction:column;overflow:hidden}

.ph{padding:13px 14px 6px;font-size:10px;font-weight:600;letter-spacing:.07em;
    text-transform:uppercase;color:var(--muted)}

/* metric cards */
.mgrid{display:grid;grid-template-columns:1fr 1fr;gap:7px;padding:0 10px 10px}
.mc{background:var(--bg3);border:1px solid var(--border);border-radius:var(--radius);padding:11px 13px}
.mc-l{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:5px}
.mc-v{font-size:21px;font-weight:700;line-height:1}

/* progress */
.prog-wrap{padding:6px 10px 10px}
.prog-row{display:flex;justify-content:space-between;margin-bottom:5px}
.prog-row span{font-size:11px;color:var(--muted)}
.prog-row strong{font-size:11px;color:var(--text)}
.prog-track{height:4px;background:var(--faint);border-radius:2px;overflow:hidden}
.prog-fill{height:100%;background:var(--green);border-radius:2px;transition:width .4s ease;width:0%}

/* status pill */
.status-wrap{padding:4px 10px 12px}
.spill{display:inline-flex;align-items:center;gap:6px;font-size:11px;font-weight:500;
       padding:4px 11px;border-radius:99px;border:1px solid var(--border2)}
.spill.idle{color:var(--muted)}
.spill.running{color:var(--green);background:var(--green-d);border-color:rgba(34,196,122,.25)}
.spill.done{color:var(--blue);background:rgba(96,165,250,.1);border-color:rgba(96,165,250,.25)}
.pdot{width:6px;height:6px;border-radius:50%;background:currentColor}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.25}}
.blink{animation:blink 1.1s infinite}

.div{height:1px;background:var(--border);margin:2px 10px}

/* persons table */
.thead{padding:8px 16px;border-bottom:1px solid var(--border);
       display:grid;grid-template-columns:52px 1fr 100px 70px 80px 90px;
       background:var(--bg2);position:sticky;top:0;z-index:2}
.th{font-size:10px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.tbody{overflow-y:auto;flex:1}
.tr2{display:grid;grid-template-columns:52px 1fr 100px 70px 80px 90px;
     padding:10px 16px;border-bottom:1px solid var(--border);align-items:center;
     transition:background .15s}
.tr2:hover{background:var(--bg3)}
.tid{font-size:13px;font-weight:700;color:var(--text)}

.bar-wrap{display:flex;align-items:center;gap:8px}
.bar-bg{flex:1;height:3px;background:var(--faint);border-radius:2px;overflow:hidden}
.bar-fill{height:100%;border-radius:2px;transition:width .5s ease}
.bar-num{font-size:11px;color:var(--muted);min-width:34px;text-align:right}

.mini{display:flex;align-items:flex-end;gap:1.5px;height:22px}
.mb{width:4px;border-radius:1px;min-height:2px;transition:height .3s}

.vtag{display:inline-flex;align-items:center;gap:4px;font-size:11px;font-weight:600;
      padding:3px 9px;border-radius:99px}
.vtag.real{color:var(--green);background:var(--green-d)}
.vtag.fake{color:var(--red);background:var(--red-d)}
.vtag.wait{color:var(--muted);background:var(--faint)}
@keyframes flash{0%,100%{opacity:1}50%{opacity:.55}}
.vtag.fake{animation:flash 1.8s infinite}

.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;
       height:180px;color:var(--muted);gap:8px;font-size:13px}

/* events */
.ev-list{flex:1;overflow-y:auto}
.ev{padding:11px 13px;border-bottom:1px solid var(--border)}
.ev.fake{border-left:2px solid var(--red)}
.ev.info{border-left:2px solid var(--blue)}
.ev.ok  {border-left:2px solid var(--green)}
.ev-t{font-size:12px;font-weight:600;margin-bottom:2px}
.ev-t.fake{color:var(--red)} .ev-t.info{color:var(--blue)} .ev-t.ok{color:var(--green)}
.ev-m{font-size:11px;color:var(--muted)}
.ev-badge{display:inline-block;font-size:10px;font-weight:700;
          background:var(--red-d);color:var(--red);padding:2px 7px;
          border-radius:4px;margin-top:3px}
.noev{padding:18px 13px;text-align:center;color:var(--muted);font-size:12px}

.file-info{padding:10px 13px;border-top:1px solid var(--border);flex-shrink:0}
.fi-name{font-size:11px;font-weight:500;color:var(--text);word-break:break-all}
.fi-meta{font-size:10px;color:var(--muted);margin-top:2px}
</style>
</head>
<body>

<div class="topbar">
  <div class="brand">
    <div class="brand-icon">
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
        <circle cx="8" cy="8" r="5.5" stroke="#22c47a" stroke-width="1.5"/>
        <circle cx="8" cy="8" r="2" fill="#22c47a"/>
        <line x1="8" y1="1" x2="8" y2="2.8" stroke="#22c47a" stroke-width="1.5" stroke-linecap="round"/>
        <line x1="8" y1="13.2" x2="8" y2="15" stroke="#22c47a" stroke-width="1.5" stroke-linecap="round"/>
        <line x1="1" y1="8" x2="2.8" y2="8" stroke="#22c47a" stroke-width="1.5" stroke-linecap="round"/>
        <line x1="13.2" y1="8" x2="15" y2="8" stroke="#22c47a" stroke-width="1.5" stroke-linecap="round"/>
      </svg>
    </div>
    <div><div class="brand-name">Detonix ZoomGuard</div>
         <div class="brand-sub">Live Deepfake Detection</div></div>
  </div>
  <div class="tr">
    <button id="snd-btn" onclick="unlockAudio()" title="Click to enable alert sounds"
      style="display:flex;align-items:center;gap:6px;background:#1a2a1a;border:1px solid #22c47a44;
             border-radius:6px;padding:4px 10px;cursor:pointer;color:#22c47a;font-size:11px;
             font-weight:600;letter-spacing:.02em;">
      <svg id="snd-icon" width="13" height="13" viewBox="0 0 13 13" fill="none">
        <path d="M2 4.5H4.5L7.5 2V11L4.5 8.5H2V4.5Z" stroke="#22c47a" stroke-width="1.2" stroke-linejoin="round"/>
        <path d="M9 4.5C9.8 5.1 10.3 5.9 10.3 6.5C10.3 7.1 9.8 7.9 9 8.5" stroke="#22c47a" stroke-width="1.2" stroke-linecap="round"/>
      </svg>
      <span id="snd-txt">Enable sound</span>
    </button>
    <span class="dot" id="dot"></span>
    <span class="lbl" id="conn-lbl">Connecting...</span>
    <span class="badge" id="sess-badge">No session</span>
  </div>
</div>

<!-- Alert overlay toast -->
<div id="alert-toast" style="display:none;position:fixed;top:60px;right:20px;z-index:9999;
     background:#1a0808;border:1px solid #ef4444;border-radius:10px;
     padding:14px 18px;min-width:280px;max-width:340px;
     box-shadow:0 0 30px rgba(239,68,68,0.35);">
  <div style="display:flex;align-items:flex-start;gap:12px;">
    <div id="toast-icon" style="width:36px;height:36px;border-radius:50%;background:rgba(239,68,68,0.2);
         border:1.5px solid #ef4444;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
        <path d="M9 2L16.5 15H1.5L9 2Z" stroke="#ef4444" stroke-width="1.5" stroke-linejoin="round"/>
        <line x1="9" y1="7" x2="9" y2="11" stroke="#ef4444" stroke-width="1.5" stroke-linecap="round"/>
        <circle cx="9" cy="13.5" r="0.8" fill="#ef4444"/>
      </svg>
    </div>
    <div style="flex:1">
      <div style="font-size:13px;font-weight:700;color:#ef4444;margin-bottom:3px;">DEEPFAKE DETECTED</div>
      <div style="font-size:12px;color:#c9a0a0;" id="toast-msg">Person flagged</div>
      <div style="margin-top:8px;height:3px;background:#3a1010;border-radius:2px;overflow:hidden;">
        <div id="toast-bar" style="height:100%;background:#ef4444;border-radius:2px;
             transition:width linear;width:100%"></div>
      </div>
    </div>
    <button onclick="hideToast()" style="background:none;border:none;color:#636b74;
            cursor:pointer;font-size:16px;line-height:1;padding:0;margin-top:-2px;">✕</button>
  </div>
</div>

<!-- Screen flash overlay -->
<div id="flash-overlay" style="display:none;position:fixed;inset:0;z-index:9998;
     background:rgba(239,68,68,0.08);pointer-events:none;"></div>

<div class="body">
  <!-- LEFT -->
  <div class="left">
    <div class="ph">Overview</div>
    <div class="mgrid">
      <div class="mc"><div class="mc-l">Persons</div><div class="mc-v" id="m-p">0</div></div>
      <div class="mc"><div class="mc-l">Alerts</div><div class="mc-v" id="m-a" style="color:var(--muted)">0</div></div>
      <div class="mc"><div class="mc-l">Deepfakes</div><div class="mc-v" id="m-f" style="color:var(--muted)">0</div></div>
      <div class="mc"><div class="mc-l">FPS</div><div class="mc-v" id="m-fps" style="color:var(--amber)">—</div></div>
    </div>
    <div class="div"></div>
    <div class="ph" style="padding-top:10px">Progress</div>
    <div class="prog-wrap">
      <div class="prog-row"><span id="prog-lbl">Waiting...</span><strong id="prog-pct">0%</strong></div>
      <div class="prog-track"><div class="prog-fill" id="prog-fill"></div></div>
    </div>
    <div class="status-wrap">
      <div class="spill idle" id="spill">
        <span class="pdot" id="pdot"></span>
        <span id="s-txt">Idle — run main.py --dashboard</span>
      </div>
    </div>
    <div class="div"></div>
    <div class="ph" style="padding-top:10px">Session info</div>
    <div style="padding:0 13px 12px;display:flex;flex-direction:column;gap:5px">
      <div style="display:flex;justify-content:space-between">
        <span style="color:var(--muted);font-size:11px">Videos done</span>
        <span style="font-size:11px;color:var(--text)" id="s-vids">0</span>
      </div>
      <div style="display:flex;justify-content:space-between">
        <span style="color:var(--muted);font-size:11px">Frames</span>
        <span style="font-size:11px;color:var(--text)" id="s-frm">0</span>
      </div>
      <div style="display:flex;justify-content:space-between">
        <span style="color:var(--muted);font-size:11px">Duration</span>
        <span style="font-size:11px;color:var(--text)" id="s-dur">—</span>
      </div>
      <div style="display:flex;justify-content:space-between">
        <span style="color:var(--muted);font-size:11px">Threshold</span>
        <span style="font-size:11px;color:var(--amber)" id="s-thr">0.65</span>
      </div>
    </div>
  </div>

  <!-- CENTER -->
  <div class="center">
    <div style="display:flex;align-items:center;justify-content:space-between;
                padding:13px 16px 9px;border-bottom:1px solid var(--border)">
      <span style="font-size:12px;font-weight:600">Person tracking</span>
      <span style="font-size:11px;color:var(--muted)" id="p-count">0 tracked</span>
    </div>
    <div class="thead">
      <span class="th">ID</span>
      <span class="th">Score history</span>
      <span class="th">Fake score</span>
      <span class="th">Frames</span>
      <span class="th">Confidence</span>
      <span class="th">Verdict</span>
    </div>
    <div class="tbody" id="tbody">
      <div class="empty">
        <svg width="28" height="28" viewBox="0 0 28 28" fill="none" opacity=".3">
          <circle cx="14" cy="14" r="12" stroke="#636b74" stroke-width="1.5"/>
          <circle cx="14" cy="14" r="4" fill="#636b74"/>
          <line x1="14" y1="2" x2="14" y2="6" stroke="#636b74" stroke-width="1.5" stroke-linecap="round"/>
          <line x1="14" y1="22" x2="14" y2="26" stroke="#636b74" stroke-width="1.5" stroke-linecap="round"/>
          <line x1="2" y1="14" x2="6" y2="14" stroke="#636b74" stroke-width="1.5" stroke-linecap="round"/>
          <line x1="22" y1="14" x2="26" y2="14" stroke="#636b74" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
        <span>Waiting for detection to start...</span>
      </div>
    </div>
  </div>

  <!-- RIGHT -->
  <div class="right">
    <div class="ph">Event log</div>
    <div class="ev-list" id="ev-list">
      <div class="noev">No events yet</div>
    </div>
    <div class="file-info">
      <div class="fi-name" id="fi-name">No video loaded</div>
      <div class="fi-meta" id="fi-meta">—</div>
    </div>
  </div>
</div>

<script>
const scoreHist = {};
let lastState = null;
let startTs = null;
let connected = false;
let logCount = 0;

function poll() {
  fetch('/state')
    .then(r => r.json())
    .then(s => {
      if (!connected) {
        connected = true;
        document.getElementById('dot').className = 'dot live';
        document.getElementById('conn-lbl').textContent = 'Connected';
      }
      render(s);
    })
    .catch(() => {
      connected = false;
      document.getElementById('dot').className = 'dot';
      document.getElementById('conn-lbl').textContent = 'Reconnecting...';
    });
}

function render(s) {
  const prev = lastState;

  // Detect new session — reset alert tracking so old alerts don't re-fire
  if (prev && (s.session_counter || 0) !== (prev.session_counter || 0)) {
    // New session started — clear all previous alert tracking
    lastState = null;
  }

  lastState = s;

  // session badge
  if (s.session_id) document.getElementById('sess-badge').textContent = s.session_id.slice(-14);

  // status
  const spill = document.getElementById('spill');
  const pdot  = document.getElementById('pdot');
  const stxt  = document.getElementById('s-txt');
  if (s.running) {
    spill.className='spill running'; pdot.classList.add('blink');
    stxt.textContent = 'Running — processing video';
    if (!startTs) startTs = Date.now();
  } else if (s.status === 'done') {
    spill.className='spill done'; pdot.classList.remove('blink');
    stxt.textContent = 'Done — session complete';
    document.getElementById('dot').className = 'dot done';
  } else {
    spill.className='spill idle'; pdot.classList.remove('blink');
    stxt.textContent = 'Idle — run: python main.py --dashboard';
    startTs = null;
  }

  // metrics
  const persons = s.persons || [];
  const fakes   = persons.filter(p => p.is_deepfake).length;
  const alerts  = (s.alerts || []).length;
  document.getElementById('m-p').textContent = persons.length;
  document.getElementById('p-count').textContent = persons.length + ' tracked';
  const mA = document.getElementById('m-a');
  mA.textContent = alerts; mA.style.color = alerts > 0 ? 'var(--red)' : 'var(--muted)';
  const mF = document.getElementById('m-f');
  mF.textContent = fakes; mF.style.color = fakes > 0 ? 'var(--red)' : 'var(--muted)';
  document.getElementById('m-fps').textContent = s.fps ? s.fps.toFixed(0) : '—';

  // progress
  const pct = s.total_frames > 0 ? Math.min(100, Math.round(s.frame / s.total_frames * 100)) : 0;
  document.getElementById('prog-fill').style.width = pct + '%';
  document.getElementById('prog-pct').textContent = pct + '%';
  document.getElementById('prog-lbl').textContent =
    s.total_frames > 0 ? `Frame ${s.frame} / ${s.total_frames}` : 'Waiting...';

  // session info
  document.getElementById('s-vids').textContent = s.videos_done || 0;
  document.getElementById('s-frm').textContent  = s.frame || 0;
  document.getElementById('s-thr').textContent  = s.threshold || 0.65;
  if (startTs && s.running) {
    const sec = Math.round((Date.now() - startTs) / 1000);
    document.getElementById('s-dur').textContent = sec + 's';
  }

  // file info
  if (s.video_name) {
    document.getElementById('fi-name').textContent = s.video_name;
    document.getElementById('fi-meta').textContent =
      [s.resolution, s.fps ? s.fps.toFixed(0)+' fps' : '', s.total_frames ? s.total_frames+' frames' : '']
      .filter(Boolean).join(' · ');
  }

  // persons table
  if (persons.length > 0) {
    // update score history
    persons.forEach(p => {
      if (!scoreHist[p.id]) scoreHist[p.id] = [];
      const last = scoreHist[p.id];
      if (!last.length || last[last.length-1] !== p.score) {
        last.push(p.score || 0);
        if (last.length > 22) last.shift();
      }
    });

    document.getElementById('tbody').innerHTML = persons
      .sort((a,b) => a.id - b.id)
      .map(p => {
        const sc   = ((p.score || 0) * 100).toFixed(1);
        const conf = ((p.confidence || 0) * 100).toFixed(1);
        const barC = p.is_deepfake ? 'var(--red)' : 'var(--green)';
        const hist = scoreHist[p.id] || [];
        const bars = hist.map(v => {
          const h = Math.max(2, Math.round(v * 22));
          const c = v > (s.threshold||.65) ? '#ef4444' : v > 0.45 ? '#f59e0b' : '#22c47a';
          return `<div class="mb" style="height:${h}px;background:${c}"></div>`;
        }).join('');
        const vtag = p.is_deepfake
          ? '<span class="vtag fake">⚠ Deepfake</span>'
          : (p.frames_analyzed||0) < 3
            ? '<span class="vtag wait">Analyzing</span>'
            : '<span class="vtag real">✓ Real</span>';
        return `<div class="tr2">
          <span class="tid">ID-${p.id}</span>
          <div class="mini">${bars}</div>
          <div class="bar-wrap">
            <div class="bar-bg"><div class="bar-fill" style="width:${sc}%;background:${barC}"></div></div>
            <span class="bar-num">${sc}%</span>
          </div>
          <span style="font-size:12px;color:var(--muted)">${p.frames_analyzed||0}</span>
          <span style="font-size:12px;color:var(--muted)">${conf}%</span>
          ${vtag}
        </div>`;
      }).join('');
  }

  // alerts / event log — detect new alerts
  const curAlerts = s.alerts || [];
  const prevAlerts = prev ? (prev.alerts || []) : [];
  if (curAlerts.length > prevAlerts.length) {
    const newOnes = curAlerts.slice(0, curAlerts.length - prevAlerts.length);
    newOnes.forEach(a => {
      addLog('fake',
        `Deepfake detected — ID-${a.id}`,
        `Score: ${(a.score*100).toFixed(1)}%  ·  Frame ${a.frame}`,
        (a.score*100).toFixed(1) + '%'
      );
      triggerAlert(a);
    });
  }

  // session start event
  if (!prev || (prev.session_id !== s.session_id && s.session_id)) {
    addLog('info', 'Session started', s.video_name || '', null);
  }
  if (prev && prev.running && !s.running && s.status === 'done') {
    addLog('ok', 'Session complete', `${persons.length} persons tracked`, null);
  }
}

function addLog(type, title, meta, badge) {
  const list = document.getElementById('ev-list');
  const noEv = list.querySelector('.noev');
  if (noEv) noEv.remove();
  const el = document.createElement('div');
  el.className = `ev ${type}`;
  el.innerHTML = `<div class="ev-t ${type}">${title}</div>
    <div class="ev-m">${meta}</div>
    ${badge ? `<div class="ev-badge">${badge}</div>` : ''}`;
  list.prepend(el);
  logCount++;
  if (logCount > 60 && list.lastChild) list.removeChild(list.lastChild);
}

// ── Alert sound + visual ─────────────────────────────────────────────────────
let audioCtx = null;
let soundEnabled = false;
let toastTimer = null;

function unlockAudio() {
  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
    soundEnabled = true;
    // Play a tiny silent sound to fully unlock
    const buf = audioCtx.createBuffer(1, 1, 22050);
    const src = audioCtx.createBufferSource();
    src.buffer = buf;
    src.connect(audioCtx.destination);
    src.start(0);
    // Update button to show sound is on
    document.getElementById('snd-btn').style.background = '#0a2a0a';
    document.getElementById('snd-btn').style.borderColor = '#22c47a';
    document.getElementById('snd-txt').textContent = 'Sound on';
    // Play a test beep so user knows it works
    setTimeout(() => playBeep(440, 0.08, 0.15), 100);
    console.log('Audio unlocked');
  } catch(e) { console.warn('Audio unlock failed:', e); }
}

function playBeep(freq, vol, dur) {
  if (!audioCtx || !soundEnabled) return;
  try {
    if (audioCtx.state === 'suspended') audioCtx.resume();
    const osc  = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.type = 'sine';
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(0.001, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(vol, audioCtx.currentTime + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + dur);
    osc.start(audioCtx.currentTime);
    osc.stop(audioCtx.currentTime + dur + 0.05);
  } catch(e) {}
}

function playAlertSound() {
  if (!soundEnabled) return;
  // Urgent triple beep: high-high-low
  playBeep(1000, 0.4, 0.10);
  setTimeout(() => playBeep(1000, 0.4, 0.10), 160);
  setTimeout(() => playBeep(700,  0.5, 0.30), 320);
}

function flashScreen() {
  const el = document.getElementById('flash-overlay');
  el.style.display = 'block';
  el.style.opacity = '1';
  let count = 0;
  const iv = setInterval(() => {
    el.style.opacity = count % 2 === 0 ? '0' : '1';
    count++;
    if (count > 5) { clearInterval(iv); el.style.display = 'none'; }
  }, 120);
}

function showToast(a) {
  const toast = document.getElementById('alert-toast');
  const msg   = document.getElementById('toast-msg');
  const bar   = document.getElementById('toast-bar');
  msg.textContent = `Person ID-${a.id}  ·  Score: ${(a.score*100).toFixed(1)}%  ·  Frame ${a.frame}`;
  toast.style.display = 'block';
  bar.style.transition = 'none';
  bar.style.width = '100%';
  // animate progress bar down over 6 seconds
  setTimeout(() => {
    bar.style.transition = 'width 6s linear';
    bar.style.width = '0%';
  }, 50);
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(hideToast, 6200);
}

function hideToast() {
  document.getElementById('alert-toast').style.display = 'none';
  if (toastTimer) { clearTimeout(toastTimer); toastTimer = null; }
}

function triggerAlert(a) {
  playAlertSound();
  flashScreen();
  showToast(a);
  // shake the m-a metric card
  const el = document.getElementById('m-a');
  el.style.transform = 'scale(1.25)';
  el.style.color = 'var(--red)';
  setTimeout(() => { el.style.transform = 'scale(1)'; }, 300);
}

setInterval(poll, 800);
poll();
</script>
</body>
</html>"""

if __name__ == "__main__":
    print("\n  Detonix ZoomGuard — Live Dashboard")
    print("  ===================================")
    print(f"  Open in browser: http://localhost:{PORT}")
    print("  Then run:        python main.py")
    print("  Press Ctrl+C to stop\n")
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), _Handler) as httpd:
        httpd.serve_forever()