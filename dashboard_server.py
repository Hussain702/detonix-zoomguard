"""
Detonix ZoomGuard — Live Dashboard Server
Run alongside main.py to see real-time results in your browser.

Usage:
  Terminal 1: python dashboard_server.py
  Terminal 2: python main.py
  Browser:    http://localhost:5000
"""

import http.server
import json
import os
import threading
import time
import socketserver
from datetime import datetime

PORT = 5000
LIVE_FILE = "logs/live_state.json"

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Detonix ZoomGuard — Live Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {
    --bg:       #0a0c0f;
    --bg2:      #111318;
    --bg3:      #181c23;
    --border:   rgba(255,255,255,0.07);
    --border2:  rgba(255,255,255,0.12);
    --text:     #e8eaf0;
    --muted:    #6b7280;
    --accent:   #00d4a0;
    --accent2:  #0ea5e9;
    --danger:   #f87171;
    --warning:  #fbbf24;
    --real:     #34d399;
    --fake:     #f87171;
    --purple:   #a78bfa;
    --font:     'Inter', system-ui, sans-serif;
    --mono:     'JetBrains Mono', monospace;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: var(--font);
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    font-size: 13px;
    line-height: 1.5;
  }

  /* ── Header ── */
  .header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 28px;
    border-bottom: 1px solid var(--border);
    background: var(--bg2);
    position: sticky; top: 0; z-index: 10;
  }
  .header-left { display: flex; align-items: center; gap: 14px; }
  .logo {
    width: 34px; height: 34px; border-radius: 9px;
    background: linear-gradient(135deg, #00d4a0, #0ea5e9);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
  }
  .logo svg { width: 18px; height: 18px; }
  .brand-name { font-size: 15px; font-weight: 600; color: var(--text); letter-spacing: -0.2px; }
  .brand-sub  { font-size: 11px; color: var(--muted); margin-top: 1px; }
  .header-right { display: flex; align-items: center; gap: 12px; }
  .status-dot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--muted);
    box-shadow: 0 0 0 2px rgba(107,114,128,0.2);
    transition: all 0.3s;
  }
  .status-dot.live {
    background: var(--accent);
    box-shadow: 0 0 0 3px rgba(0,212,160,0.2), 0 0 12px rgba(0,212,160,0.4);
    animation: pulse 2s infinite;
  }
  .status-dot.alert { background: var(--danger); box-shadow: 0 0 0 3px rgba(248,113,113,0.25), 0 0 12px rgba(248,113,113,0.4); animation: pulse 1s infinite; }
  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.5; } }
  .status-text { font-size: 11px; color: var(--muted); font-family: var(--mono); }

  /* ── Layout ── */
  .main { padding: 24px 28px; max-width: 1200px; margin: 0 auto; }

  /* ── Metric cards ── */
  .metrics { display: grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap: 12px; margin-bottom: 20px; }
  .metric-card {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px 18px;
    transition: border-color 0.3s;
  }
  .metric-card:hover { border-color: var(--border2); }
  .metric-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 8px; }
  .metric-value { font-size: 26px; font-weight: 600; color: var(--text); line-height: 1; font-variant-numeric: tabular-nums; }
  .metric-value.green  { color: var(--real); }
  .metric-value.red    { color: var(--fake); }
  .metric-value.blue   { color: var(--accent2); }
  .metric-value.purple { color: var(--purple); }
  .metric-sub { font-size: 10px; color: var(--muted); margin-top: 5px; font-family: var(--mono); }

  /* ── Panels ── */
  .grid2 { display: grid; grid-template-columns: 1fr 380px; gap: 16px; margin-bottom: 16px; }
  .grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .panel {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden;
  }
  .panel-head {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 18px;
    border-bottom: 1px solid var(--border);
  }
  .panel-title { font-size: 12px; font-weight: 500; color: var(--text); letter-spacing: 0.02em; }
  .panel-badge {
    font-size: 10px; font-family: var(--mono);
    background: var(--bg3); color: var(--muted);
    padding: 2px 8px; border-radius: 99px;
    border: 1px solid var(--border);
  }
  .panel-body { padding: 16px 18px; }

  /* ── Person table ── */
  .person-table { width: 100%; border-collapse: collapse; }
  .person-table th {
    text-align: left; font-size: 10px; text-transform: uppercase;
    letter-spacing: 0.07em; color: var(--muted);
    padding: 0 0 10px; font-weight: 400;
  }
  .person-table td { padding: 9px 0; border-top: 1px solid var(--border); vertical-align: middle; }
  .person-table tr:first-child td { border-top: none; }

  .pid-cell { font-family: var(--mono); font-size: 12px; color: var(--purple); font-weight: 500; }
  .score-cell { padding-right: 12px !important; }
  .bar-wrap { display: flex; align-items: center; gap: 8px; }
  .bar-bg { flex: 1; height: 3px; background: var(--bg3); border-radius: 99px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 99px; transition: width 0.6s ease; }
  .bar-pct { font-family: var(--mono); font-size: 11px; color: var(--muted); min-width: 38px; text-align: right; }
  .frames-cell { font-family: var(--mono); font-size: 11px; color: var(--muted); }

  .pill {
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 10px; font-weight: 500; padding: 3px 9px;
    border-radius: 99px; letter-spacing: 0.03em;
  }
  .pill-real    { background: rgba(52,211,153,0.1);  color: var(--real);  border: 1px solid rgba(52,211,153,0.2); }
  .pill-fake    { background: rgba(248,113,113,0.1); color: var(--fake);  border: 1px solid rgba(248,113,113,0.2); }
  .pill-analyze { background: rgba(107,114,128,0.1); color: var(--muted); border: 1px solid var(--border); }

  /* ── Mini sparkline chart ── */
  .sparkline-wrap { position: relative; height: 160px; }

  /* ── Alert log ── */
  .alert-item {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 10px 0; border-bottom: 1px solid var(--border);
  }
  .alert-item:last-child { border-bottom: none; padding-bottom: 0; }
  .alert-icon {
    width: 24px; height: 24px; border-radius: 6px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    background: rgba(248,113,113,0.12); margin-top: 1px;
  }
  .alert-title { font-size: 12px; font-weight: 500; color: var(--text); }
  .alert-meta  { font-size: 10px; color: var(--muted); margin-top: 2px; font-family: var(--mono); }
  .alert-score { font-family: var(--mono); font-size: 11px; color: var(--fake); margin-top: 2px; }
  .empty-state { text-align: center; padding: 24px 0; color: var(--muted); font-size: 12px; }
  .empty-icon  { font-size: 24px; margin-bottom: 8px; opacity: 0.4; }

  /* ── Progress bar ── */
  .progress-section { margin-bottom: 20px; }
  .progress-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
  .progress-label { font-size: 11px; color: var(--muted); }
  .progress-pct   { font-family: var(--mono); font-size: 11px; color: var(--accent); }
  .progress-bg { height: 4px; background: var(--bg3); border-radius: 99px; overflow: hidden; }
  .progress-fill { height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent2)); border-radius: 99px; transition: width 0.5s ease; }

  /* ── FPS bar ── */
  .fps-bar { display: flex; align-items: center; gap: 10px; margin-top: 10px; }
  .fps-val  { font-family: var(--mono); font-size: 20px; font-weight: 600; color: var(--accent2); }
  .fps-label{ font-size: 10px; color: var(--muted); }

  /* ── Idle state ── */
  .idle-screen {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    min-height: 60vh; text-align: center; gap: 12px;
  }
  .idle-logo {
    width: 56px; height: 56px; border-radius: 16px;
    background: linear-gradient(135deg, #00d4a0, #0ea5e9);
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 8px;
  }
  .idle-title { font-size: 18px; font-weight: 600; color: var(--text); }
  .idle-sub   { font-size: 13px; color: var(--muted); max-width: 340px; line-height: 1.6; }
  .idle-code  {
    font-family: var(--mono); font-size: 12px;
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 20px; color: var(--accent);
    margin-top: 8px;
  }
  .blink { animation: blink 1.2s step-end infinite; }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

  /* ── Responsive ── */
  @media (max-width: 900px) {
    .metrics { grid-template-columns: repeat(3, 1fr); }
    .grid2   { grid-template-columns: 1fr; }
  }
  @media (max-width: 600px) {
    .metrics { grid-template-columns: repeat(2, 1fr); }
    .main    { padding: 16px; }
    .header  { padding: 12px 16px; }
  }
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <div class="logo">
      <svg viewBox="0 0 18 18" fill="none">
        <circle cx="9" cy="9" r="5.5" stroke="white" stroke-width="1.5"/>
        <circle cx="9" cy="9" r="2" fill="white"/>
        <line x1="9" y1="1.5" x2="9" y2="4" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
        <line x1="9" y1="14" x2="9" y2="16.5" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
        <line x1="1.5" y1="9" x2="4" y2="9" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
        <line x1="14" y1="9" x2="16.5" y2="9" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
      </svg>
    </div>
    <div>
      <div class="brand-name">Detonix ZoomGuard</div>
      <div class="brand-sub">Real-time deepfake detection · QAU CS Final Year</div>
    </div>
  </div>
  <div class="header-right">
    <div id="status-dot" class="status-dot"></div>
    <span id="status-text" class="status-text">waiting...</span>
  </div>
</div>

<div class="main" id="main-content">
  <div class="idle-screen" id="idle-screen">
    <div class="idle-logo">
      <svg width="28" height="28" viewBox="0 0 18 18" fill="none">
        <circle cx="9" cy="9" r="5.5" stroke="white" stroke-width="1.5"/>
        <circle cx="9" cy="9" r="2" fill="white"/>
        <line x1="9" y1="1.5" x2="9" y2="4" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
        <line x1="9" y1="14" x2="9" y2="16.5" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
        <line x1="1.5" y1="9" x2="4" y2="9" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
        <line x1="14" y1="9" x2="16.5" y2="9" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
      </svg>
    </div>
    <div class="idle-title">Dashboard ready<span class="blink">_</span></div>
    <div class="idle-sub">Run <strong>python main.py</strong> in your terminal to start processing. Results will appear here live.</div>
    <div class="idle-code">python main.py</div>
  </div>

  <div id="live-content" style="display:none;">
    <div class="progress-section">
      <div class="progress-head">
        <span class="progress-label" id="prog-file">Processing...</span>
        <span class="progress-pct" id="prog-pct">0%</span>
      </div>
      <div class="progress-bg"><div class="progress-fill" id="prog-fill" style="width:0%"></div></div>
    </div>

    <div class="metrics">
      <div class="metric-card">
        <div class="metric-label">Frame</div>
        <div class="metric-value blue" id="m-frame">0</div>
        <div class="metric-sub" id="m-total">of 0</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Speed</div>
        <div class="metric-value blue" id="m-fps">0</div>
        <div class="metric-sub">frames / sec</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Persons</div>
        <div class="metric-value purple" id="m-persons">0</div>
        <div class="metric-sub">tracked</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Alerts</div>
        <div class="metric-value" id="m-alerts" style="color:var(--real)">0</div>
        <div class="metric-sub">deepfakes flagged</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Duration</div>
        <div class="metric-value blue" id="m-dur">0s</div>
        <div class="metric-sub" id="m-session">session</div>
      </div>
    </div>

    <div class="grid2">
      <div class="panel">
        <div class="panel-head">
          <span class="panel-title">Person tracking</span>
          <span class="panel-badge" id="persons-count">0 persons</span>
        </div>
        <div class="panel-body" id="persons-table-wrap">
          <div class="empty-state"><div class="empty-icon">◎</div>No faces detected yet</div>
        </div>
      </div>

      <div class="panel">
        <div class="panel-head">
          <span class="panel-title">Detection alerts</span>
          <span class="panel-badge" id="alerts-count">0 alerts</span>
        </div>
        <div class="panel-body" id="alerts-wrap">
          <div class="empty-state"><div class="empty-icon">◎</div>No alerts yet</div>
        </div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-head">
        <span class="panel-title">Score history</span>
        <span class="panel-badge">last 60 data points per person</span>
      </div>
      <div class="panel-body">
        <div style="position:relative; height:200px;">
          <canvas id="score-chart" role="img" aria-label="Line chart of deepfake scores over time per tracked person">Score history over time.</canvas>
        </div>
      </div>
    </div>

  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
let chart = null;
let chartData = {};  // track_id -> [scores]
let prevAlertCount = 0;
let isLive = false;

function fmt(n) { return typeof n === 'number' ? n.toFixed(1) : n; }

function updateProgress(state) {
  const pct = state.total_frames > 0
    ? Math.round((state.frame / state.total_frames) * 100) : 0;
  document.getElementById('prog-fill').style.width = pct + '%';
  document.getElementById('prog-pct').textContent  = pct + '%';
  document.getElementById('prog-file').textContent =
    (state.video_name || 'Processing...').split(/[\\/]/).pop();
}

function updateMetrics(state) {
  document.getElementById('m-frame').textContent   = state.frame || 0;
  document.getElementById('m-total').textContent   = 'of ' + (state.total_frames || 0);
  document.getElementById('m-fps').textContent     = fmt(state.fps || 0);
  document.getElementById('m-persons').textContent = (state.persons || []).length;
  const alerts = (state.alerts || []).length;
  const el = document.getElementById('m-alerts');
  el.textContent = alerts;
  el.style.color = alerts > 0 ? 'var(--fake)' : 'var(--real)';
  document.getElementById('m-dur').textContent = (state.duration || 0) + 's';
  document.getElementById('m-session').textContent = state.session_id || '';
}

function updatePersons(persons) {
  const wrap = document.getElementById('persons-table-wrap');
  document.getElementById('persons-count').textContent = persons.length + ' person' + (persons.length !== 1 ? 's' : '');

  if (!persons.length) {
    wrap.innerHTML = '<div class="empty-state"><div class="empty-icon">◎</div>No faces detected yet</div>';
    return;
  }

  let html = `<table class="person-table">
    <thead><tr>
      <th>ID</th><th>Fake score</th><th>Frames</th><th>Verdict</th>
    </tr></thead><tbody>`;

  persons.forEach(p => {
    const score  = (p.score * 100);
    const barClr = p.verdict === 'fake' ? '#f87171' : p.verdict === 'real' ? '#34d399' : '#6b7280';
    const pill   = p.verdict === 'fake'
      ? '<span class="pill pill-fake">Deepfake</span>'
      : p.verdict === 'real'
      ? '<span class="pill pill-real">Real</span>'
      : '<span class="pill pill-analyze">Analyzing</span>';

    html += `<tr>
      <td class="pid-cell">ID-${p.id}</td>
      <td class="score-cell">
        <div class="bar-wrap">
          <div class="bar-bg"><div class="bar-fill" style="width:${score.toFixed(1)}%;background:${barClr}"></div></div>
          <span class="bar-pct">${score.toFixed(1)}%</span>
        </div>
      </td>
      <td class="frames-cell">${p.frames_analyzed}</td>
      <td>${pill}</td>
    </tr>`;
  });

  html += '</tbody></table>';
  wrap.innerHTML = html;
}

function updateAlerts(alerts) {
  const wrap = document.getElementById('alerts-wrap');
  document.getElementById('alerts-count').textContent = alerts.length + ' alert' + (alerts.length !== 1 ? 's' : '');

  if (!alerts.length) {
    wrap.innerHTML = '<div class="empty-state"><div class="empty-icon">◎</div>No alerts — looking good</div>';
    return;
  }

  // Flash header if new alert
  if (alerts.length > prevAlertCount) {
    document.getElementById('status-dot').className = 'status-dot alert';
    setTimeout(() => document.getElementById('status-dot').className = 'status-dot live', 3000);
  }
  prevAlertCount = alerts.length;

  const last10 = alerts.slice(-10).reverse();
  wrap.innerHTML = last10.map(a => `
    <div class="alert-item">
      <div class="alert-icon">
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
          <path d="M6 1L11 10H1L6 1Z" stroke="#f87171" stroke-width="1.2" stroke-linejoin="round"/>
          <line x1="6" y1="5" x2="6" y2="7.5" stroke="#f87171" stroke-width="1.2" stroke-linecap="round"/>
          <circle cx="6" cy="9" r="0.6" fill="#f87171"/>
        </svg>
      </div>
      <div>
        <div class="alert-title">Person ID-${a.track_id} flagged as deepfake</div>
        <div class="alert-meta">Frame ${a.frame} · ${(a.video || '').split(/[\\/]/).pop()}</div>
        <div class="alert-score">Score: ${(a.score * 100).toFixed(1)}%</div>
      </div>
    </div>`).join('');
}

function updateChart(persons) {
  if (!chart) return;
  const colors = ['#00d4a0','#0ea5e9','#a78bfa','#f87171','#fbbf24','#34d399','#fb7185','#60a5fa'];

  persons.forEach((p, i) => {
    const key = 'ID-' + p.id;
    if (!chartData[key]) {
      chartData[key] = [];
      chart.data.datasets.push({
        label: key,
        data: chartData[key],
        borderColor: colors[i % colors.length],
        backgroundColor: 'transparent',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.4,
      });
    }
    if (p.score !== undefined) {
      chartData[key].push((p.score * 100));
      if (chartData[key].length > 60) chartData[key].shift();
    }
  });

  chart.data.labels = Array.from({ length: 60 }, (_, i) => i);
  chart.update('none');
}

function initChart() {
  const ctx = document.getElementById('score-chart').getContext('2d');
  chart = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: {
        legend: { display: true, position: 'top',
          labels: { color: '#6b7280', font: { size: 11, family: 'JetBrains Mono' },
                    boxWidth: 12, boxHeight: 2, padding: 16 } },
        tooltip: { mode: 'index', intersect: false,
          callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}%` } }
      },
      scales: {
        x: { display: false },
        y: {
          min: 0, max: 100,
          grid: { color: 'rgba(255,255,255,0.05)' },
          border: { display: false },
          ticks: { color: '#6b7280', font: { size: 10, family: 'JetBrains Mono' },
                   callback: v => v + '%', stepSize: 25 }
        }
      }
    }
  });
}

function poll() {
  fetch('/state')
    .then(r => r.json())
    .then(state => {
      const running = state.running;
      const persons = state.persons || [];
      const alerts  = state.alerts  || [];

      if (running && !isLive) {
        isLive = true;
        document.getElementById('idle-screen').style.display  = 'none';
        document.getElementById('live-content').style.display = 'block';
        if (!chart) initChart();
      }

      if (isLive) {
        document.getElementById('status-dot').className  = 'status-dot live';
        document.getElementById('status-text').textContent = running ? 'live' : 'done';

        updateProgress(state);
        updateMetrics(state);
        updatePersons(persons);
        updateAlerts(alerts);
        updateChart(persons);
      }

      if (!running && isLive) {
        document.getElementById('status-text').textContent = 'done';
        document.getElementById('status-dot').className = 'status-dot';
      }
    })
    .catch(() => {});
}

setInterval(poll, 800);
poll();
</script>
</body>
</html>"""

# ── Live state written by orchestrator ───────────────────────────────────────
_state = {
    "running": False, "frame": 0, "total_frames": 0,
    "fps": 0.0, "duration": 0, "session_id": "",
    "video_name": "", "persons": [], "alerts": []
}
_lock = threading.Lock()

def update_state(**kwargs):
    with _lock:
        _state.update(kwargs)
    os.makedirs("logs", exist_ok=True)
    with open(LIVE_FILE, "w") as f:
        json.dump(_state, f)

def get_state():
    try:
        with open(LIVE_FILE) as f:
            return json.load(f)
    except Exception:
        return _state

# ── HTTP handler ──────────────────────────────────────────────────────────────
class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode())

        elif self.path == "/state":
            data = json.dumps(get_state()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        else:
            self.send_response(404); self.end_headers()


def start_server():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.allow_reuse_address = True
        print(f"\n  Detonix ZoomGuard Dashboard")
        print(f"  Open in browser: http://localhost:{PORT}")
        print(f"  Now run:         python main.py\n")
        httpd.serve_forever()

if __name__ == "__main__":
    start_server()