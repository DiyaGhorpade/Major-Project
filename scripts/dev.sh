#!/usr/bin/env bash
# Start the whole Money Plant stack: ML service :8001 -> backend :8000 -> frontend :5173.
#
#   ./scripts/dev.sh                 set up (first run only), then start all three services
#   ./scripts/dev.sh --setup-only    create venvs / install deps / check artifacts + SHAP report, then exit
#   ./scripts/dev.sh --no-frontend   start only the ML service and the backend
#
# Environment: PYTHON=/path/to/python (>= 3.12) to pick the interpreter used to create the venvs.
# Ctrl-C stops every service. Logs: .dev-logs/{ml,backend,frontend,explain}.log
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ML_DIR="$ROOT/money_plant_ml_pipeline"
BE_DIR="$ROOT/backend"
FE_DIR="$ROOT/frontend"
LOG_DIR="$ROOT/.dev-logs"

ML_PORT=8001
BE_PORT=8000
FE_PORT=5173
ML_WORKERS=4

SETUP_ONLY=0
NO_FRONTEND=0
for arg in "$@"; do
  case "$arg" in
    --setup-only) SETUP_ONLY=1 ;;
    --no-frontend) NO_FRONTEND=1 ;;
    -h|--help) sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $arg (use --setup-only, --no-frontend or --help)" >&2; exit 2 ;;
  esac
done

if [ -t 1 ]; then B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'; else B=""; G=""; Y=""; R=""; N=""; fi
info() { echo "${B}==>${N} $*"; }
ok()   { echo "${G}ok${N}  $*"; }
warn() { echo "${Y}warning:${N} $*" >&2; }
die()  { echo "${R}error:${N} $*" >&2; exit 1; }

ML_PY="$ML_DIR/.venv/bin/python"
BE_PY="$BE_DIR/.venv/bin/python"

# ---------------------------------------------------------------- prerequisites
# The pinned ML dependencies (requirements-tested.txt: numpy 2.5, scipy 1.18, xgboost 3.4, shap 0.54)
# need Python >= 3.12. The backend alone would run on 3.11, but both venvs use the same interpreter.
find_python() {
  local c
  # 3.13 is the version the pinned ML dependencies were tested on, so it is preferred.
  for c in "${PYTHON:-}" python3.13 python3.12 python3.14 python3 python; do
    [ -n "$c" ] || continue
    command -v "$c" >/dev/null 2>&1 || continue
    if "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' >/dev/null 2>&1; then
      command -v "$c"
      return 0
    fi
  done
  return 1
}

check_prerequisites() {
  info "Checking prerequisites"
  if ! SYS_PY="$(find_python)"; then
    die "Python >= 3.12 not found. The pinned ML dependencies (numpy 2.5 / scipy 1.18 / xgboost 3.4 / shap 0.54)
       need Python 3.12 or newer. Install it (e.g. 'sudo apt install python3.12 python3.12-venv', 'brew install python@3.13',
       or conda/pyenv) and re-run, optionally with PYTHON=/path/to/python3.12 ./scripts/dev.sh"
  fi
  "$SYS_PY" -c 'import venv, ensurepip' >/dev/null 2>&1 \
    || die "The 'venv'/'ensurepip' modules are missing for $SYS_PY (on Debian/Ubuntu: sudo apt install python3-venv)."
  ok "python: $SYS_PY ($("$SYS_PY" -c 'import platform; print(platform.python_version())'))"

  if [ "$NO_FRONTEND" -eq 0 ] || [ "$SETUP_ONLY" -eq 1 ]; then
    command -v node >/dev/null 2>&1 || die "Node.js >= 20 not found. Install it from https://nodejs.org or with your package manager / nvm."
    local major
    major="$(node -p 'process.versions.node.split(".")[0]')"
    [ "$major" -ge 20 ] || die "Node.js >= 20 is required (found $(node -v))."
    command -v npm >/dev/null 2>&1 || die "npm not found (it normally ships with Node.js)."
    ok "node: $(node -v), npm: $(npm -v)"
  fi
  [ -f "$ML_DIR/requirements-tested.txt" ] || die "Missing $ML_DIR/requirements-tested.txt: is this the full repository?"
}

# ---------------------------------------------------------------- helpers
hash_files() { # content hash of the given files (+ the interpreter version, so a new Python reinstalls)
  { for f in "$@"; do cat "$f"; done; "$SYS_PY" -c 'import sys; print(sys.version)'; } | "$SYS_PY" -c 'import hashlib,sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())'
}

ensure_venv() { # name dir requirements-file...
  local name="$1" dir="$2"; shift 2
  local marker="$dir/.venv/.deps-hash" want have
  if [ ! -x "$dir/.venv/bin/python" ]; then
    info "Creating $name virtualenv ($dir/.venv)"
    "$SYS_PY" -m venv "$dir/.venv" || die "Could not create the $name virtualenv."
  fi
  want="$(hash_files "$@")"
  have="$(cat "$marker" 2>/dev/null || true)"
  if [ "$want" != "$have" ]; then
    info "Installing $name dependencies (first run takes a few minutes)"
    local args=() f
    for f in "$@"; do args+=(-r "$f"); done
    "$dir/.venv/bin/python" -m pip install --disable-pip-version-check -q --upgrade pip \
      || warn "could not upgrade pip, continuing"
    "$dir/.venv/bin/python" -m pip install --disable-pip-version-check -q "${args[@]}" \
      || die "Installing the $name dependencies failed (see pip output above)."
    echo "$want" > "$marker"
  fi
  ok "$name environment ready"
}

ensure_frontend() {
  local marker="$FE_DIR/node_modules/.dev-lock-hash" want have
  want="$("$SYS_PY" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$FE_DIR/package-lock.json")"
  have="$(cat "$marker" 2>/dev/null || true)"
  if [ ! -d "$FE_DIR/node_modules" ] || [ "$want" != "$have" ]; then
    info "Installing frontend dependencies (npm ci)"
    (cd "$FE_DIR" && npm ci --no-audit --no-fund) || die "npm ci failed in $FE_DIR."
    echo "$want" > "$marker"
  fi
  ok "frontend dependencies ready"
}

latest_run_id() {
  "$SYS_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$ML_DIR/artifacts/latest.json" 2>/dev/null
}

check_artifacts() {
  local run=""
  if [ -f "$ML_DIR/artifacts/latest.json" ]; then run="$(latest_run_id || true)"; fi
  if [ -z "$run" ] || [ ! -f "$ML_DIR/artifacts/$run/manifest.json" ]; then
    {
      echo "${R}error:${N} the ML model artifacts are missing (artifacts/latest.json or its run directory)."
      echo "Training is an explicit step and is never run automatically. Train the models once, then re-run this script:"
      echo
      echo "    cd money_plant_ml_pipeline"
      echo "    .venv/bin/python scripts/train_all.py --input data/raw/plant_sensor_data.csv"
      echo
    } >&2
    exit 1
  fi
  RUN_ID="$run"
  ok "ML artifacts: run $RUN_ID"
}

ensure_shap_report() {
  if compgen -G "$ML_DIR/reports/$RUN_ID/evaluation_*/shap_global_importance.json" >/dev/null; then
    ok "global SHAP report present"
    return 0
  fi
  info "Generating the global SHAP report for $RUN_ID (one-off, explanation only: no training; takes a minute or two)"
  mkdir -p "$LOG_DIR"
  if (cd "$ML_DIR" && "$ML_PY" scripts/generate_explanations.py --input data/raw/plant_sensor_data.csv --run-id "$RUN_ID") \
      > "$LOG_DIR/explain.log" 2>&1 \
     && compgen -G "$ML_DIR/reports/$RUN_ID/evaluation_*/shap_global_importance.json" >/dev/null; then
    ok "global SHAP report written (log: .dev-logs/explain.log)"
  else
    warn "could not generate the global SHAP report (see .dev-logs/explain.log). The app still works; the Global SHAP page will show its empty state."
    tail -n 5 "$LOG_DIR/explain.log" >&2 || true
  fi
}

port_in_use() { # port
  "$SYS_PY" - "$1" <<'EOF'
import socket, sys
s = socket.socket()
s.settimeout(0.5)
try:
    sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
finally:
    s.close()
EOF
}

check_ports() {
  local busy=0 spec port name
  for spec in "$ML_PORT:ML service" "$BE_PORT:backend" "$@"; do
    port="${spec%%:*}"; name="${spec#*:}"
    if port_in_use "$port"; then
      echo "${R}error:${N} port $port ($name) is already in use." >&2
      command -v ss >/dev/null 2>&1 && ss -ltnp 2>/dev/null | grep -E ":$port\b" >&2 || true
      busy=1
    fi
  done
  [ "$busy" -eq 0 ] || die "Free the port(s) above (stop the other process) and re-run."
}

http_get() { # url -> body on stdout, non-zero on failure
  "$SYS_PY" - "$1" <<'EOF'
import sys, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=3) as r:
        sys.stdout.write(r.read().decode())
except Exception:
    sys.exit(1)
EOF
}

# ---------------------------------------------------------------- process management
PGIDS=()
CLEANED=0

start_group() { # logname command... : run in its own process group, output to .dev-logs/<logname>.log, echoed with a prefix
  local name="$1"; shift
  local log="$LOG_DIR/$name.log" pid
  : > "$log"
  # live, prefixed view of the log without the routine 2xx/3xx access lines
  start_raw bash -c 'tail -n0 -F "$1" 2>/dev/null | grep --line-buffered -v -E "HTTP/1\.[01]\" (2|3)[0-9][0-9]| -> (2|3)[0-9][0-9] \(" | sed -u "s/^/[$2] /"' _ "$log" "$name"
  start_raw bash -c 'exec "$@" >> "$0" 2>&1' "$log" "$@"
  LAST_PID="$LAST_PGID"
}

start_raw() { # command... : background process in its own process group; sets LAST_PGID
  if command -v setsid >/dev/null 2>&1; then
    setsid "$@" &
    LAST_PGID=$!
  else
    set -m
    "$@" &
    LAST_PGID=$!
    set +m
  fi
  PGIDS+=("$LAST_PGID")
}

cleanup() {
  local code=$?
  trap - INT TERM EXIT
  if [ "$CLEANED" -eq 0 ] && [ "${#PGIDS[@]}" -gt 0 ]; then
    CLEANED=1
    echo
    info "Stopping services"
    local pg i
    for pg in "${PGIDS[@]}"; do kill -TERM -- "-$pg" 2>/dev/null || true; done
    for i in 1 2 3 4 5 6 7 8 9 10; do # up to ~5 s for a clean exit
      local alive=0
      for pg in "${PGIDS[@]}"; do kill -0 -- "-$pg" 2>/dev/null && alive=1; done
      [ "$alive" -eq 0 ] && break
      sleep 0.5
    done
    for pg in "${PGIDS[@]}"; do kill -KILL -- "-$pg" 2>/dev/null || true; done
    ok "all services stopped"
  fi
  exit "$code"
}

wait_for() { # label pid url jq-ish-python-condition timeout-seconds logfile
  local label="$1" pid="$2" url="$3" cond="$4" timeout="$5" log="$6" i body
  for ((i = 0; i < timeout * 2; i++)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "${R}error:${N} the $label process exited during startup. Last log lines:" >&2
      tail -n 25 "$log" >&2 || true
      exit 1
    fi
    if body="$(http_get "$url" 2>/dev/null)" && printf '%s' "$body" | "$SYS_PY" -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if ($cond) else 1)" 2>/dev/null; then
      return 0
    fi
    sleep 0.5
  done
  echo "${R}error:${N} $label did not become ready within ${timeout}s ($url). Last log lines:" >&2
  tail -n 25 "$log" >&2 || true
  exit 1
}

# ---------------------------------------------------------------- main
cd "$ROOT"
check_prerequisites
ensure_venv "ML" "$ML_DIR" "$ML_DIR/requirements-tested.txt"
ensure_venv "backend" "$BE_DIR" "$BE_DIR/requirements.txt"
if [ "$NO_FRONTEND" -eq 0 ] || [ "$SETUP_ONLY" -eq 1 ]; then ensure_frontend; fi
check_artifacts
mkdir -p "$LOG_DIR"
ensure_shap_report

if [ "$SETUP_ONLY" -eq 1 ]; then
  ok "setup complete. Start everything with ./scripts/dev.sh"
  exit 0
fi

if [ "$NO_FRONTEND" -eq 1 ]; then check_ports; else check_ports "$FE_PORT:frontend"; fi
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

info "Starting the ML service on :$ML_PORT ($ML_WORKERS workers)"
cd "$ML_DIR"
start_group ml env PYTHONPATH=src "$ML_PY" -m uvicorn plant_ml.api.main:app --host 127.0.0.1 --port "$ML_PORT" --workers "$ML_WORKERS"
ML_PID="$LAST_PID"
cd "$ROOT"
wait_for "ML service" "$ML_PID" "http://127.0.0.1:$ML_PORT/health" 'd.get("status") == "ok" and d.get("models_loaded") is True' 120 "$LOG_DIR/ml.log"
ok "ML service healthy"

info "Starting the backend on :$BE_PORT (single process)"
cd "$BE_DIR"
start_group backend "$BE_PY" -m uvicorn app.main:app --host 127.0.0.1 --port "$BE_PORT"
BE_PID="$LAST_PID"
cd "$ROOT"
wait_for "backend" "$BE_PID" "http://127.0.0.1:$BE_PORT/api/health" 'd.get("status") in ("ok", "degraded")' 90 "$LOG_DIR/backend.log"
ok "backend healthy"

FE_PID=""
if [ "$NO_FRONTEND" -eq 0 ]; then
  info "Starting the frontend on :$FE_PORT"
  cd "$FE_DIR"
  start_group frontend npm run dev -- --host 127.0.0.1 --port "$FE_PORT" --strictPort
  FE_PID="$LAST_PID"
  cd "$ROOT"
  wait_for "frontend" "$FE_PID" "http://127.0.0.1:$FE_PORT/api/health" 'd.get("status") in ("ok", "degraded")' 60 "$LOG_DIR/frontend.log"
  ok "frontend healthy (proxying /api to the backend)"
fi

echo
echo "${B}Money Plant Monitoring is running${N}"
[ "$NO_FRONTEND" -eq 0 ] && echo "  App:      http://127.0.0.1:$FE_PORT    <- open this and press ${B}Run${N} in the top bar"
echo "  Backend:  http://127.0.0.1:$BE_PORT/docs"
echo "  ML:       http://127.0.0.1:$ML_PORT/health"
echo "  Logs:     $LOG_DIR/"
echo "Press Ctrl-C to stop everything."
echo

# The backend and frontend are designed to degrade gracefully, so an ML crash only warns; losing either of them stops everything.
MONITORED=("$BE_PID")
[ -n "$FE_PID" ] && MONITORED+=("$FE_PID")
ML_WARNED=0
while true; do
  sleep 1 &
  wait $! || true
  for pid in "${MONITORED[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "${R}error:${N} a service (pid $pid) exited unexpectedly. See $LOG_DIR/." >&2
      exit 1
    fi
  done
  if [ "$ML_WARNED" -eq 0 ] && ! kill -0 "$ML_PID" 2>/dev/null; then
    ML_WARNED=1
    warn "the ML service exited (see $LOG_DIR/ml.log). The app keeps serving stored data; a Run will fail until it is back. Restart with Ctrl-C and ./scripts/dev.sh."
  fi
done
