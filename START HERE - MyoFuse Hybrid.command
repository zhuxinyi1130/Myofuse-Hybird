#!/bin/zsh
set -euo pipefail

script_dir="${0:A:h}"
cd "$script_dir"

show_message() {
  /usr/bin/osascript -e "display dialog \"$1\" with title \"MyoFuse Hybrid\" buttons {\"OK\"} default button \"OK\"" >/dev/null 2>&1 || true
}

runtime_ready() {
  local candidate="$1"
  [[ -x "$candidate" ]] &&
    "$candidate" -c \
      'import flask, torch, cv2, pandas; from aicsimageio import AICSImage; from cellpose import models; from napari_svetlana.CNN2D import CNN2D' \
      >/dev/null 2>&1
}

find_existing_runtime() {
  local candidate
  for candidate in \
    "${script_dir}/.venv/bin/python" \
    "${script_dir}/../work/myofuse-venv/bin/python" \
    "${script_dir}/../../../work/myofuse-venv/bin/python"
  do
    if runtime_ready "$candidate"; then
      print -r -- "$candidate"
      return 0
    fi
  done
  return 1
}

find_python39() {
  local candidate
  for candidate in \
    /opt/homebrew/bin/python3.9 \
    /usr/local/bin/python3.9 \
    /Library/Frameworks/Python.framework/Versions/3.9/bin/python3.9 \
    "${commands[python3.9]:-}" \
    "${commands[python3]:-}"
  do
    if [[ -n "$candidate" && -x "$candidate" ]] &&
       "$candidate" -c 'import sys; assert sys.version_info[:2] == (3, 9)' >/dev/null 2>&1; then
      print -r -- "$candidate"
      return 0
    fi
  done
  return 1
}

python_bin="$(find_existing_runtime || true)"

if [[ -z "$python_bin" ]]; then
  python39="$(find_python39 || true)"
  if [[ -z "$python39" ]]; then
    choice="$(/usr/bin/osascript <<'APPLESCRIPT' 2>/dev/null || true
display dialog "MyoFuse Hybrid needs Python 3.9 for its first-time setup. Install Python 3.9, then double-click START HERE again." with title "Python 3.9 required" buttons {"Cancel", "Open Python download"} default button "Open Python download"
button returned of result
APPLESCRIPT
)"
    if [[ "$choice" == "Open Python download" ]]; then
      /usr/bin/open "https://www.python.org/downloads/release/python-3913/"
    fi
    exit 1
  fi

  choice="$(/usr/bin/osascript <<'APPLESCRIPT' 2>/dev/null || true
display dialog "First-time setup will install the analysis packages. It needs an internet connection and may take 10–20 minutes. Later launches will be immediate." with title "Set up MyoFuse Hybrid" buttons {"Cancel", "Install"} default button "Install"
button returned of result
APPLESCRIPT
)"
  if [[ "$choice" != "Install" ]]; then
    exit 0
  fi

  print "Installing MyoFuse Hybrid. Keep this window open..."
  export BOOTSTRAP_PYTHON="$python39"
  if ! "${script_dir}/setup_mac.sh" 2>&1 | tee "${script_dir}/setup.log"; then
    show_message "Setup did not complete. Please send setup.log to the developer."
    read -k 1 "?Press any key to close."
    exit 1
  fi
  python_bin="${script_dir}/.venv/bin/python"
fi

port="$("$python_bin" - <<'PY'
import socket
for port in range(8765, 8791):
    with socket.socket() as connection:
        if connection.connect_ex(("127.0.0.1", port)) != 0:
            print(port)
            break
else:
    raise SystemExit("No free local port found between 8765 and 8790")
PY
)"

print "MyoFuse Hybrid is opening in your browser."
print "Using Python: $python_bin"
print "Local address: http://127.0.0.1:$port/"
print "Keep this window open while using the application."

export MYOFUSE_ANALYSIS_PYTHON="$python_bin"
exec "$python_bin" "${script_dir}/manual_review_web.py" \
  --analysis-python "$python_bin" \
  --port "$port" \
  --open-browser
