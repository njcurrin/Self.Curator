#!/bin/bash
set -e

# Setup PATH for Python venv
export PATH="/opt/venv/bin:${PATH}"

# Ensure workspace directories exist (may be on bind-mounted volumes)
mkdir -p /workspace/curator/data \
         /workspace/curator/configs \
         /workspace/curator/logs \
         /workspace/curator/jobs \
         /workspace/curator/cache/hf-hub

# Start supervisord in foreground (supervisord runs as PID 1)
exec /usr/bin/supervisord -c /etc/supervisor/supervisord.conf
