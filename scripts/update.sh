#!/usr/bin/env bash
# Pull latest code from GitHub and restart AITrading
# Run on the cloud server: bash /opt/aitrading/scripts/update.sh

set -euo pipefail
APP_DIR="/opt/aitrading"
SERVICE="aitrading"

echo "Pulling latest code..."
git -C "$APP_DIR" pull

echo "Updating dependencies..."
"$APP_DIR/venv/bin/pip" install -r "$APP_DIR/requirements.txt" -q

echo "Restarting service..."
systemctl restart "$SERVICE"
sleep 2
systemctl is-active "$SERVICE" && echo "AITrading restarted OK" || echo "ERROR: check journalctl -u $SERVICE"
