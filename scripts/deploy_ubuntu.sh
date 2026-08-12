#!/usr/bin/env bash
# AITrading — one-shot Ubuntu server setup
# Tested on Ubuntu 22.04 / 24.04
# Run as root or with sudo: bash deploy_ubuntu.sh

set -euo pipefail

REPO_URL="https://github.com/flyguy91355/AITrading.git"
APP_DIR="/opt/aitrading"
APP_USER="aitrading"
SERVICE_NAME="aitrading"
PORT="8080"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[✓]${NC} $*"; }
warn()    { echo -e "${YELLOW}[!]${NC} $*"; }
section() { echo -e "\n${GREEN}━━━ $* ━━━${NC}"; }

if [[ $EUID -ne 0 ]]; then
  echo -e "${RED}Run as root: sudo bash deploy_ubuntu.sh${NC}"
  exit 1
fi

section "System update"
apt-get update -q
apt-get upgrade -y -q
apt-get install -y -q \
  git curl wget ufw fail2ban \
  software-properties-common \
  build-essential libssl-dev libffi-dev \
  sqlite3 pango1.0-tools libpango-1.0-0 libpangoft2-1.0-0  # weasyprint PDF deps
info "System packages installed"

section "Python 3.12"
if ! python3.12 --version &>/dev/null; then
  add-apt-repository -y ppa:deadsnakes/ppa
  apt-get update -q
  apt-get install -y -q python3.12 python3.12-venv python3.12-dev
fi
python3.12 --version
info "Python 3.12 ready"

section "App user"
if ! id "$APP_USER" &>/dev/null; then
  useradd -r -m -d "$APP_DIR" -s /bin/bash "$APP_USER"
  info "Created user: $APP_USER"
else
  info "User $APP_USER already exists"
fi

section "Clone / update repository"
if [[ -d "$APP_DIR/.git" ]]; then
  warn "Repo already cloned — pulling latest"
  sudo -u "$APP_USER" git -C "$APP_DIR" pull
else
  sudo -u "$APP_USER" git clone "$REPO_URL" "$APP_DIR"
  info "Repo cloned to $APP_DIR"
fi

section "Python virtual environment"
if [[ ! -d "$APP_DIR/venv" ]]; then
  sudo -u "$APP_USER" python3.12 -m venv "$APP_DIR/venv"
fi
sudo -u "$APP_USER" "$APP_DIR/venv/bin/pip" install --upgrade pip -q
sudo -u "$APP_USER" "$APP_DIR/venv/bin/pip" install -r "$APP_DIR/requirements.txt" -q
info "Dependencies installed"

section "Data directory"
mkdir -p "$APP_DIR/data"
chown "$APP_USER:$APP_USER" "$APP_DIR/data"
info "data/ directory ready"

section "API credentials"
ENV_FILE="$APP_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
  warn ".env already exists — skipping (edit $ENV_FILE manually if needed)"
else
  cp "$APP_DIR/config/credentials.env" "$ENV_FILE"
  chown "$APP_USER:$APP_USER" "$ENV_FILE"
  chmod 600 "$ENV_FILE"

  echo ""
  echo "Enter your API keys (press Enter to skip and fill in later):"
  echo ""

  read -rp "  ANTHROPIC_API_KEY:    " ANT_KEY
  read -rp "  ALPACA_API_KEY:       " ALP_KEY
  read -rp "  ALPACA_SECRET_KEY:    " ALP_SEC
  read -rp "  FINNHUB_API_KEY:      " FIN_KEY
  read -rp "  NEWSAPI_API_KEY:      " NEWS_KEY
  read -rp "  PORT [$PORT]:         " CUSTOM_PORT
  CUSTOM_PORT="${CUSTOM_PORT:-$PORT}"

  [[ -n "$ANT_KEY" ]]  && sed -i "s|ANTHROPIC_API_KEY=.*|ANTHROPIC_API_KEY=$ANT_KEY|" "$ENV_FILE"
  [[ -n "$ALP_KEY" ]]  && sed -i "s|ALPACA_API_KEY=.*|ALPACA_API_KEY=$ALP_KEY|" "$ENV_FILE"
  [[ -n "$ALP_SEC" ]]  && sed -i "s|ALPACA_SECRET_KEY=.*|ALPACA_SECRET_KEY=$ALP_SEC|" "$ENV_FILE"
  [[ -n "$FIN_KEY" ]]  && sed -i "s|FINNHUB_API_KEY=.*|FINNHUB_API_KEY=$FIN_KEY|" "$ENV_FILE"
  [[ -n "$NEWS_KEY" ]] && sed -i "s|NEWSAPI_API_KEY=.*|NEWSAPI_API_KEY=$NEWS_KEY|" "$ENV_FILE"
  echo "PORT=$CUSTOM_PORT" >> "$ENV_FILE"
  PORT="$CUSTOM_PORT"

  info ".env created at $ENV_FILE"
fi

section "Systemd service"
cat > "/etc/systemd/system/${SERVICE_NAME}.service" <<EOF
[Unit]
Description=AITrading — AI Stock Research & Trading
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${APP_USER}
WorkingDirectory=${APP_DIR}
Environment=PATH=${APP_DIR}/venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=${APP_DIR}/venv/bin/python start.py --mode web
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=${SERVICE_NAME}

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
info "Systemd service installed and enabled"

section "Firewall (ufw)"
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh           # Always keep SSH open
ufw allow "$PORT/tcp"   # Dashboard — restrict to your IP after setup if desired
ufw --force enable
info "Firewall enabled — SSH + port $PORT open"

section "Fail2ban (SSH brute-force protection)"
systemctl enable fail2ban
systemctl start fail2ban
info "Fail2ban active"

section "SQLite backup cron"
BACKUP_SCRIPT="/opt/aitrading-backup.sh"
cat > "$BACKUP_SCRIPT" <<'BKUP'
#!/bin/bash
SRC="/opt/aitrading/data/aitrading.db"
DST="/opt/aitrading/data/backups/aitrading-$(date +%Y%m%d).db"
mkdir -p /opt/aitrading/data/backups
sqlite3 "$SRC" ".backup '$DST'"
# Keep last 14 days
find /opt/aitrading/data/backups -name "*.db" -mtime +14 -delete
BKUP
chmod +x "$BACKUP_SCRIPT"
chown "$APP_USER:$APP_USER" "$BACKUP_SCRIPT"
# Run at 8 PM ET (midnight UTC) on weekdays
(crontab -l 2>/dev/null; echo "0 0 * * 1-5 $BACKUP_SCRIPT") | crontab -
info "Daily SQLite backup configured (data/backups/)"

section "Log rotation"
cat > "/etc/logrotate.d/$SERVICE_NAME" <<EOF
/var/log/journal/*${SERVICE_NAME}* {
    weekly
    rotate 4
    compress
    missingok
    notifempty
}
EOF
info "Log rotation configured"

section "Start AITrading"
systemctl start "$SERVICE_NAME"
sleep 3
if systemctl is-active --quiet "$SERVICE_NAME"; then
  info "AITrading is RUNNING"
else
  warn "Service did not start — check logs: journalctl -u $SERVICE_NAME -f"
fi

SERVER_IP=$(curl -s ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  AITrading deployed successfully!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Dashboard:    http://${SERVER_IP}:${PORT}"
echo "  App dir:      $APP_DIR"
echo "  Credentials:  $ENV_FILE"
echo "  Logs:         journalctl -u $SERVICE_NAME -f"
echo "  Stop:         systemctl stop $SERVICE_NAME"
echo "  Restart:      systemctl restart $SERVICE_NAME"
echo ""
echo "  SECURE REMOTE ACCESS (recommended):"
echo "  SSH tunnel:   ssh -L ${PORT}:localhost:${PORT} ${APP_USER}@${SERVER_IP}"
echo "  Then open:    http://localhost:${PORT}"
echo ""
echo "  RESTRICT DASHBOARD TO YOUR IP (optional, recommended for live trading):"
echo "  ufw delete allow ${PORT}/tcp"
echo "  ufw allow from YOUR.HOME.IP to any port ${PORT}"
echo ""
if grep -q "your-key-here\|change-me\|placeholder" "$ENV_FILE" 2>/dev/null; then
  warn "Some API keys are not set — edit $ENV_FILE then: systemctl restart $SERVICE_NAME"
fi
