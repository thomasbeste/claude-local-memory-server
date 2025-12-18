#!/bin/bash
#
# Claude Local Memory Server - Systemd Installation Script (LXC/VM/Bare Metal)
#
# For Docker deployment, use install-docker.sh instead.
#
# Usage:
#   sudo ./scripts/install-systemd.sh
#
# Options:
#   --port PORT         HTTP port (default: 8420)
#   --api-key KEY       Set API key for authentication
#   --data-dir DIR      Data directory (default: /var/lib/claude-local-memory-server)
#   --no-service        Don't install systemd service
#

set -e

# --- Configuration ---
INSTALL_DIR="/opt/claude-local-memory-server"
DATA_DIR="/var/lib/claude-local-memory-server"
SERVICE_USER="claude-memory"
PORT=8420
API_KEY=""
INSTALL_SERVICE=true

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --no-service)
            INSTALL_SERVICE=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --port PORT         HTTP port (default: 8420)"
            echo "  --api-key KEY       Set API key for authentication"
            echo "  --data-dir DIR      Data directory (default: /var/lib/claude-local-memory-server)"
            echo "  --no-service        Don't install systemd service"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# --- Check root ---
if [[ $EUID -ne 0 ]]; then
    log_error "This script must be run as root (use sudo)"
    exit 1
fi

# --- Detect OS ---
if [[ -f /etc/debian_version ]]; then
    OS="debian"
    PKG_MANAGER="apt"
elif [[ -f /etc/redhat-release ]]; then
    OS="rhel"
    PKG_MANAGER="dnf"
else
    log_warn "Unknown OS, assuming Debian-based"
    OS="debian"
    PKG_MANAGER="apt"
fi

log_info "Detected OS: $OS"

# --- Install system dependencies ---
log_info "Installing system dependencies..."

if [[ "$PKG_MANAGER" == "apt" ]]; then
    apt update
    apt install -y \
        python3 \
        python3-venv \
        python3-pip \
        curl \
        ca-certificates
elif [[ "$PKG_MANAGER" == "dnf" ]]; then
    dnf install -y \
        python3 \
        python3-pip \
        curl \
        ca-certificates
fi

# --- Install uv ---
log_info "Installing uv package manager..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    
    # Add to system profile
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> /etc/profile.d/uv.sh
fi

# Ensure uv is available
export PATH="$HOME/.local/bin:$PATH"

if command -v uv &> /dev/null; then
    log_info "uv installed: $(uv --version)"
else
    log_warn "uv not found in PATH, falling back to pip"
fi

# --- Create service user ---
log_info "Creating service user: $SERVICE_USER"
if ! id "$SERVICE_USER" &>/dev/null; then
    useradd -r -s /bin/false -d "$INSTALL_DIR" "$SERVICE_USER"
fi

# --- Create directories ---
log_info "Creating directories..."
mkdir -p "$INSTALL_DIR"
mkdir -p "$DATA_DIR"

# --- Setup application ---
log_info "Setting up application in $INSTALL_DIR..."
cd "$INSTALL_DIR"

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Copy application files if running from source directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/pyproject.toml" ]]; then
    log_info "Installing from local source..."
    cp -r "$SCRIPT_DIR/src" "$INSTALL_DIR/"
    cp "$SCRIPT_DIR/pyproject.toml" "$INSTALL_DIR/"
    cp "$SCRIPT_DIR/README.md" "$INSTALL_DIR/" 2>/dev/null || true
    
    # Install with server dependencies
    pip install -e ".[server]"
else
    log_info "Installing from PyPI..."
    pip install "claude-local-memory-server[server]"
fi

# --- Set permissions ---
log_info "Setting permissions..."
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
chown -R "$SERVICE_USER:$SERVICE_USER" "$DATA_DIR"

# --- Generate API key if not provided ---
if [[ -z "$API_KEY" ]]; then
    API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    log_info "Generated API key: $API_KEY"
    log_warn "Save this key! You'll need it to configure clients."
fi

# --- Create environment file ---
log_info "Creating environment file..."
cat > "$INSTALL_DIR/.env" << EOF
# Claude Local Memory Server Configuration
MEMORY_API_KEY=$API_KEY
DATA_DIR=$DATA_DIR
PORT=$PORT
EOF

chmod 600 "$INSTALL_DIR/.env"
chown "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR/.env"

# --- Install systemd service ---
if [[ "$INSTALL_SERVICE" == true ]]; then
    log_info "Installing systemd service..."

    cat > /etc/systemd/system/claude-local-memory-server.service << EOF
[Unit]
Description=Claude Local Memory Server
Documentation=https://github.com/thomasbeste/claude-local-memory-server
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR
EnvironmentFile=$INSTALL_DIR/.env
ExecStart=$INSTALL_DIR/.venv/bin/python -m claude_memory.cli server \\
    --host 0.0.0.0 \\
    --port $PORT \\
    --data-dir $DATA_DIR
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$DATA_DIR
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

    # Reload and enable service
    systemctl daemon-reload
    systemctl enable claude-local-memory-server
    systemctl start claude-local-memory-server

    log_info "Service installed and started"

    # Wait a moment and check status
    sleep 2
    if systemctl is-active --quiet claude-local-memory-server; then
        log_info "Service is running"
    else
        log_error "Service failed to start. Check: journalctl -u claude-local-memory-server"
    fi
fi

# --- Print summary ---
echo ""
echo "========================================"
echo "  Claude Local Memory Server Installed!"
echo "========================================"
echo ""
echo "  Install directory: $INSTALL_DIR"
echo "  Data directory:    $DATA_DIR"
echo "  Port:              $PORT"
echo "  API Key:           $API_KEY"
echo ""
echo "  Service commands:"
echo "    systemctl status claude-local-memory-server"
echo "    systemctl restart claude-local-memory-server"
echo "    journalctl -u claude-local-memory-server -f"
echo ""
echo "  Test the server:"
echo "    curl http://localhost:$PORT/health"
echo "    curl http://localhost:$PORT/stats -H 'X-API-Key: $API_KEY'"
echo ""
echo "  Client configuration:"
echo "    MEMORY_SERVER=http://$(hostname -I | awk '{print $1}'):$PORT"
echo "    MEMORY_API_KEY=$API_KEY"
echo ""
echo "========================================"

# --- Save config for easy reference ---
cat > "$INSTALL_DIR/client-config.txt" << EOF
# Claude Local Memory Server - Client Configuration
# Copy this to your development machines

MEMORY_SERVER=http://$(hostname -I | awk '{print $1}'):$PORT
MEMORY_API_KEY=$API_KEY

# Claude Code MCP config (replace CLIENT_NAME with e.g. 'laptop', 'desktop'):
claude mcp add-json "memory" '{
  "command": "python",
  "args": ["-m", "claude_memory.cli", "client"],
  "env": {
    "MEMORY_SERVER": "http://$(hostname -I | awk '{print $1}'):$PORT",
    "MEMORY_API_KEY": "$API_KEY",
    "MEMORY_CLIENT_ID": "CLIENT_NAME"
  }
}'
EOF

log_info "Client config saved to: $INSTALL_DIR/client-config.txt"
