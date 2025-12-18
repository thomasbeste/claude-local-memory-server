#!/bin/bash
#
# Claude Local Memory Server - Uninstall Script
#

set -e

INSTALL_DIR="/opt/claude-local-memory-server"
DATA_DIR="/var/lib/claude-local-memory-server"
SERVICE_USER="claude-memory"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root"
    exit 1
fi

echo "This will remove Claude Local Memory Server."
echo "Data in $DATA_DIR will be preserved unless you use --purge"
echo ""
read -p "Continue? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Stop and disable service
if systemctl is-active --quiet claude-local-memory-server 2>/dev/null; then
    log_info "Stopping service..."
    systemctl stop claude-local-memory-server
fi

if systemctl is-enabled --quiet claude-local-memory-server 2>/dev/null; then
    log_info "Disabling service..."
    systemctl disable claude-local-memory-server
fi

# Remove service file
if [[ -f /etc/systemd/system/claude-local-memory-server.service ]]; then
    log_info "Removing systemd service..."
    rm /etc/systemd/system/claude-local-memory-server.service
    systemctl daemon-reload
fi

# Remove install directory
if [[ -d "$INSTALL_DIR" ]]; then
    log_info "Removing $INSTALL_DIR..."
    rm -rf "$INSTALL_DIR"
fi

# Remove user
if id "$SERVICE_USER" &>/dev/null; then
    log_info "Removing user $SERVICE_USER..."
    userdel "$SERVICE_USER" 2>/dev/null || true
fi

# Optionally remove data
if [[ "$1" == "--purge" ]]; then
    if [[ -d "$DATA_DIR" ]]; then
        log_warn "Removing data directory $DATA_DIR..."
        rm -rf "$DATA_DIR"
    fi
else
    log_info "Data preserved in $DATA_DIR"
    log_info "Run with --purge to remove data as well"
fi

log_info "Uninstall complete"
