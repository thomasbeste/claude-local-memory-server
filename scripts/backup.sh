#!/bin/bash
#
# Claude Local Memory Server - Backup Script
#
# Usage:
#   ./backup.sh                     # Backup to /var/backups/claude-local-memory-server/
#   ./backup.sh /path/to/backup     # Backup to custom location
#

set -e

DATA_DIR="/var/lib/claude-local-memory-server"
BACKUP_DIR="${1:-/var/backups/claude-local-memory-server}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/memories_$TIMESTAMP.parquet"

mkdir -p "$BACKUP_DIR"

if [[ ! -f "$DATA_DIR/memories.parquet" ]]; then
    echo "No data to backup at $DATA_DIR/memories.parquet"
    exit 1
fi

cp "$DATA_DIR/memories.parquet" "$BACKUP_FILE"

# Also backup with zstd compression if available
if command -v zstd &> /dev/null; then
    zstd -q "$BACKUP_FILE" -o "${BACKUP_FILE}.zst"
    rm "$BACKUP_FILE"
    BACKUP_FILE="${BACKUP_FILE}.zst"
fi

echo "Backup created: $BACKUP_FILE"
echo "Size: $(du -h "$BACKUP_FILE" | cut -f1)"

# Cleanup old backups (keep last 30)
ls -t "$BACKUP_DIR"/memories_*.parquet* 2>/dev/null | tail -n +31 | xargs -r rm --

echo "Done"
