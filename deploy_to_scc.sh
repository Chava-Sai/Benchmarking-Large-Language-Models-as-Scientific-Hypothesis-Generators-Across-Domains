#!/bin/bash
# Deploy local code to BU SCC cluster.
# Usage: bash deploy_to_scc.sh

set -e

SCC_USER="saichava"
SCC_HOST="scc1.bu.edu"
REMOTE_DIR="$SCC_USER@$SCC_HOST:/projectnb/cs505am/students/saichava/icml2026"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Deploying ICML project to BU SCC ==="
echo "Local:  $LOCAL_DIR"
echo "Remote: ~/icml2026 on $SCC_HOST"
echo ""

# Create remote directory
ssh "$SCC_USER@$SCC_HOST" "mkdir -p ~/icml2026/logs"

# Sync code (skip large data/results/checkpoints)
rsync -avz --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.egg-info' \
    --exclude='.env' \
    --exclude='.env.*' \
    --exclude='.env.save' \
    --exclude='venv/' \
    --exclude='data/raw/' \
    --exclude='data/processed/' \
    --exclude='data/finetune/' \
    --exclude='data/embeddings/' \
    --exclude='checkpoints/' \
    --exclude='results/' \
    --exclude='logs/' \
    "$LOCAL_DIR/" \
    "$REMOTE_DIR/"

echo ""
echo "=== Code synced. Now on SCC run: ==="
echo ""
echo "  ssh saichava@scc1.bu.edu"
echo "  cd ~/icml2026"
echo "  bash scripts/00_setup_env.sh"
