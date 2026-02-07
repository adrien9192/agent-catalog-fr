#!/bin/bash
# Script quotidien pour ajouter 3-5 nouveaux cas d'usage d'Agents IA
# Usage: ./scripts/daily-add-use-cases.sh
# Cron: 0 6 * * * cd /path/to/project && ./scripts/daily-add-use-cases.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$SCRIPT_DIR/logs"
TODAY=$(date +%Y-%m-%d)
LOG_FILE="$LOG_DIR/daily-$TODAY.log"

# Create logs directory
mkdir -p "$LOG_DIR"

echo "[$TODAY] Starting daily use case generation..." | tee -a "$LOG_FILE"

cd "$PROJECT_DIR"

# Run Claude Code with the prompt
npx claude -p "$(cat "$SCRIPT_DIR/add-use-cases-prompt.md")" \
  --allowedTools "Edit,Read,Write,Bash,Glob,Grep,WebSearch,WebFetch" \
  2>&1 | tee -a "$LOG_FILE"

echo "[$TODAY] Daily use case generation complete." | tee -a "$LOG_FILE"
