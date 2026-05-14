#!/bin/bash
# =============================================================
# AI Gen API v2 — Video reaper
#
# Deletes video files older than RETENTION_DAYS from both the
# serverless output staging dir and the pod-mode ComfyUI output dir.
# Safe to run repeatedly: it's idempotent and only touches video
# files (mp4/webm/gif/mov) — images are left alone.
#
# Invoked by the supervisor loop in /workspace/start_cleanup.sh, which
# runs cleanup.sh once a day. Can also be run manually for spot cleanup.
# =============================================================
set -u

RETENTION_DAYS="${RETENTION_DAYS:-3}"
LOG="/workspace/cleanup.log"

# Directories to scan. Both live on the network volume — the pod sees
# /workspace/, serverless workers see the same files at /runpod-volume/.
SERVERLESS_OUTPUTS="/workspace/outputs"
POD_VIDEO_DIR="/workspace/runpod-slim/ComfyUI/output/video"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"; }

# Cap log at last 200 lines so it doesn't grow unbounded
tail -200 "$LOG" > "${LOG}.tmp" 2>/dev/null && mv "${LOG}.tmp" "$LOG"

log "reaper start (RETENTION_DAYS=$RETENTION_DAYS)"

reap_dir() {
  local dir="$1"
  local label="$2"
  [ -d "$dir" ] || { log "  $label: skip (dir doesn't exist yet)"; return; }

  # Delete video files older than RETENTION_DAYS.
  # -mtime +N matches files modified > N*24h ago.
  local deleted
  deleted=$(find "$dir" -type f \
    \( -iname '*.mp4' -o -iname '*.webm' -o -iname '*.mov' -o -iname '*.gif' \) \
    -mtime "+$RETENTION_DAYS" -print -delete 2>/dev/null | wc -l)

  # Reap now-empty job_id subdirs (serverless layout: /workspace/outputs/<job_id>/)
  local empty_dirs
  empty_dirs=$(find "$dir" -mindepth 1 -type d -empty -print -delete 2>/dev/null | wc -l)

  log "  $label: deleted $deleted file(s), $empty_dirs empty subdir(s)"
}

reap_dir "$SERVERLESS_OUTPUTS" "serverless outputs"
reap_dir "$POD_VIDEO_DIR"      "pod-mode videos"

log "reaper done"
