#!/bin/bash
# scripts/watch_and_submit.sh — Submit queued jobs one by one as slots open.
#
# Usage:
#   bash scripts/watch_and_submit.sh [INTERVAL_SECONDS]
#
# Edit the JOBS array below with the commands to run in order.
# The script checks squeue every INTERVAL seconds and submits the next
# job as soon as fewer than MAX_JOBS are running/queued.

INTERVAL="${1:-120}"   # check every 2 min by default
MAX_JOBS=2             # max jobs allowed in queue at once

# ── Job list — edit this ──────────────────────────────────────────────────────

JOBS=()

# ── Main loop ─────────────────────────────────────────────────────────────────

total=${#JOBS[@]}
next=0

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  watch_and_submit — ${total} jobs queued, checking every ${INTERVAL}s"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

while [[ $next -lt $total ]]; do
    # Count current jobs (subtract 1 for the header line)
    current=$(( $(squeue -u "$USER" | wc -l) - 1 ))

    if [[ $current -lt $MAX_JOBS ]]; then
        slots=$(( MAX_JOBS - current ))
        echo "[$(date '+%H:%M:%S')] ${current}/${MAX_JOBS} jobs active — submitting up to ${slots} job(s)"

        for (( s=0; s<slots && next<total; s++ )); do
            echo "  → [${next+1}/${total}] ${JOBS[$next]}"
            eval "${JOBS[$next]}"
            next=$(( next + 1 ))
        done
    else
        echo "[$(date '+%H:%M:%S')] ${current}/${MAX_JOBS} jobs active — waiting ${INTERVAL}s …"
    fi

    [[ $next -lt $total ]] && sleep "$INTERVAL"
done

echo ""
echo "✓ All ${total} jobs submitted."
