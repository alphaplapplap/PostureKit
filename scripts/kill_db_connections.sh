#!/bin/bash
#
# Emergency PostgreSQL Connection Killer for PostureKit
#
# Forcefully terminates all idle connections to PostureKit databases.
# Use this when Python cleanup (prepare_eject.py) doesn't work.
#
# Usage:
#     bash scripts/kill_db_connections.sh
#
#     # Or make executable and run directly:
#     chmod +x scripts/kill_db_connections.sh
#     ./scripts/kill_db_connections.sh
#
# What it does:
# - Connects to PostgreSQL
# - Finds all connections to posturekit_* databases
# - Terminates idle connections (safe to kill)
# - Preserves this script's own connection
# - Releases file handles to external drives
#
# When to use:
# - prepare_eject.py failed or timed out
# - "diskutil eject" still fails after Python cleanup
# - Emergency situations requiring immediate drive removal
# - Database connection deadlocks
#
# Safety:
# - Only kills IDLE connections (not active queries)
# - Skips the connection running this script
# - Does not kill other PostgreSQL databases
# - Safe to run multiple times
#
# WARNING: This will disconnect any active PostureKit sessions!
#

set -e  # Exit on error

echo "========================================================================"
echo "POSTUREKIT - EMERGENCY DATABASE CONNECTION TERMINATION"
echo "========================================================================"
echo

echo "Checking PostgreSQL connection..."
if ! psql -U postgres -c '\q' 2>/dev/null; then
    echo "✗ ERROR: Cannot connect to PostgreSQL"
    echo "  Make sure PostgreSQL is running:"
    echo "    brew services start postgresql@16"
    exit 1
fi

echo "✓ PostgreSQL is running"
echo

echo "Finding PostureKit database connections..."

# Count connections before
CONN_COUNT=$(psql -U postgres -t -c "
    SELECT count(*)
    FROM pg_stat_activity
    WHERE datname LIKE 'posturekit%'
      AND pid != pg_backend_pid()
      AND state IN ('idle', 'idle in transaction');
" 2>/dev/null | tr -d ' ')

if [ "$CONN_COUNT" -eq 0 ]; then
    echo "✓ No idle connections found"
    echo "✓ Safe to eject external drives"
    echo
    echo "If eject still fails, try:"
    echo "  diskutil unmount force /Volumes/Projects"
    exit 0
fi

echo "Found $CONN_COUNT idle connection(s) to terminate"
echo

echo "Terminating idle connections..."
psql -U postgres <<EOF
SELECT
    pg_terminate_backend(pid) as terminated,
    datname,
    state,
    application_name
FROM pg_stat_activity
WHERE datname LIKE 'posturekit%'
  AND pid != pg_backend_pid()
  AND state IN ('idle', 'idle in transaction');
EOF

echo
echo "✓ Idle connections terminated"

# Verify cleanup
REMAINING=$(psql -U postgres -t -c "
    SELECT count(*)
    FROM pg_stat_activity
    WHERE datname LIKE 'posturekit%'
      AND pid != pg_backend_pid();
" 2>/dev/null | tr -d ' ')

if [ "$REMAINING" -gt 0 ]; then
    echo "⚠  Warning: $REMAINING connection(s) still active"
    echo "   (These may be running queries - cannot safely terminate)"
    echo
    echo "   Wait for queries to finish or use nuclear option:"
    echo "     sudo killall postgres"
    echo "     brew services restart postgresql@16"
else
    echo "✓ All PostureKit connections closed"
fi

echo
echo "========================================================================"
echo "CLEANUP COMPLETE"
echo "========================================================================"
echo
echo "External drives should now be safe to eject:"
echo "  • Eject in Finder"
echo "  • diskutil eject disk6"
echo "  • diskutil unmount force /Volumes/Projects  (if normal eject fails)"
echo
