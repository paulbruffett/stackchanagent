#!/usr/bin/env bash
# Install (or re-install) the brain as a lingering systemd user service on this
# host. Idempotent — re-run it after editing the unit template. No root needed.
#
#   ./deploy/install.sh                    # track main
#   STACKCHAN_BRANCH=my-branch ./deploy/install.sh
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BRANCH=${STACKCHAN_BRANCH:-main}
BIN="$HOME/.local/bin"
UNIT_DIR="$HOME/.config/systemd/user"

echo "repo:   $REPO"
echo "branch: $BRANCH"

mkdir -p "$BIN" "$UNIT_DIR"
install -m 755 "$REPO/deploy/update-brain.sh" "$BIN/stackchan-brain-update"
sed -e "s|@REPO@|$REPO|g" -e "s|@BRANCH@|$BRANCH|g" \
    "$REPO/deploy/stackchan-brain.service" > "$UNIT_DIR/stackchan-brain.service"

# Linger = the user manager starts at boot without anyone logging in. Without
# it the service would only start on first SSH login.
loginctl enable-linger "$USER"

systemctl --user daemon-reload
systemctl --user enable stackchan-brain.service

echo
echo "installed. next:"
echo "  systemctl --user restart stackchan-brain    # start + pull $BRANCH"
echo "  journalctl --user -u stackchan-brain -f     # follow"
