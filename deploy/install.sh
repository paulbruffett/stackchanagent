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

# A typo'd branch is indistinguishable from an outage once it is baked into the
# unit: `git fetch origin nosuchbranch` fails, update-brain.sh reports it as
# "network not up yet?", and the robot silently runs whatever commit is on disk
# forever. Catch it here instead. --exit-code returns 2 for "no such ref" and
# 128 for "could not talk to origin"; only the first is the operator's mistake,
# so an install with the network down still goes through.
rc=0
git -C "$REPO" ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1 || rc=$?
case "$rc" in
    0) ;;
    2) echo "error: branch '$BRANCH' does not exist on origin" >&2; exit 1 ;;
    *) echo "warning: could not verify '$BRANCH' on origin (offline, or no origin remote)" >&2 ;;
esac

# Both of these are otherwise only discovered at service start, deep inside a
# start job whose log goes to a volatile journal. Say it now, while someone is
# watching. update-brain.sh hard-codes this uv path too — STACKCHAN_UV is not
# set in the unit, so this is the binary the service will actually look for.
if [ ! -x "$HOME/.local/bin/uv" ]; then
    echo "warning: no uv at $HOME/.local/bin/uv — the venv cannot be built or repaired" >&2
fi
if [ ! -f "$REPO/brain/.venv/.stackchan-sync-ok" ]; then
    echo "warning: $REPO/brain/.venv is missing or unverified — the first start runs uv sync (slow)" >&2
fi

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
echo "  systemctl --user restart stackchan-brain      # start + pull $BRANCH"
echo "  journalctl --user-unit=stackchan-brain -f     # follow"
