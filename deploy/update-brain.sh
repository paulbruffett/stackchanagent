#!/usr/bin/env bash
# Bring the brain checkout up to the tip of its branch, then re-sync the venv
# if (and only if) the dependency manifests moved. Runs as the ExecStartPre of
# stackchan-brain.service, so every host boot and every `systemctl --user
# restart stackchan-brain` lands on fresh code.
#
# Never fatal. If the network is down, origin is unreachable, or `uv sync`
# blows up, we log it and exit 0 so the robot still comes up on whatever code
# is already on disk. A dead brain is worse than a stale one.
#
# This runs from the *installed* copy at ~/.local/bin/stackchan-brain-update,
# not from the repo — a `git reset --hard` mid-run would otherwise rewrite the
# file bash is still reading. The copy re-installs itself (atomically) when the
# repo version changes; that lands on the next start.
set -uo pipefail

REPO=${STACKCHAN_REPO:-$HOME/code/stackchanagent}
BRANCH=${STACKCHAN_BRANCH:-main}
UV=${STACKCHAN_UV:-$HOME/.local/bin/uv}
SELF=${STACKCHAN_SELF:-$HOME/.local/bin/stackchan-brain-update}
UNIT=${STACKCHAN_UNIT:-$HOME/.config/systemd/user/stackchan-brain.service}

# Restart=always re-runs this script on every crash-restart. Without a cooldown
# a crash loop turns into a fetch loop against GitHub. XDG_RUNTIME_DIR is wiped
# on reboot, so a boot always fetches.
STAMP=${XDG_RUNTIME_DIR:-/tmp}/stackchan-brain-fetch.stamp
FETCH_COOLDOWN=60

log() { echo "update-brain: $*"; }

cd "$REPO" 2>/dev/null || { log "no checkout at $REPO — skipping update"; exit 0; }

if [ -f "$STAMP" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$STAMP") ))
    if [ "$age" -lt "$FETCH_COOLDOWN" ]; then
        log "fetched ${age}s ago — skipping update this restart"
        exit 0
    fi
fi

before=$(git rev-parse HEAD)

# The user manager has no network-online.target to order against, and
# NetworkManager-wait-online is disabled on this host, so the first attempt at
# boot often races DHCP. Retry for ~30 s before giving up.
fetched=0
for attempt in 1 2 3 4 5 6; do
    if timeout 60 git fetch --prune origin "$BRANCH"; then
        fetched=1
        break
    fi
    log "fetch attempt $attempt failed (network not up yet?) — retrying in 5s"
    sleep 5
done
if [ "$fetched" -ne 1 ]; then
    log "could not reach origin — starting on the code already on disk ($before)"
    exit 0
fi
touch "$STAMP"

# Hard reset: this checkout is a deploy target, not a workspace. Local edits to
# *tracked* files are discarded on purpose. Untracked and ignored files survive
# (.env, brain/.venv/, brain/wheels/*.whl), and runtime state lives outside the
# repo in ~/.stackchan/memory.db.
if ! git checkout -f "$BRANCH" 2>/dev/null; then
    git checkout -f -b "$BRANCH" --track "origin/$BRANCH" || {
        log "could not check out $BRANCH — starting on $before"
        exit 0
    }
fi
git reset --hard "origin/$BRANCH" || { log "reset failed — starting on $before"; exit 0; }

after=$(git rev-parse HEAD)
if [ "$before" = "$after" ]; then
    log "already at $BRANCH $after"
else
    log "updated $before -> $after ($BRANCH)"
fi

# Re-sync deps only when the manifests actually moved. A `uv sync` on every
# boot costs a network round trip and churns the venv for nothing.
need_sync=0
[ -x "$REPO/brain/.venv/bin/python" ] || need_sync=1
if [ "$before" != "$after" ] &&
   ! git diff --quiet "$before" "$after" -- brain/pyproject.toml brain/uv.lock; then
    need_sync=1
fi
if [ "$need_sync" -eq 1 ]; then
    if [ -x "$UV" ]; then
        log "dependency manifests changed — uv sync"
        # --frozen: install exactly what uv.lock says, never re-resolve. Keeps
        # the deploy checkout clean and makes the installed set reproducible.
        ( cd "$REPO/brain" && timeout 900 "$UV" sync --frozen ) ||
            log "uv sync FAILED — starting with the venv as-is"
    else
        log "uv not found at $UV — skipping dependency sync"
    fi
fi

# Propagate our own changes. Atomic rename, so the bash process currently
# reading this file keeps its old inode; the new copy runs next start.
if [ -f "$REPO/deploy/update-brain.sh" ] && ! cmp -s "$REPO/deploy/update-brain.sh" "$SELF"; then
    if install -m 755 "$REPO/deploy/update-brain.sh" "$SELF.new" && mv -f "$SELF.new" "$SELF"; then
        log "update script refreshed from repo — active next start"
    fi
fi

# The unit file needs a `systemctl --user daemon-reload` to take effect, which
# is not safe to issue from inside this unit's own start job. Just say so.
if [ -f "$REPO/deploy/stackchan-brain.service" ] && [ -f "$UNIT" ]; then
    if ! sed -e "s|@REPO@|$REPO|g" -e "s|@BRANCH@|$BRANCH|g" \
        "$REPO/deploy/stackchan-brain.service" | cmp -s - "$UNIT"; then
        log "NOTE: deploy/stackchan-brain.service changed in the repo — re-run deploy/install.sh to apply"
    fi
fi

exit 0
