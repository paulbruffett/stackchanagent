#!/usr/bin/env bash
# Bring the brain checkout up to the tip of its branch, then re-sync the venv
# if (and only if) it is not already the venv those manifests describe. Runs as
# the ExecStartPre of stackchan-brain.service, so every host boot and every
# `systemctl --user restart stackchan-brain` lands on fresh code.
#
# Never fatal. If the network is down, origin is unreachable, or `uv sync`
# blows up, we log it and exit 0 so the robot still comes up on whatever code
# is already on disk. A dead brain is worse than a stale one.
#
# Never permanently give up either: every failure here has to be retried on the
# next start, and a failure the operator cannot see is the same as no failure.
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

# Written only when `uv sync` ran to completion, and holds the manifest hash it
# was built from. Lives inside the venv so removing the venv removes the claim.
SYNC_STAMP="$REPO/brain/.venv/.stackchan-sync-ok"

# journald is volatile on this host (no /var/log/journal), so a "reset failed"
# line is gone at the next reboot and three weeks of dead auto-update reads
# exactly like one bad night. This breadcrumb survives reboots.
FAIL_MARKER=${STACKCHAN_FAIL_MARKER:-$HOME/.stackchan/update-failed}

log() { echo "update-brain: $*"; }

# Keeps the *first* failure date across starts, so the log line says how long
# the checkout has been frozen rather than implying a one-off blip.
mark_update_failed() {
    local now first
    now=$(date -Is)
    first=$(cut -d' ' -f1 "$FAIL_MARKER" 2>/dev/null)
    [ -n "$first" ] || first=$now
    mkdir -p "$(dirname "$FAIL_MARKER")" 2>/dev/null &&
        printf '%s %s %s\n' "$first" "$now" "$*" >"$FAIL_MARKER" 2>/dev/null
    log "$* — auto-update has been failing since $first (see $FAIL_MARKER)"
}

cd "$REPO" 2>/dev/null || { log "no checkout at $REPO — skipping update"; exit 0; }

# A function, not a run of `exit 0`s: a skipped or failed update must still
# fall through to the dependency block below, which is the only code that can
# rebuild a venv a power cut left half-synced. That repair has to be reachable
# on a boot where origin is unreachable too — `uv sync --frozen` can succeed
# from ~/.cache/uv and brain/wheels/ with no network at all.
update_checkout() {
    if [ -f "$STAMP" ]; then
        local age=$(( $(date +%s) - $(stat -c %Y "$STAMP") ))
        if [ "$age" -lt "$FETCH_COOLDOWN" ]; then
            log "fetched ${age}s ago — skipping update this restart"
            return
        fi
    fi

    local before
    before=$(git rev-parse HEAD)

    # The user manager has no network-online.target to order against, and
    # NetworkManager-wait-online is disabled on this host, so the first attempt
    # at boot often races DHCP. Retry for ~30 s — but bound the whole ladder by
    # wall clock, not by attempt count: with a blackholed origin every attempt
    # burns its full timeout, and TimeoutStartSec is far too generous to cut it
    # short. Every second here is a second the robot cannot answer.
    #
    # Arm the cooldown *before* the ladder, not after a success: an unreachable
    # origin is precisely the case that must not re-run the ladder on the next
    # crash-restart. GIT_TERMINAL_PROMPT=0 so a credential prompt can never
    # block the start job forever.
    touch "$STAMP"
    local deadline=$(( $(date +%s) + 30 ))
    local fetched=0 attempt=0
    while :; do
        attempt=$(( attempt + 1 ))
        if GIT_TERMINAL_PROMPT=0 timeout 30 git fetch --prune origin "$BRANCH"; then
            fetched=1
            break
        fi
        [ "$(date +%s)" -ge "$deadline" ] && break
        log "fetch attempt $attempt failed (network not up yet?) — retrying in 5s"
        sleep 5
    done
    if [ "$fetched" -ne 1 ]; then
        mark_update_failed "could not reach origin after $attempt attempts — starting on the code already on disk ($before)"
        return
    fi

    # A power cut mid-`git reset` leaves .git/index.lock behind, and from then
    # on every checkout/reset fails identically, on every boot, forever. This
    # script is the only thing that writes to this checkout, so a lock nobody
    # has touched in 5 minutes is stale by construction.
    find "$REPO/.git" -maxdepth 1 -name index.lock -mmin +5 -delete 2>/dev/null

    # Hard reset: this checkout is a deploy target, not a workspace. Local edits
    # to *tracked* files are discarded on purpose. Untracked and ignored files
    # survive (.env, brain/.venv/, brain/wheels/*.whl), and runtime state lives
    # outside the repo in ~/.stackchan/memory.db.
    if ! git checkout -f "$BRANCH" 2>/dev/null; then
        git checkout -f -b "$BRANCH" --track "origin/$BRANCH" || {
            mark_update_failed "could not check out $BRANCH — starting on $before"
            return
        }
    fi
    git reset --hard "origin/$BRANCH" || {
        mark_update_failed "reset failed — starting on $before"
        return
    }
    rm -f "$FAIL_MARKER"

    local after
    after=$(git rev-parse HEAD)
    if [ "$before" = "$after" ]; then
        log "already at $BRANCH $after"
    else
        log "updated $before -> $after ($BRANCH)"
    fi
}
update_checkout

# Re-sync deps when the venv is not the one these manifests describe. Keyed on
# a stamp written after a *completed* sync rather than on the interpreter
# existing: uv creates .venv/bin/python before it installs anything into it, so
# a sync killed partway (the 900 s timeout, a power cut, the cp312 ctranslate2
# wheel in brain/wheels/ missing) leaves an interpreter that passes -x but
# cannot import anthropic. With the checkout already at the tip, a HEAD-to-HEAD
# manifest diff would never fire again and that venv would stay broken forever.
manifest_hash=$(cat "$REPO/brain/pyproject.toml" "$REPO/brain/uv.lock" 2>/dev/null |
    sha256sum | cut -d' ' -f1)
need_sync=0
[ -x "$REPO/brain/.venv/bin/python" ] || need_sync=1
if [ -z "$manifest_hash" ] || [ "$(cat "$SYNC_STAMP" 2>/dev/null)" != "$manifest_hash" ]; then
    need_sync=1
fi
if [ "$need_sync" -eq 1 ]; then
    if [ -x "$UV" ]; then
        log "venv does not match brain/pyproject.toml + brain/uv.lock — uv sync"
        # --frozen: install exactly what uv.lock says, never re-resolve. Keeps
        # the deploy checkout clean and makes the installed set reproducible.
        # --extra dev: the device keeps pytest/ruff, so `python -m pytest` works
        # over SSH and an auto-sync doesn't silently uninstall them.
        if ( cd "$REPO/brain" && timeout 900 "$UV" sync --frozen --extra dev ); then
            printf '%s\n' "$manifest_hash" >"$SYNC_STAMP" ||
                log "could not write $SYNC_STAMP — the next start will sync again"
        else
            rm -f "$SYNC_STAMP"
            log "uv sync FAILED — starting with the venv as-is, retrying next start"
        fi
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
