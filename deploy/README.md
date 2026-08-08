# deploy

Runs the brain as a systemd **user** service on the Jetson: starts at boot,
restarts on crash, and pulls the tracked branch on every start.

The Jetson has no passwordless sudo, so this is a user unit plus
`loginctl enable-linger` (which the user is allowed to set for itself) rather
than a system unit. Nothing here needs root.

## Install

```bash
ssh jetson
cd ~/code/stackchanagent
./deploy/install.sh          # STACKCHAN_BRANCH=... to track something else
systemctl --user restart stackchan-brain
```

Re-run `install.sh` after editing `stackchan-brain.service` — the unit needs a
`daemon-reload`, which the update script deliberately does not issue from
inside its own start job. `update-brain.sh` re-installs *itself* automatically
and logs a NOTE when the unit template has drifted.

## Operating

```bash
systemctl --user status stackchan-brain
systemctl --user restart stackchan-brain     # also pulls + updates
journalctl --user -u stackchan-brain -f
journalctl --user -u stackchan-brain -b      # this boot only
```

To hack on the Jetson checkout without the next restart wiping it, stop the
timer on updates by masking the branch instead:

```bash
systemctl --user stop stackchan-brain
# ... edit, run `python agent_server.py` by hand ...
systemctl --user start stackchan-brain       # resets to origin/main again
```

## What the update does

On each start, `update-brain.sh`:

1. Skips entirely if it already fetched in the last 60 s (a crash loop under
   `Restart=always` must not become a fetch loop).
2. `git fetch origin <branch>`, retrying for ~30 s — the user manager has no
   `network-online.target` to order against and DHCP may not be up yet.
3. `git checkout -f <branch> && git reset --hard origin/<branch>`. **The
   checkout is a deploy target, not a workspace**: local edits to tracked files
   are discarded. Untracked/ignored files survive — `.env`,
   `brain/.venv/`, `brain/wheels/*.whl` — and runtime state lives in
   `~/.stackchan/memory.db`, outside the repo.
4. `uv sync --frozen` only if `brain/pyproject.toml` or `brain/uv.lock` changed
   between the old and new HEAD (or the venv is missing).

Every step is non-fatal. No network, unreachable origin, failed sync — it logs
and starts the brain on whatever is already on disk.

## Ports

- `8765` — firmware WebSocket
- `8080` — web console
