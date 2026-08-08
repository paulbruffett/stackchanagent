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
systemctl --user restart stackchan-brain          # also pulls + updates
journalctl --user-unit=stackchan-brain -f
journalctl --user-unit=stackchan-brain -b         # this boot only
```

Note `--user-unit=`, not `--user -u`. This host has no `/var/log/journal`, so
journald is volatile and the plain `--user` journal namespace is empty;
`--user-unit=` reads the system journal filtered to the unit and works.

To hack on the Jetson checkout without the next restart wiping it, stop the
timer on updates by masking the branch instead:

```bash
systemctl --user stop stackchan-brain
# ... edit, run `python agent_server.py` by hand ...
systemctl --user start stackchan-brain       # resets to origin/main again
```

## What the update does

On each start, `update-brain.sh`:

1. Skips the git half if it already fetched in the last 60 s (a crash loop under
   `Restart=always` must not become a fetch loop). The venv check in step 4 runs
   regardless — a broken venv has to be repairable on a restart, and on a boot
   where origin is unreachable.
2. `git fetch origin <branch>`, retrying for ~30 s — the user manager has no
   `network-online.target` to order against and DHCP may not be up yet. The
   cooldown stamp is armed before the ladder, so a failed fetch is throttled
   the same as a successful one.
3. `git checkout -f <branch> && git reset --hard origin/<branch>`, after
   clearing a `.git/index.lock` older than 5 minutes (a power cut mid-reset
   leaves one behind and every later checkout fails on it). **The checkout is a
   deploy target, not a workspace**: local edits to tracked files are discarded.
   Untracked/ignored files survive — `.env`, `brain/.venv/`,
   `brain/wheels/*.whl` — and runtime state lives in `~/.stackchan/memory.db`,
   outside the repo.
4. `uv sync --frozen` unless `brain/.venv/.stackchan-sync-ok` matches the hash
   of `brain/pyproject.toml` + `brain/uv.lock`. That stamp is written only by a
   sync that ran to completion: `uv` creates `.venv/bin/python` before it
   installs anything into it, so "the interpreter exists" does not mean the
   dependencies are there, and an interrupted sync must re-run on the next
   start rather than crash-looping on `ImportError` forever.

Every step is non-fatal. No network, unreachable origin, failed sync — it logs
and starts the brain on whatever is already on disk. Nothing is permanently
given up on either: each failure is retried on the next start, and the unit has
**no start limit** (`StartLimitIntervalSec=0`, `RestartSec=30`) so it can never
land in `failed` with nobody around to run `systemctl --user reset-failed`.

When the git half fails, `~/.stackchan/update-failed` holds
`<first-failure> <last-failure> <reason>` — this host's journal is volatile, so
that file is the only way to tell a bad night from three weeks of a frozen
checkout. It is removed on the next successful update.

## Ports

- `8765` — firmware WebSocket (all interfaces)
- `8080` — web console (LAN address only, see below)

## Web console access

The console's MCP tab persists a command line that the brain then *runs* as
this user, so it is not open: every API route needs a shared secret, and the
console binds the LAN address rather than `0.0.0.0`.

Put a token in the repo-root `.env` (untracked, so `git reset --hard` on each
start leaves it alone):

```
CONSOLE_TOKEN=some-long-random-string      # e.g. `openssl rand -base64 24`
```

Without one the brain mints a token per run and logs the URL to open:

```
journalctl --user-unit=stackchan-brain -b | grep 'web console'
# web console on http://192.168.4.21:8080/#token=…
```

Open that URL once — the page stores the token and rewrites the address bar,
so the bookmark stays `http://192.168.4.21:8080/`. Paste a new token any time
via the prompt the page shows on a 401.

Two knobs, both in `.env`:

- `CONSOLE_TOKEN` — the shared secret. Unset = per-run token (above).
- `CONSOLE_BIND` — interface override. Default is the LAN address, which
  means `curl localhost:8080` on the Jetson no longer reaches the console —
  use the LAN address, or set `CONSOLE_BIND=0.0.0.0` (or `127.0.0.1` for
  ssh-tunnel-only).

Requests are also rejected when the `Host` header is a name we don't
recognise (only IP literals, `localhost`, `*.local` and this host's own name
pass), so a web page you visit elsewhere can't rebind its domain to the
Jetson and drive the console as if it were same-origin.
