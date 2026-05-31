# Local MCP servers (Phase 9b)

Small FastMCP servers the brain launches as stdio MCP servers. Register
them in the web console's **MCP** tab, then click **Reload**.

## weather

No setup. Register as a stdio server:

- **command:** `python` (use the venv's python on the Jetson, e.g. `.venv/bin/python`)
- **args:** `mcp_servers/weather.py`

Tool: `get_weather(location?)`. Defaults to the `DEFAULT_LOCATION` config
knob (Seattle) when no location is given. Uses the keyless Open-Meteo
geocoding + forecast APIs — no secret.

## hue

Controls Philips Hue lights via the bridge's local CLIP v2 API. Needs two
values in the brain's project-root `.env` (set once, out-of-band):

```
HUE_BRIDGE_IP=192.168.x.y
HUE_TOKEN=<application key from pairing>
```

Register as a stdio server:

- **command:** `python` (venv python on the Jetson)
- **args:** `mcp_servers/hue.py`
- **secret env var name:** `HUE_TOKEN`  (the registry stores only the name;
  the value stays in `.env`)

`HUE_BRIDGE_IP` is non-secret and passes through to the server's
environment automatically; `HUE_TOKEN` is injected via the `env_ref`.

Tools: `list_lights`, `set_light(name, on?, brightness_pct?)`,
`set_room(name, on?, brightness_pct?)`.

### One-time pairing (get HUE_TOKEN)

1. Find the bridge IP: open <https://discovery.meethue.com> (returns
   `internalipaddress`), or check your router. Put it in `HUE_BRIDGE_IP`.
2. **Press the round link button on top of the bridge**, then within 30 s run:

   ```bash
   curl -k -X POST https://$HUE_BRIDGE_IP/api \
     -H 'Content-Type: application/json' \
     -d '{"devicetype":"stackchan#brain","generateclientkey":true}'
   ```

   The response contains `"username":"<long string>"` — that string is
   your application key. Put it in `HUE_TOKEN`.
   (If you see `"link button not pressed"`, press it and retry.)
3. Restart the brain (or just reload MCP from the console). Verify with
   "computer, what lights do you see?".

`-k` is needed because the bridge uses a self-signed certificate; the
server connects the same way (`verify=False`) — fine for a LAN device.
