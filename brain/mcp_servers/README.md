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
