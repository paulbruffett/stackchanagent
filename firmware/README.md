
## Build

### Fetch Dependencies

```bash
python3 ./fetch_repos.py
```

### Tool Chains

[ESP-IDF v5.5.4](https://docs.espressif.com/projects/esp-idf/en/v5.5.4/esp32s3/index.html) installed at `~/esp/esp-idf`.

### Activate ESP-IDF in the shell

`idf.py` is not on `$PATH` by default — each new terminal needs the
ESP-IDF environment sourced first:

```bash
source ~/esp/esp-idf/export.sh
```

Verify with `which idf.py`. Subsequent `idf.py` commands in the same
shell will work.

### Configure the brain host (one-time per dev machine)

The firmware connects to `stackchan-brain.local` by default. For Mac
dev (where zeroconf can't bind mDNS port 5353), point it at your Mac's
own `.local` hostname:

```bash
idf.py menuconfig
# → Stackchan Brain
# → Brain host
# Type your host (e.g. "Pauls-Mac-mini.local"), save, quit.
```

The choice persists in `sdkconfig` (which is per-checkout, not
checked in). Revert via the same menu — defaults are
`stackchan-brain.local` + port `8765`.

### Build

```bash
idf.py build
```

After adding a new `.cpp` under `main/`, run `idf.py reconfigure` once
so the CMake glob picks it up.

### Flash

```bash
idf.py -p /dev/cu.usbmodem21101 flash monitor
```

Replace the port with whatever your CoreS3 enumerates as (`ls
/dev/cu.usbmodem*`). Exit the serial monitor with `Ctrl+]`.
