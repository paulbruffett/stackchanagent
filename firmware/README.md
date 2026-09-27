
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

### Signing key

Every app image is signed (RSA-3072, the secure-boot-v2 signature format,
but **without** hardware secure boot — no eFuses are burned, USB flashing
works as always). An OTA image is only accepted if it is signed with the
same key as the firmware already running, so a rogue update can't be pushed
over the network.

The private key lives **outside the repo**, at
`~/.stackchan-keys/firmware_signing_key.pem` (override with the
`STACKCHAN_SIGNING_KEY` environment variable; `firmware/CMakeLists.txt`
feeds the absolute path to `CONFIG_SECURE_BOOT_SIGNING_KEY`). Create it once
if it is missing:

```bash
mkdir -m 700 -p ~/.stackchan-keys
espsecure.py generate_signing_key --version 2 --scheme rsa3072 \
    ~/.stackchan-keys/firmware_signing_key.pem
chmod 600 ~/.stackchan-keys/firmware_signing_key.pem
```

**Back it up** (password manager / offline copy). Losing it means the robot
refuses every future OTA image, and the only fix is a USB flash of a build
signed with a new key (the same USB flash is how you rotate the key).

An existing `sdkconfig` predates signing and pins
`# CONFIG_SECURE_SIGNED_APPS_NO_SECURE_BOOT is not set`, which beats
`sdkconfig.defaults`. Delete that one line (or turn on *Security features →
Require signed app images* in `idf.py menuconfig`) and rebuild; the build log
then shows "Generating signed binary image". A defaulted key path is only
seeded once — to move the key later, edit `CONFIG_SECURE_BOOT_SIGNING_KEY`
in `sdkconfig`.

### Over-the-air updates

The first build with OTA support and signing has to go over USB (the
firmware before it has no `ota` command and no signing key to check
against). After that, builds can go through the brain console:
**Firmware** tab → upload `build/stack-chan.bin` → **Send to robot**.

- The brain checks the upload (ESP32-S3 app image, fits the 0x4f0000 slot,
  has a signature block) and identifies it by its ELF SHA-256.
- It sends the update only while the robot is free (no conversation), as a
  one-time download URL on the console port. The robot only downloads from
  the brain it is connected to, on its own task, verifies size, SHA-256 and
  the signature, flashes the spare OTA slot and reboots.
- The new image is on probation: marked valid after 30 s of unbroken brain
  link, rolled back automatically if that hasn't happened within 5 min of
  boot, or if it crashes / is reset first. The console shows *done* or
  *rolled back?*.
- A queued update can be cancelled and expires after an hour.

Only the app is updated — bootloader, partition table and the assets
partition still need USB.
