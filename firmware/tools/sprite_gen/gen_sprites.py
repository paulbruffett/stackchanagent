#!/usr/bin/env python3
"""Generate Stack-chan avatar sprites via OpenRouter (Google "nano banana" image models).

Reads a JSON manifest describing each sprite (name, target size, prompt) plus a
shared reference image, then loops over the sprites and writes one PNG each.
Retained as the asset-generation tool: add entries to sprites.json and re-run to
produce new sprites; existing PNGs are skipped unless --force.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python3 gen_sprites.py                       # generate every missing sprite
    python3 gen_sprites.py --only rocky_busy     # one (or a comma list)
    python3 gen_sprites.py --force               # regenerate everything
    python3 gen_sprites.py --model google/gemini-3.1-flash-image-preview

Requires Pillow (pip3 install Pillow): the image models emit opaque JPEG with a
flat key-color background, which this script chroma-keys to real alpha and
normalizes to PNG (downscaled + transparent-padded to the target px).
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES = 4
TIMEOUT_S = 300


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def load_dotenv(start: Path) -> None:
    """Populate missing env vars from the nearest .env walking up from `start`."""
    for d in [start, *start.parents]:
        env = d / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
            return


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def build_messages(sprite: dict, manifest: dict, base_dir: Path, ref_default) -> list:
    """Assemble the user message: prefix + sprite prompt, optionally + reference image(s)."""
    use_ref = sprite.get("use_reference", True)
    prefix = sprite.get("prefix") or (
        manifest.get("global_prefix", "") if use_ref else manifest.get("overlay_prefix", "")
    )
    size = sprite.get("size", manifest.get("default_size", 160))
    text = (
        f"{prefix}\n\n{sprite['prompt']}\n\n"
        f"Output a single square image suitable for downscaling to {size}x{size} pixels."
    )

    content = [{"type": "text", "text": text}]
    if use_ref:
        refs = sprite.get("reference_image", ref_default)
        if isinstance(refs, str):
            refs = [refs]
        for ref in refs or []:
            ref_path = (base_dir / ref).resolve()
            if not ref_path.exists():
                die(f"reference image not found for '{sprite['name']}': {ref_path}")
            content.append({"type": "image_url", "image_url": {"url": data_url(ref_path)}})
    return [{"role": "user", "content": content}]


def bytes_data_url(raw: bytes) -> str:
    mime = "image/jpeg" if raw[:2] == b"\xff\xd8" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


HARMONIZE_PREFIX = (
    "Image 1 shows a creature in a target pose. Image 2 is the reference character 'Rocky'. "
    "Repaint the creature from image 1 so it EXACTLY matches Rocky's body design, brown carapace, "
    "glowing green accents, textures, proportions, and EXACTLY FIVE legs/arms from image 2 — while "
    "keeping image 1's pose, orientation, and composition unchanged. Faceless: no eyes, no mouth, no "
    "face. Low-res pixel-art look, hard edges, no anti-aliasing. Fill the ENTIRE background edge to "
    "edge with solid flat pure magenta (hex FF00FF) — NOT a checkerboard, NOT a gradient, no shadows. "
    "Do not use magenta anywhere on the character. No text."
)


def build_harmonize_messages(pose_raw: bytes, ref_rel: str, manifest: dict, base_dir: Path) -> list:
    """Pass 2: feed the posed (drifted) image + the reference, ask to repaint onto the reference design."""
    ref_path = (base_dir / ref_rel).resolve()
    if not ref_path.exists():
        die(f"harmonize reference not found: {ref_path}")
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": manifest.get("harmonize_prefix", HARMONIZE_PREFIX)},
                {"type": "image_url", "image_url": {"url": bytes_data_url(pose_raw)}},
                {"type": "image_url", "image_url": {"url": data_url(ref_path)}},
            ],
        }
    ]


def extract_image_b64(message: dict) -> str | None:
    """Pull the generated image's base64 payload out of an OpenRouter chat response."""
    def from_url(url: str) -> str | None:
        return url.split("base64,", 1)[1] if url and "base64," in url else None

    for img in message.get("images") or []:
        url = (img.get("image_url") or {}).get("url") or img.get("url", "")
        if (b64 := from_url(url)):
            return b64

    content = message.get("content")
    if isinstance(content, str):
        return from_url(content)
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                url = (part.get("image_url") or {}).get("url", "")
                if (b64 := from_url(url)):
                    return b64
    return None


def request_image(model: str, messages: list, api_key: str) -> dict:
    body = json.dumps(
        {"model": model, "messages": messages, "modalities": ["image", "text"]}
    ).encode()
    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/paulbruffett/stackchan",
            "X-Title": "stackchan-sprite-gen",
        },
        method="POST",
    )
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:400]
            last_err = f"HTTP {e.code}: {detail}"
            if e.code in (408, 409, 429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                wait = 2 ** attempt
                print(f"    transient {e.code}, retry {attempt}/{MAX_RETRIES} in {wait}s")
                time.sleep(wait)
                continue
            break
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = str(e)
            if attempt < MAX_RETRIES:
                wait = 2 ** attempt
                print(f"    network error, retry {attempt}/{MAX_RETRIES} in {wait}s")
                time.sleep(wait)
                continue
            break
    raise RuntimeError(last_err or "unknown error")


def generate(model: str, messages: list, api_key: str) -> bytes:
    """One image request -> decoded image bytes."""
    resp = request_image(model, messages, api_key)
    b64 = extract_image_b64(resp["choices"][0]["message"])
    if not b64:
        raise RuntimeError("no image in response (check model supports image output)")
    return base64.b64decode(b64)


def hex_rgb(value: str) -> tuple:
    v = value.lstrip("#")
    return tuple(int(v[i : i + 2], 16) for i in (0, 2, 4))


def chroma_key(im, key_color: str, tol: int = 70):
    """Make pixels near key_color transparent. nano-banana emits opaque JPEG, so
    the background is a flat key color we key out here (not real alpha)."""
    from PIL import ImageChops

    kr, kg, kb = hex_rgb(key_color)
    r, g, b = im.convert("RGB").split()
    dr = r.point(lambda v: abs(v - kr))
    dg = g.point(lambda v: abs(v - kg))
    db = b.point(lambda v: abs(v - kb))
    diff = ImageChops.lighter(ImageChops.lighter(dr, dg), db)  # max per-channel distance
    alpha = diff.point(lambda v: 255 if v > tol else 0)
    im = im.convert("RGBA")
    im.putalpha(alpha)
    return im


def save_png(raw: bytes, out_path: Path, size: int, resize: bool, key_color: str | None) -> None:
    """Normalize model output (often JPEG) to a real PNG: key out the background,
    optionally downscale + transparent-pad to the target px."""
    from io import BytesIO

    from PIL import Image

    im = Image.open(BytesIO(raw)).convert("RGBA")
    if key_color:
        im = chroma_key(im, key_color)
    if resize:
        im.thumbnail((size, size), Image.LANCZOS)
        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2), im)
        im = canvas
    im.save(out_path)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(script_dir / "sprites.json"))
    ap.add_argument("--only", help="comma-separated sprite names to generate")
    ap.add_argument("--force", action="store_true", help="regenerate sprites that already exist")
    ap.add_argument("--model", help="override the model id in the manifest")
    ap.add_argument("--reference", help="reference image path (overrides reference_image in the manifest), relative to the manifest dir or absolute")
    ap.add_argument("--no-resize", action="store_true", help="keep full-res keyed output instead of downscaling to target px")
    ap.add_argument("--no-harmonize", action="store_true", help="skip the pass-2 reference harmonize for harmonize sprites (faster/cheaper pose-only output)")
    ap.add_argument("--open", action="store_true", help="open each generated PNG in the default viewer (test workflow)")
    args = ap.parse_args()

    load_dotenv(script_dir)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        die("OPENROUTER_API_KEY is not set (env or .env)")

    try:
        import PIL  # noqa: F401 - required for chroma-key + PNG normalization
    except ImportError:
        die("Pillow is required: pip3 install Pillow")

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        die(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    base_dir = manifest_path.parent

    model = args.model or manifest.get("model")
    if not model:
        die("no model specified in manifest or --model")

    ref_default = args.reference or manifest.get("reference_image")

    out_dir = (base_dir / manifest.get("output_dir", "out")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(filter(None, (args.only or "").split(",")))
    sprites = [s for s in manifest["sprites"] if not wanted or s["name"] in wanted]
    if wanted:
        missing = wanted - {s["name"] for s in sprites}
        if missing:
            die(f"unknown sprite name(s): {', '.join(sorted(missing))}")
    if not sprites:
        die("no sprites to generate")

    print(f"model: {model}")
    print(f"output: {out_dir}")
    print(f"sprites: {len(sprites)}\n")

    failures = []
    for sprite in sprites:
        name = sprite["name"]
        size = sprite.get("size", manifest.get("default_size", 160))
        out_path = out_dir / f"{name}.png"
        if out_path.exists() and not args.force:
            print(f"  {name}: exists, skipping (use --force)")
            continue
        print(f"  {name}: generating ({size}px)...")
        try:
            messages = build_messages(sprite, manifest, base_dir, ref_default)
            raw = generate(model, messages, api_key)
            # Pass 2: repaint the posed image onto the reference design (faithful + posed).
            if sprite.get("harmonize") and not args.no_harmonize:
                ref_rel = sprite.get("reference_image", ref_default)
                if not ref_rel:
                    raise RuntimeError("harmonize requires a reference image (manifest or --reference)")
                print("    harmonizing against reference...")
                raw = generate(model, build_harmonize_messages(raw, ref_rel, manifest, base_dir), api_key)
            key_color = sprite.get("key_color", manifest.get("key_color"))
            save_png(raw, out_path, size, resize=not args.no_resize, key_color=key_color)
            print(f"    -> {out_path}")
            if args.open:
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.run([opener, str(out_path)], check=False)
        except Exception as e:  # noqa: BLE001 - report and continue the batch
            print(f"    FAILED: {e}")
            failures.append(name)

    print()
    if failures:
        print(f"done with {len(failures)} failure(s): {', '.join(failures)}")
        sys.exit(1)
    print("done")


if __name__ == "__main__":
    main()
