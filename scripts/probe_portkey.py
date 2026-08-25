#!/usr/bin/env python
"""Discover which model slugs this Portkey gateway actually serves.

Model availability depends on the gateway's provider configs, so it has to be
probed rather than assumed. Sends one trivial text request per candidate, then
one single-frame vision request to the survivors -- a model that answers text
may still reject image content, and this benchmark is useless without vision.
"""
import base64, io, os, sys
from portkey_ai import Portkey

CANDIDATES = [
    "@anthropic-default/claude-opus-5",
    "@anthropic-default/claude-sonnet-5",
    "@anthropic-default/claude-opus-4-8",
    "@anthropic-default/claude-sonnet-4-6",
    "@google-gemini-default/gemini-3-pro",
    "@google-gemini-default/gemini-2.5-pro",
    "@google-gemini-default/gemini-2.5-flash",
]

def tiny_jpeg() -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (128, 64, 200)).save(buf, format="JPEG")
    return base64.standard_b64encode(buf.getvalue()).decode()

def main():
    key = os.environ.get("PORTKEY_API_KEY")
    if not key:
        sys.exit("PORTKEY_API_KEY is not set")
    client = Portkey(api_key=key)
    img = tiny_jpeg()
    ok_text = []

    print("=== text ===")
    for m in CANDIDATES:
        try:
            r = client.chat.completions.create(
                model=m, max_tokens=16,
                messages=[{"role": "user", "content": "Reply with exactly: ok"}])
            print(f"  OK    {m}  -> {(r.choices[0].message.content or '').strip()!r}")
            ok_text.append(m)
        except Exception as e:
            print(f"  FAIL  {m}  -> {type(e).__name__}: {str(e)[:110]}")

    print("\n=== vision (1 image) ===")
    vision = []
    for m in ok_text:
        try:
            r = client.chat.completions.create(
                model=m, max_tokens=24,
                messages=[{"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
                    {"type": "text", "text": "Name the dominant colour in one word."}]}])
            print(f"  OK    {m}  -> {(r.choices[0].message.content or '').strip()[:40]!r}")
            vision.append(m)
        except Exception as e:
            print(f"  FAIL  {m}  -> {type(e).__name__}: {str(e)[:110]}")

    print("\n=== many-image limit probe (survivors only) ===")
    for m in vision:
        for n in (20, 60, 100, 128):
            try:
                client.chat.completions.create(
                    model=m, max_tokens=16,
                    messages=[{"role": "user", "content":
                        [{"type": "image_url",
                          "image_url": {"url": f"data:image/jpeg;base64,{img}"}}] * n
                        + [{"type": "text", "text": "How many images? One number."}]}])
                print(f"  OK    {m}  n={n}")
            except Exception as e:
                print(f"  FAIL  {m}  n={n}  {type(e).__name__}: {str(e)[:90]}")
                break

    print("\nvision-capable:", vision or "(none)")

if __name__ == "__main__":
    main()
