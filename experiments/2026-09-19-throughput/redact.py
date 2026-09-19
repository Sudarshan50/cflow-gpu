#!/usr/bin/env python3
"""Redact live secrets and customer data from material destined for a PUBLIC repo.

Fails loudly (non-zero exit) if any known-live secret survives, so a redaction
miss cannot silently become a published credential.
"""
import ipaddress
import pathlib
import re
import sys

DEPLOY = pathlib.Path("/scratch/deploy")


def read_secret(path):
    p = DEPLOY / path
    if not p.exists():
        return None
    v = p.read_text().strip()
    return v or None


# Values that must never appear in the output.
SECRETS = {}
for label, src in [("VLLM_API_KEY", "api-key.txt"),
                   ("DASH_PASSWORD", "dash-password.txt")]:
    v = read_secret(src)
    if v:
        SECRETS[v] = f"<REDACTED_{label}>"

env = DEPLOY / "vllm-k3.env"
if env.exists():
    for line in env.read_text().splitlines():
        if line.startswith("VLLM_API_KEY="):
            v = line.split("=", 1)[1].strip()
            if v:
                SECRETS[v] = "<REDACTED_VLLM_API_KEY>"

# Any sk- style token, even one we have not enumerated (e.g. a rotated key
# captured earlier in the session).
TOKEN_RE = re.compile(r"sk-[A-Za-z0-9_-]{16,}")
# Customer source addresses scraped from the nginx access log.
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
IPV6_RE = re.compile(r"\b(?:[0-9a-fA-F]{0,4}:){3,7}[0-9a-fA-F]{0,4}\b")

# Addresses that are not customer data and are useful to keep for context.
KEEP_V4 = {"127.0.0.1", "0.0.0.0", "127.0.1.1", "255.255.255.255"}


def scrub_v4(m):
    ip = m.group(0)
    if ip in KEEP_V4:
        return ip
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return ip
    if a.is_private or a.is_loopback or a.is_unspecified:
        return ip
    return "<REDACTED_CLIENT_IP>"


def scrub_v6(m):
    s = m.group(0)
    if s.count(":") < 3 or s in ("::1", "::"):
        return s
    # Version strings and times can look like this; require a hex group >1 char.
    try:
        ipaddress.ip_address(s)
    except ValueError:
        return s
    return "<REDACTED_CLIENT_IP6>"


def redact(text):
    for secret, placeholder in SECRETS.items():
        text = text.replace(secret, placeholder)
    text = TOKEN_RE.sub("<REDACTED_TOKEN>", text)
    text = IPV4_RE.sub(scrub_v4, text)
    text = IPV6_RE.sub(scrub_v6, text)
    return text


def verify(text, where):
    bad = []
    for secret in SECRETS:
        if secret in text:
            bad.append(f"{where}: live secret survived redaction")
    if TOKEN_RE.search(text):
        bad.append(f"{where}: sk- token survived redaction")
    return bad


def main(argv):
    if len(argv) != 3:
        print("usage: redact.py <src> <dst>", file=sys.stderr)
        return 2
    src, dst = pathlib.Path(argv[1]), pathlib.Path(argv[2])
    if not SECRETS:
        print("WARNING: no live secrets found to redact against", file=sys.stderr)

    failures = []
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file())
    for p in files:
        try:
            text = p.read_text(errors="replace")
        except Exception as e:
            print(f"skip {p}: {e}", file=sys.stderr)
            continue
        out = redact(text)
        failures += verify(out, str(p))
        target = dst if src.is_file() else dst / p.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(out)
        print(f"redacted {p} -> {target}")

    if failures:
        print("\nREDACTION FAILED:", file=sys.stderr)
        for f in failures:
            print("  " + f, file=sys.stderr)
        return 1
    print(f"\nOK: {len(files)} file(s), no live secrets remain")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
