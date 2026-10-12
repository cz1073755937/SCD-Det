"""Fail if the public tree contains common private, generated, or oversized artifacts."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_DIRS = {"__pycache__", "datasets", "runs", "runs_revision", "weights"}
BLOCKED_SUFFIXES = {
    ".docx",
    ".engine",
    ".onnx",
    ".pdf",
    ".pth",
    ".pt",
    ".tar",
    ".zip",
}
TEXT_SUFFIXES = {"", ".cff", ".md", ".patch", ".py", ".toml", ".txt", ".yaml", ".yml"}
SENSITIVE_PATTERNS = {
    "Linux home path": re.compile(r"/home/[A-Za-z0-9._-]+/"),
    "macOS home path": re.compile(r"/Users/[A-Za-z0-9._-]+/"),
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"gh[ps]_[A-Za-z0-9]{30,}"),
}
MAX_FILE_BYTES = 5 * 1024 * 1024


def main() -> None:
    failures: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(part in BLOCKED_DIRS for part in relative.parts):
            failures.append(f"blocked directory: {relative}")
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            failures.append(f"blocked file type: {relative}")
        if path.stat().st_size > MAX_FILE_BYTES:
            failures.append(f"file exceeds 5 MiB: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            for label, pattern in SENSITIVE_PATTERNS.items():
                if pattern.search(text):
                    failures.append(f"{label}: {relative}")

    if failures:
        raise SystemExit("Release check failed:\n- " + "\n- ".join(sorted(set(failures))))
    print(f"Release check passed: {ROOT}")


if __name__ == "__main__":
    main()
