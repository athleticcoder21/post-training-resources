#!/usr/bin/env python3
"""Check local links, anchors, and resource sections in the guide."""

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")
IMAGE = re.compile(r"(?:<img\s+[^>]*src=[\"']([^\"']+)[\"']|!\[[^]]*\]\(([^)]+)\))")


def anchor(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[`*_~]", "", text).strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return re.sub(r" +", "-", text)


def main() -> int:
    guide = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "README.md"
    lines = guide.read_text(encoding="utf-8").splitlines()
    errors: list[str] = []
    anchors: set[str] = set()
    counts: dict[str, int] = {}

    for line in lines:
        match = HEADING.match(line)
        if match:
            base = anchor(match.group(2))
            count = counts.get(base, 0)
            counts[base] = count + 1
            anchors.add(base if count == 0 else f"{base}-{count}")

    current_section = ""
    section_line = 0
    resource_count = 0
    section_urls: set[str] = set()

    def finish_section() -> None:
        if section_line and resource_count == 0:
            errors.append(f"{guide.name}:{section_line}: {current_section!r} has no resources")

    for number, line in enumerate(lines, 1):
        heading = HEADING.match(line)
        if heading:
            if len(heading.group(1)) <= 2:
                finish_section()
                section_line = 0
                section_urls = set()
            elif len(heading.group(1)) == 3:
                finish_section()
                current_section = heading.group(2)
                section_line = number
                resource_count = 0
                section_urls = set()

        targets = LINK.findall(line)
        targets.extend(left or right for left, right in IMAGE.findall(line))
        for target in targets:
            if target.startswith("#"):
                if target[1:] not in anchors:
                    errors.append(f"{guide.name}:{number}: broken anchor {target}")
                continue

            parsed = urlsplit(target)
            if parsed.scheme in {"http", "https"}:
                if parsed.scheme != "https":
                    errors.append(f"{guide.name}:{number}: use HTTPS for {target}")
                if section_line and target in LINK.findall(line):
                    resource_count += 1
                    normalized = target.rstrip("/")
                    if normalized in section_urls:
                        errors.append(f"{guide.name}:{number}: duplicate in {current_section!r}: {target}")
                    section_urls.add(normalized)
                continue

            if parsed.scheme:
                errors.append(f"{guide.name}:{number}: unsupported link {target}")
                continue

            local = (guide.parent / unquote(parsed.path)).resolve()
            if not local.is_relative_to(ROOT) or not local.is_file():
                errors.append(f"{guide.name}:{number}: missing local file {target}")

    finish_section()

    if errors:
        print("Guide checks failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Guide checks passed: {len(anchors)} headings and local links verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
