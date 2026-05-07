#!/usr/bin/env python3
"""Generate the portfolio-pages section of README.md from docs/pages/*.html.

For each HTML page in docs/pages/, extracts the <title> tag (with
" · AA Pricing Forecast" suffix stripped) and the <meta name="description">
content. Groups pages by category and replaces the block between
the AA-PORTFOLIO-PAGES markers in README.md.

Links are emitted as absolute GitHub Pages URLs so they're clickable from
the README on github.com. The base URL can be overridden via the
PORTFOLIO_BASE_URL env var (e.g. for a different fork or a custom domain).

Usage:
    python3 scripts/generate_readme_links.py            # rewrite README in place
    python3 scripts/generate_readme_links.py --check    # exit non-zero if stale
    PORTFOLIO_BASE_URL=https://example.com/ python3 scripts/generate_readme_links.py

CI: .github/workflows/sync-readme-pages.yml runs this on push to master
and auto-commits the result.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGES_DIR = ROOT / "docs" / "pages"
README = ROOT / "README.md"

# Default base URL of the deployed GitHub Pages site. Override with
# PORTFOLIO_BASE_URL=... at script invocation time.
DEFAULT_BASE_URL = "https://srikanthvelpuri.github.io/timesfm/"

MARK_START = "<!-- AA-PORTFOLIO-PAGES:START -->"
MARK_END = "<!-- AA-PORTFOLIO-PAGES:END -->"

# Page categorization. Filenames without ".html". Pages not listed here
# fall through to "Other pages" so adding a page never silently drops it
# from the README.
CATEGORIES: list[tuple[str, list[str]]] = [
    (
        "Strategy & framing",
        [
            "when-to-use",
            "pricing-algorithms",
            "base-price-strategy",
            "alt-models-bakeoff",
            "market-routes",
            "business-impact",
        ],
    ),
    (
        "Model architecture deep-dives",
        [
            "model-choice",
            "timesfm-internals",
            "foundation-ts-survey",
            "transformer-variants-ts",
        ],
    ),
    (
        "Technique deep-dives",
        [
            "finetuning",
            "peft-deepdive",
            "probabilistic-heads",
            "causal-elasticity-deepdive",
            "pricing-optimization",
            "hierarchical-forecasting",
        ],
    ),
    (
        "Production engineering",
        [
            "inference",
            "evaluation",
            "monitoring",
            "azure-architecture",
        ],
    ),
    (
        "Tradeoffs & interview prep",
        [
            "tradeoffs-deepdive",
            "interview-qa",
            "cheatsheet",
        ],
    ),
    (
        "Scenarios",
        [
            "scenario-holiday",
            "scenario-fuel",
            "scenario-competitor",
            "scenario-weather",
            "scenario-coldstart",
            "scenario-cabin",
        ],
    ),
]

TITLE_RE = re.compile(r"<title>(.*?)</title>", re.DOTALL | re.IGNORECASE)
META_DESC_RE = re.compile(
    r'<meta\s+name=[\'"]description[\'"]\s+content=[\'"](.*?)[\'"]\s*/?>',
    re.DOTALL | re.IGNORECASE,
)
TITLE_SUFFIX = " · AA Pricing Forecast"


def html_unescape(s: str) -> str:
    """Stdlib-only HTML entity unescape, for the entities we actually use."""
    replacements = [
        ("&middot;", "·"),
        ("&mdash;", "—"),
        ("&ndash;", "–"),
        ("&hellip;", "…"),
        ("&times;", "×"),
        ("&plusmn;", "±"),
        ("&deg;", "°"),
        ("&rsquo;", "’"),
        ("&lsquo;", "‘"),
        ("&rdquo;", "”"),
        ("&ldquo;", "“"),
        ("&apos;", "'"),
        ("&quot;", '"'),
        ("&lt;", "<"),
        ("&gt;", ">"),
        # &amp; must be last so we don't double-decode.
        ("&amp;", "&"),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    return s


def extract_meta(html: str) -> tuple[str, str]:
    title_match = TITLE_RE.search(html)
    desc_match = META_DESC_RE.search(html)
    title = html_unescape(title_match.group(1).strip()) if title_match else ""
    desc = html_unescape(desc_match.group(1).strip()) if desc_match else ""
    if title.endswith(TITLE_SUFFIX):
        title = title[: -len(TITLE_SUFFIX)].rstrip()
    title = re.sub(r"\s+", " ", title)
    desc = re.sub(r"\s+", " ", desc)
    return title, desc


def collect_pages() -> dict[str, tuple[str, str]]:
    pages: dict[str, tuple[str, str]] = {}
    for path in sorted(PAGES_DIR.glob("*.html")):
        title, desc = extract_meta(path.read_text(encoding="utf-8"))
        if not title:
            print(f"WARN: no <title> in {path}", file=sys.stderr)
            continue
        pages[path.stem] = (title, desc)
    return pages


def page_url(base_url: str, fname: str) -> str:
    return f"{base_url.rstrip('/')}/pages/{fname}.html"


def render_section(pages: dict[str, tuple[str, str]], base_url: str) -> str:
    lines: list[str] = []
    lines.append("")
    lines.append(
        "_Auto-generated from `docs/pages/*.html`. Run "
        "`python3 scripts/generate_readme_links.py` to refresh, or push to "
        "`master` and the [`sync-readme-pages`](.github/workflows/sync-readme-pages.yml) "
        "workflow will refresh on your behalf._"
    )
    lines.append("")
    lines.append(
        f"**Entry point:** [{base_url}]({base_url}) — {len(pages)} pages "
        f"deployed on GitHub Pages."
    )
    lines.append("")
    seen: set[str] = set()
    for cat_title, files in CATEGORIES:
        cat_pages = [(f, pages[f]) for f in files if f in pages]
        if not cat_pages:
            continue
        lines.append(f"### {cat_title}")
        lines.append("")
        for fname, (title, desc) in cat_pages:
            seen.add(fname)
            link = page_url(base_url, fname)
            if desc:
                lines.append(f"- **[{title}]({link})** — {desc}")
            else:
                lines.append(f"- **[{title}]({link})**")
        lines.append("")
    other = sorted(f for f in pages if f not in seen)
    if other:
        lines.append("### Other pages")
        lines.append("")
        for fname in other:
            title, desc = pages[fname]
            link = page_url(base_url, fname)
            if desc:
                lines.append(f"- **[{title}]({link})** — {desc}")
            else:
                lines.append(f"- **[{title}]({link})**")
        lines.append("")
    return "\n".join(lines)


def update_readme(section: str, *, check: bool) -> int:
    readme = README.read_text(encoding="utf-8")
    if MARK_START not in readme or MARK_END not in readme:
        print(
            f"ERROR: markers {MARK_START!r} / {MARK_END!r} not found in README.md.\n"
            "Add them where the portfolio links should appear.",
            file=sys.stderr,
        )
        return 2
    pattern = re.compile(
        re.escape(MARK_START) + r".*?" + re.escape(MARK_END), re.DOTALL
    )
    new_block = f"{MARK_START}\n{section}\n{MARK_END}"
    new_readme = pattern.sub(new_block, readme)
    if new_readme == readme:
        print("README.md already up to date.")
        return 0
    if check:
        print("README.md is stale; run scripts/generate_readme_links.py.", file=sys.stderr)
        return 1
    README.write_text(new_readme, encoding="utf-8")
    print("README.md updated.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if README would change; do not write.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("PORTFOLIO_BASE_URL", DEFAULT_BASE_URL),
        help=(
            "Base URL of the deployed GitHub Pages site "
            f"(default: env PORTFOLIO_BASE_URL or {DEFAULT_BASE_URL})."
        ),
    )
    args = parser.parse_args(argv)
    pages = collect_pages()
    if not pages:
        print("ERROR: no pages found in docs/pages/.", file=sys.stderr)
        return 2
    section = render_section(pages, args.base_url)
    return update_readme(section, check=args.check)


if __name__ == "__main__":
    sys.exit(main())
