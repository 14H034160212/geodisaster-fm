"""Render REPORT_TONIGHT.md into outputs/site/report.html so it is reachable at
the live blog deployment as https://geodisaster-fm.pages.dev/report.html .

Uses the same minimal-monospace style as the main blog so the report and the
narrative dashboard look like part of one project. Also adds a small banner
that links back to the main index.
"""
from __future__ import annotations
from pathlib import Path
import markdown

SRC = Path("REPORT_TONIGHT.md")
OUT = Path("outputs/site/report.html")

CSS = """
:root {
  --fg: #1a1a1a; --muted: #6b7280; --bg: #fbfaf6;
  --accent: #1f5fbe; --good: #1c7f4f; --warn: #a86a1f; --bad: #b3261e;
  --line: #e5e3da; --kbd: #f0eee5;
}
* { box-sizing: border-box; }
html, body { background: var(--bg); color: var(--fg); }
body {
  max-width: 860px; margin: 0 auto; padding: 36px 32px 80px;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
  font-size: 15.5px; line-height: 1.65;
}
header.banner {
  display: flex; justify-content: space-between; align-items: center;
  padding: 14px 18px; border: 1px solid var(--line); border-radius: 8px;
  background: white; margin-bottom: 28px; font-size: 13.5px;
}
header.banner a { color: var(--accent); text-decoration: none; }
header.banner a:hover { text-decoration: underline; }
h1 { font-size: 30px; font-weight: 700; line-height: 1.25; margin: 18px 0 10px; }
h2 { font-size: 22px; font-weight: 650; margin: 36px 0 12px;
     padding-top: 14px; border-top: 1px solid var(--line); }
h3 { font-size: 17.5px; font-weight: 650; margin: 24px 0 8px; color: #2a2a2a; }
h4 { font-size: 15.5px; font-weight: 650; margin: 20px 0 4px; color: #444; }
p, li { color: var(--fg); }
strong { color: #000; font-weight: 650; }
em { color: #444; }
ul, ol { padding-left: 22px; }
code, kbd {
  font-family: 'JetBrains Mono', SF Mono, Menlo, monospace;
  font-size: 13px; background: var(--kbd); padding: 1px 5px; border-radius: 3px;
}
pre { background: white; padding: 14px 18px; border: 1px solid var(--line);
      border-radius: 6px; overflow-x: auto; font-size: 13px; }
pre code { background: transparent; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 14px; }
th, td { padding: 7px 10px; text-align: left; border-bottom: 1px solid var(--line); vertical-align: top; }
th { background: white; font-weight: 650; }
td.num, .num { font-variant-numeric: tabular-nums; font-family: 'JetBrains Mono', monospace; }
blockquote { border-left: 3px solid var(--accent); padding: 4px 16px;
             margin: 14px 0; background: #f3f1ea; color: #333; font-style: italic; }
hr { border: none; border-top: 1px solid var(--line); margin: 28px 0; }
.footer { color: var(--muted); font-size: 12.5px; margin-top: 48px;
          padding-top: 14px; border-top: 1px solid var(--line); }
"""

BANNER = """
<header class="banner">
  <div>
    <strong>GeoDisaster-FM</strong> · advisor progress report (snapshot)
  </div>
  <div>
    <a href="./index.html">← Main blog / live dashboard</a> ·
    <a href="https://github.com/14H034160212/geodisaster-fm">GitHub</a>
  </div>
</header>
"""

FOOTER = """
<div class="footer">
  Snapshot of <code>REPORT_TONIGHT.md</code> committed to
  <a href="https://github.com/14H034160212/geodisaster-fm">14H034160212/geodisaster-fm</a>.
  This page is auto-generated from the markdown source by
  <code>scripts/render_report_page.py</code>; the live blog narrative is at
  <a href="./index.html">index.html</a>.
</div>
"""


def main():
    md_text = SRC.read_text(encoding="utf-8")
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"])
    body_html = md.convert(md_text)
    page = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>GeoDisaster-FM — advisor progress report</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono&display=swap" rel="stylesheet"/>
<style>{CSS}</style>
</head><body>
{BANNER}
{body_html}
{FOOTER}
</body></html>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"Saved {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
