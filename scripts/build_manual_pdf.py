#!/usr/bin/env python3
"""Convert USER_MANUAL.md to USER_MANUAL.pdf using weasyprint."""

import sys
from pathlib import Path

try:
    import markdown
    from weasyprint import HTML, CSS
except ImportError:
    print("ERROR: Run: pip install markdown weasyprint", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).parent.parent
MD_PATH = REPO_ROOT / "USER_MANUAL.md"
PDF_PATH = REPO_ROOT / "USER_MANUAL.pdf"

CSS_STYLES = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

@page {
    size: letter;
    margin: 1in 0.85in 1in 0.85in;
    @bottom-center {
        content: "AITrading User Manual  ·  Page " counter(page) " of " counter(pages);
        font-family: 'Inter', sans-serif;
        font-size: 9pt;
        color: #94a3b8;
    }
    @top-right {
        content: "AITrading";
        font-family: 'Inter', sans-serif;
        font-size: 9pt;
        color: #94a3b8;
    }
}

* { box-sizing: border-box; }

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 10.5pt;
    line-height: 1.65;
    color: #1e293b;
    margin: 0;
    padding: 0;
}

/* Cover-style title */
h1:first-of-type {
    font-size: 28pt;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 6pt 0;
    padding-bottom: 10pt;
    border-bottom: 3px solid #3b82f6;
    break-after: avoid;
}

h1 { font-size: 18pt; font-weight: 700; color: #0f172a; margin: 20pt 0 8pt 0; break-after: avoid; }
h2 { font-size: 14pt; font-weight: 600; color: #1e3a5f; margin: 16pt 0 6pt 0; border-bottom: 1px solid #e2e8f0; padding-bottom: 3pt; break-after: avoid; }
h3 { font-size: 11pt; font-weight: 600; color: #334155; margin: 12pt 0 4pt 0; break-after: avoid; }
h4 { font-size: 10.5pt; font-weight: 600; color: #475569; margin: 10pt 0 3pt 0; break-after: avoid; }

p { margin: 0 0 7pt 0; orphans: 3; widows: 3; }
li { margin-bottom: 3pt; }
ul, ol { margin: 0 0 8pt 0; padding-left: 20pt; }

a { color: #2563eb; text-decoration: none; }

/* Inline code */
code {
    font-family: 'JetBrains Mono', 'Consolas', 'Monaco', monospace;
    font-size: 9pt;
    background: #f1f5f9;
    color: #be185d;
    padding: 1pt 4pt;
    border-radius: 3pt;
    border: 1px solid #e2e8f0;
}

/* Code blocks */
pre {
    background: #0f172a;
    color: #e2e8f0;
    padding: 12pt 14pt;
    border-radius: 6pt;
    margin: 8pt 0 10pt 0;
    overflow-x: auto;
    break-inside: avoid;
}
pre code {
    background: none;
    color: #e2e8f0;
    border: none;
    padding: 0;
    font-size: 8.5pt;
    line-height: 1.55;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 8pt 0 10pt 0;
    font-size: 9.5pt;
    break-inside: avoid;
}
th {
    background: #1e3a5f;
    color: #ffffff;
    font-weight: 600;
    padding: 6pt 8pt;
    text-align: left;
    border: 1px solid #1e3a5f;
}
td {
    padding: 5pt 8pt;
    border: 1px solid #e2e8f0;
    vertical-align: top;
}
tr:nth-child(even) td { background: #f8fafc; }

/* Blockquote / callout */
blockquote {
    background: #eff6ff;
    border-left: 4pt solid #3b82f6;
    margin: 8pt 0;
    padding: 8pt 12pt;
    border-radius: 0 4pt 4pt 0;
    break-inside: avoid;
}
blockquote p { margin: 0; color: #1e40af; }

/* Horizontal rule */
hr {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 14pt 0;
}

/* Table of contents section */
h2:first-of-type + ol, h2:first-of-type + ul {
    font-size: 10pt;
    column-count: 2;
    column-gap: 20pt;
}

/* Page breaks before major sections */
h2 { break-before: auto; }

/* Keep headings with their content */
h2 + p, h2 + ul, h2 + ol, h2 + pre, h2 + table { break-before: avoid; }
h3 + p, h3 + ul, h3 + ol, h3 + pre, h3 + table { break-before: avoid; }

/* Italic and bold */
em { color: #0369a1; font-style: italic; }
strong { color: #0f172a; font-weight: 600; }

/* Warning/note callouts (bold text starting with Warning) */
p strong:first-child { }
"""


def build_pdf():
    md_text = MD_PATH.read_text(encoding="utf-8")

    md_parser = markdown.Markdown(
        extensions=["tables", "fenced_code", "toc", "attr_list"],
    )
    html_body = md_parser.convert(md_text)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width">
<title>AITrading User Manual</title>
</head>
<body>
{html_body}
</body>
</html>"""

    print(f"Converting {MD_PATH.name} → {PDF_PATH.name} ...")
    html_doc = HTML(string=full_html, base_url=str(REPO_ROOT))
    css = CSS(string=CSS_STYLES)
    html_doc.write_pdf(str(PDF_PATH), stylesheets=[css])
    size_kb = PDF_PATH.stat().st_size // 1024
    print(f"Done — {PDF_PATH.name} ({size_kb} KB)")


if __name__ == "__main__":
    build_pdf()
