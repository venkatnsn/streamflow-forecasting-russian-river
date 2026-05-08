"""Convert docs/article.md into docs/index.html for GitHub Pages + Medium import.

Same renderer pattern as the rainfall-runoff tutorial article: figure
placeholders -> <img>, gist placeholders -> <pre><code> + URL annotation.
"""
from pathlib import Path
import html
import re

HERE = Path(__file__).parent

GISTS = {
    "GIST_DATA_ACQUISITION":  ("01_data_acquisition.py",
                                "https://gist.github.com/venkatnsn/0dc1ca7c3cdc79ce2f0b8e705261f694"),
    "GIST_TWO_BUCKET_MODEL":  ("02_two_bucket_model.py",
                                "https://gist.github.com/venkatnsn/f27b6a8dad86a948f6a557274f360c22"),
    "GIST_CALIBRATE_DE":      ("03_calibrate_de.py",
                                "https://gist.github.com/venkatnsn/f1cd8e6a8d629899fcbde18ee86d7e15"),
    "GIST_RSA_GLUE":          ("04_rsa_glue.py",
                                "https://gist.github.com/venkatnsn/58fa29514dbeb37fd4ebb58d2ee7b008"),
}
FIGS = {
    "FIG_HERO":         "images/04_glue_uncertainty_band.png",
    "FIG_CAL_VAL":      "images/02_calibration_validation.png",
    "FIG_SENSITIVITY":  "images/03_sensitivity_cdf.png",
    "FIG_GLUE":         "images/04_glue_uncertainty_band.png",
    "FIG_SCENARIOS":    "images/05_climate_scenarios.png",
}


def md_to_html(md: str) -> str:
    lines = md.split("\n")
    out = []
    i = 0

    def inline(s: str) -> str:
        s = re.sub(r"`([^`]+)`",
                   lambda m: f"<code>{html.escape(m.group(1))}</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", s)
        s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
        return s

    while i < len(lines):
        line = lines[i]

        m = re.fullmatch(r"\s*<<([A-Z_]+)>>\s*", line)
        if m:
            key = m.group(1)
            if key in GISTS:
                fname, url = GISTS[key]
                src = (HERE / "gists" / fname).read_text(encoding="utf-8")
                out.append(
                    f'<div class="gist-block">\n'
                    f'<pre><code class="language-python">'
                    f'{html.escape(src.rstrip())}</code></pre>\n'
                    f'<p class="gist-link">Gist (paste this URL alone on a line in '
                    f'Medium for an embedded code block): '
                    f'<a href="{url}">{url}</a></p>\n</div>'
                )
            elif key in FIGS:
                out.append(f'<p><img src="{FIGS[key]}" alt="{key.lower()}"></p>')
            else:
                out.append(f"<!-- unknown placeholder: {key} -->")
            i += 1
            continue

        if line.strip() == "---":
            out.append("<hr>"); i += 1; continue

        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            n = len(m.group(1))
            out.append(f"<h{n}>{inline(m.group(2).strip())}</h{n}>")
            i += 1; continue

        if line.startswith("```"):
            j = i + 1; buf = []
            while j < len(lines) and not lines[j].startswith("```"):
                buf.append(lines[j]); j += 1
            out.append(f"<pre><code>{html.escape(chr(10).join(buf))}</code></pre>")
            i = j + 1; continue

        if line.startswith("> "):
            buf = []
            while i < len(lines) and lines[i].startswith("> "):
                buf.append(lines[i][2:]); i += 1
            out.append(f"<blockquote><p>{inline(' '.join(buf).strip())}</p></blockquote>")
            continue

        if re.match(r"^\s*[-*]\s+", line):
            buf = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                item = re.sub(r"^\s*[-*]\s+", "", lines[i])
                buf.append(f"<li>{inline(item.strip())}</li>")
                i += 1
            out.append("<ul>\n" + "\n".join(buf) + "\n</ul>"); continue

        if re.match(r"^\s*\d+\.\s+", line):
            buf = []
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                item = re.sub(r"^\s*\d+\.\s+", "", lines[i])
                buf.append(f"<li>{inline(item.strip())}</li>")
                i += 1
            out.append("<ol>\n" + "\n".join(buf) + "\n</ol>"); continue

        # Tables (pipes)
        if line.lstrip().startswith("|") and i + 1 < len(lines) and \
                re.match(r"^\s*\|[\s\-:|]+\|\s*$", lines[i + 1]):
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            th = "".join(f"<th>{inline(h)}</th>" for h in header)
            body = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r)
                            + "</tr>" for r in rows)
            out.append(f'<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>')
            continue

        if not line.strip():
            i += 1; continue

        buf = [line]; i += 1
        while (i < len(lines) and lines[i].strip()
               and not lines[i].startswith("#")
               and not lines[i].startswith("```")
               and not lines[i].startswith("> ")
               and not re.fullmatch(r"\s*<<([A-Z_]+)>>\s*", lines[i])
               and not re.match(r"^\s*[-*]\s+", lines[i])
               and not re.match(r"^\s*\d+\.\s+", lines[i])
               and not lines[i].lstrip().startswith("|")
               and lines[i].strip() != "---"):
            buf.append(lines[i]); i += 1
        out.append(f"<p>{inline(' '.join(s.strip() for s in buf))}</p>")
    return "\n\n".join(out)


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Forecasting California's Russian River Under Climate Change</title>
<meta name="description" content="A 30-year, end-to-end Python pipeline: real USGS streamflow, ERA5 climate forcings, calibration, sensitivity, GLUE uncertainty, and climate-change scenarios.">
<style>
  body { max-width: 760px; margin: 0 auto; padding: 48px 24px 96px;
         font-family: 'Charter','Iowan Old Style',Georgia,serif;
         font-size: 19px; line-height: 1.65; color: #222; }
  h1, h2, h3 { font-family: 'Inter','Helvetica Neue',Arial,sans-serif;
               line-height: 1.25; color: #111; }
  h1 { font-size: 2.0em; margin: 0 0 0.3em; }
  h2 { font-size: 1.45em; margin: 1.8em 0 0.4em; }
  h3 { font-size: 1.15em; margin: 1.4em 0 0.3em; }
  p  { margin: 0.8em 0; }
  a  { color: #0b66c3; }
  img { max-width: 100%; height: auto; display: block;
         margin: 1.4em auto; box-shadow: 0 2px 6px rgba(0,0,0,0.06);
         border-radius: 4px; }
  blockquote { border-left: 3px solid #aaa; margin: 1em 0; padding: 0.2em 1.2em;
                color: #666; font-style: italic; }
  pre, code { font-family: 'JetBrains Mono','Menlo','Consolas',monospace; }
  pre { background: #f7f7f7; padding: 14px 18px; border-radius: 6px;
         overflow-x: auto; font-size: 0.85em; line-height: 1.55; }
  code { font-size: 0.92em; }
  pre code { font-size: 1em; }
  hr { border: none; border-top: 1px solid #ddd; margin: 2.4em 0; }
  table { border-collapse: collapse; width: 100%; margin: 1.2em 0;
           font-size: 0.92em; }
  th { background: #222831; color: white; padding: 10px 14px;
        text-align: left; font-weight: 600;
        font-family: 'Inter','Helvetica Neue',Arial,sans-serif; }
  td { padding: 9px 14px; border-bottom: 1px solid #eee;
        vertical-align: top;
        font-family: 'Inter','Helvetica Neue',Arial,sans-serif; }
  tbody tr:nth-child(even) td { background: #f7f7f7; }
  .gist-link { font-family: 'Inter','Helvetica Neue',Arial,sans-serif;
                font-size: 0.78em; color: #666; margin-top: -0.6em; }
  ul, ol { padding-left: 1.4em; }
  li { margin: 0.3em 0; }
</style>
</head>
<body>
__BODY__
</body>
</html>
"""


def main() -> None:
    md = (HERE / "article.md").read_text(encoding="utf-8")
    body = md_to_html(md)
    page = TEMPLATE.replace("__BODY__", body)
    (HERE / "index.html").write_text(page, encoding="utf-8")
    print(f"Wrote {(HERE / 'index.html')}")


if __name__ == "__main__":
    main()
