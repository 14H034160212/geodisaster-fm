"""Assemble the official Springer Nature (sn-jnl, sn-nature style) submission
from MANUSCRIPT.md.

Pipeline:
  1. Split MANUSCRIPT.md into: title / abstract / body (Introduction onward).
  2. pandoc the body with --shift-heading-level-by=-1 so that
       ## -> \section, ### -> \subsection, #### -> \subsubsection.
  3. Drop the markdown References section from the body (sn-jnl handles
     the bibliography; we keep [Author Year] text refs out of the
     rendered body to avoid a duplicate list — see note below).
  4. Emit main_sn.tex using the sn-jnl class with the sn-nature reference
     style, the extracted abstract in \abstract{}, author/affil
     placeholders, and \input{body_sn} + the figures block.
"""
import re, subprocess
from pathlib import Path

ROOT = Path("..").resolve()
MD = ROOT / "MANUSCRIPT.md"
OUT_DIR = Path(".").resolve()
PANDOC = "/tmp/pandoc-3.5/bin/pandoc"


_UNICODE_TEX = {
    "§": r"\S{}", "±": r"$\pm$", "²": r"$^{2}$", "³": r"$^{3}$",
    "·": r"$\cdot$", "×": r"$\times$", "é": r"\'e", "ŷ": r"$\hat{y}$",
    "Δ": r"$\Delta$", "Σ": r"$\Sigma$", "α": r"$\alpha$", "β": r"$\beta$",
    "γ": r"$\gamma$", "λ": r"$\lambda$", "π": r"$\pi$", "ρ": r"$\rho$",
    "σ": r"$\sigma$", "τ": r"$\tau$", "—": "---",
    "⁴": r"$^{4}$", "⁵": r"$^{5}$", "⁶": r"$^{6}$", "⁷": r"$^{7}$",
    "⁸": r"$^{8}$", "⁻": r"$^{-}$",
    "₀": r"$_{0}$", "₁": r"$_{1}$", "₂": r"$_{2}$", "₅": r"$_{5}$",
    "₇": r"$_{7}$", "₉": r"$_{9}$",
    "ℝ": r"$\mathbb{R}$", "→": r"$\rightarrow$", "↔": r"$\leftrightarrow$",
    "⇒": r"$\Rightarrow$", "⇔": r"$\Leftrightarrow$", "∈": r"$\in$",
    "−": "$-$", "∪": r"$\cup$", "≈": r"$\approx$", "≠": r"$\neq$",
    "≤": r"$\leq$", "≥": r"$\geq$", "⊆": r"$\subseteq$",
    "̂": "",  # stray combining circumflex
}

def _unicode_to_tex(t):
    for u, x in _UNICODE_TEX.items():
        t = t.replace(u, x)
    return t


def _texesc(t):
    # escape LaTeX specials first (text has no backslashes of its own)
    for a, b in [("\\", r"\textbackslash{}"), ("%", r"\%"), ("&", r"\&"),
                 ("_", r"\_"), ("#", r"\#"), ("$", r"\$")]:
        t = t.replace(a, b)
    # then markdown emphasis -> LaTeX
    t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\\emph{\1}", t)
    return t

src = MD.read_text()

# 1. title = the single leading '# ' line
title = re.search(r"^# (.+)$", src, re.M).group(1).strip()

# 2. abstract block
abstract = src.split("## Abstract", 1)[1].split("## Introduction", 1)[0]
abstract = abstract.replace("*(149 words)*", "")
abstract = abstract.replace("---", "").strip()
# collapse newlines to spaces (abstract is one paragraph)
abstract = re.sub(r"\s+", " ", abstract).strip()

# 3. body = from '## Introduction' to before '## References'
body_md = "## Introduction" + src.split("## Introduction", 1)[1]
body_md = body_md.split("## References", 1)[0].rstrip()
# Also drop the trailing human-input stubs (Acknowledgements/Contributions/
# Competing interests are added via sn-jnl backmatter macros instead).
for marker in ["## Acknowledgements", "## Author contributions",
               "## Competing interests", "## Code & data availability",
               "## Supplementary Information"]:
    if marker in body_md:
        body_md = body_md.split(marker, 1)[0].rstrip()

(OUT_DIR / "_body_tmp.md").write_text(body_md)

# 4. pandoc the body, shifting heading levels down by one
subprocess.run([PANDOC, "_body_tmp.md", "-f", "markdown", "-t", "latex",
                "--wrap=preserve", "--shift-heading-level-by=-1",
                "-o", "body_sn.tex"], check=True)
(OUT_DIR / "_body_tmp.md").unlink()

# 4a. Fix combining-mark sequences pdflatex can't render (p-hat, y-hat),
#     before the standalone-symbol newunicodechar table handles the rest.
_bt = Path("body_sn.tex").read_text()
_bt = _bt.replace("p̂", r"\(\hat{p}\)").replace("ŷ", r"\(\hat{y}\)")
_bt = _bt.replace("̂", "")  # drop any stray combining circumflex
Path("body_sn.tex").write_text(_bt)

# escape % and & already handled by pandoc; nothing else needed

# 4b. Convert pandoc-escaped [Author Year] text citations -> \cite{key}
import itertools
AUTHOR_YEAR_TO_KEY = {
    "Bonafilia 2020": "bonafilia2020sen1floods11",
    "Gupta 2019": "gupta2019xbd",
    "Brown 2024": "brown2024alphaearth",
    "Jakubik 2023": "jakubik2023prithvi",
    "Xiong 2024": "xiong2024dofa",
    "Ronneberger 2015": "ronneberger2015unet",
    "Xu 2022": "xu2022causal",
    "Zhang 2025": "zhang2025stimp",
    "Schulman 2017": "schulman2017ppo",
    "Schulman 2016": "schulman2016gae",
    "Guo 2017": "guo2017calibration",
    "Platt 1999": "platt1999probabilistic",
    "Zadrozny 2002": "zadrozny2002isotonic",
    "Zadrozny \& Elkan 2002": "zadrozny2002isotonic",
    "Sener 2018": "sener2018coreset",
    "Gal 2017": "gal2017dropout",
    "Elkan 2001": "elkan2001foundations",
    "Saerens 2002": "saerens2002adjusting",
    "Lipton 2018": "lipton2018detecting",
    "Garg 2020": "garg2020unified",
}
import re as _re
btex = Path("body_sn.tex").read_text()

def _repl(m):
    inner = m.group(1)
    keys = []
    for ay, key in AUTHOR_YEAR_TO_KEY.items():
        if ay in inner and key not in keys:
            keys.append(key)
    if not keys:
        return m.group(0)  # leave untouched (e.g. [refs])
    return r"\cite{" + ",".join(keys) + "}"

# match {[} ... {]} groups that contain at least one 4-digit year
btex2 = _re.sub(r"\{\[\}([^\[\]]*?\d{4}[^\[\]]*?)\{\]\}", _repl, btex)
# also handle the bare "[Author Year]" if pandoc didn't escape (defensive)
btex2 = _re.sub(r"\[([^\[\]]*?\d{4}[^\[\]]*?)\]",
                lambda m: _repl(m) if any(ay in m.group(1) for ay in AUTHOR_YEAR_TO_KEY) else m.group(0),
                btex2)
n = btex.count("{[}") 
btex2 = _unicode_to_tex(btex2)
Path("body_sn.tex").write_text(btex2)
print(f"citation conversion: {btex2.count(chr(92)+'cite')} \\cite commands written")

print(f"title chars: {len(title)}")
print(f"abstract words: {len(abstract.split())}")
print("body_sn.tex written")

# 5. main_sn.tex
main = r"""%% Official Springer Nature / Nature Portfolio submission template (sn-jnl).
%% Reference style: sn-nature (Nature Portfolio journals).
%% Build:  pdflatex main_sn && bibtex main_sn && pdflatex main_sn && pdflatex main_sn
\documentclass[sn-nature,lineno]{sn-jnl}

\usepackage{graphicx}
\usepackage{multirow}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{amsthm}
\usepackage{mathrsfs}
\usepackage[title]{appendix}
\usepackage{xcolor}
\usepackage{textcomp}
\usepackage{manyfoot}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{calc}
\usepackage{algorithm}
\usepackage{algorithmicx}
\usepackage{algpseudocode}
\usepackage{listings}

% --- pdflatex compatibility for the pandoc-generated body ---
% (Unicode is substituted to LaTeX directly in body_sn.tex, so this
%  source compiles on any TeX Live / Overleaf with no extra packages.)
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\pandocbounded}[1]{#1}

\theoremstyle{thmstyleone}
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}[theorem]{Proposition}
\theoremstyle{thmstyletwo}
\newtheorem{example}{Example}
\newtheorem{remark}{Remark}
\theoremstyle{thmstylethree}
\newtheorem{definition}{Definition}

\raggedbottom

% Graphics path so \includegraphics{figXX.png} resolves
\graphicspath{{../outputs/figures/}}

\begin{document}

\title[Cross-disaster mapping is a calibration problem]{__TITLE__}

\author*[1]{\fnm{Qiming} \sur{Bao}}\email{qiming.bao@example.edu}
\author[2]{\fnm{Yanbing} \sur{Bai}}\email{ybbai@example.edu}

\affil*[1]{\orgdiv{[Department TBD]}, \orgname{[Institution TBD]},
  \orgaddress{\city{[City]}, \country{[Country]}}}
\affil[2]{\orgdiv{[Department TBD]}, \orgname{[Institution TBD]},
  \orgaddress{\city{[City]}, \country{[Country]}}}

\abstract{__ABSTRACT__}

\keywords{cross-disaster generalisation, calibration drift, label shift,
foundation models, active learning, flood mapping, disaster response}

\maketitle

\input{body_sn}

\backmatter

\bmhead{Supplementary information}
The within-event-protocol ablations (sample-efficiency sweep;
decision-aligned reward A/B) and the full per-fold result JSONs are
provided as Supplementary Information.

\bmhead{Acknowledgements}
[TBD]

\section*{Declarations}
\begin{itemize}
\item \textbf{Funding}: [TBD]
\item \textbf{Competing interests}: The authors declare no competing
  interests.
\item \textbf{Data availability}: All benchmark data are public
  (Sen1Floods11, xBD, HLS Burn-Scars). All intermediate results,
  figures, and code are available at
  \url{https://github.com/14H034160212/geodisaster-fm}.
\item \textbf{Code availability}: As above; a versioned release with a
  Zenodo DOI will accompany publication.
\item \textbf{Author contributions}: [TBD]
\end{itemize}

\clearpage
\input{figures_sn}

\bibliography{../refs}

\end{document}
"""
main = main.replace("__TITLE__", _unicode_to_tex(_texesc(title))).replace("__ABSTRACT__", _unicode_to_tex(_texesc(abstract)))
(OUT_DIR / "main_sn.tex").write_text(main)
print("main_sn.tex written")
