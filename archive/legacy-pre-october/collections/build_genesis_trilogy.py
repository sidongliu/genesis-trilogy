#!/usr/bin/env python3
"""Build the Genesis Trilogy unified volume from three complete collections.

Combines HAFF, Q-RAIF, and T-DOME complete collections into a single
book-class LaTeX document with:
  - \part{} structure (one per pillar)
  - prefixed labels to avoid conflicts
  - unified deduplicated bibliography
  - trilogy-level front matter
"""

import re
import os
import shutil

BASE = "/Users/sidongliu/Documents/Sidong/Genesis Trilogy"

def read_file(path):
    with open(path) as f:
        return f.read()

def extract_main_body(tex, start_marker=r'\mainmatter', end_markers=None):
    """Extract content between start_marker and the first end_marker found."""
    if end_markers is None:
        end_markers = [r'\backmatter', r'\begin{thebibliography}']

    idx = tex.find(start_marker)
    if idx >= 0:
        idx = tex.index('\n', idx) + 1
    else:
        idx = 0

    end_idx = len(tex)
    for marker in end_markers:
        m = tex.find(marker)
        if m >= 0 and m < end_idx:
            end_idx = m

    return tex[idx:end_idx].strip()

def extract_acknowledgments(tex):
    """Extract acknowledgments section from HAFF collection."""
    m = re.search(
        r'(\\chapter\*\{Acknowledgments\}.*?)(?=\\backmatter|\\begin\{thebibliography\})',
        tex, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None

def extract_bib_entries(tex):
    """Extract individual \\bibitem entries from a .tex file."""
    m = re.search(
        r'\\begin\{thebibliography\}\{99\}\s*(.*?)\\end\{thebibliography\}',
        tex, re.DOTALL)
    if not m:
        return {}
    content = m.group(1)
    entries = {}
    parts = re.split(r'(\\bibitem\{[^}]+\})', content)
    for i in range(1, len(parts), 2):
        key_match = re.match(r'\\bibitem\{([^}]+)\}', parts[i])
        if key_match and i+1 < len(parts):
            key = key_match.group(1)
            entry_text = parts[i] + parts[i+1].rstrip()
            entries[key] = entry_text
    return entries

def prefix_refs(content, prefix):
    """Prefix all \\label, \\ref, \\eqref with a collection-specific prefix."""
    content = re.sub(
        r'\\label\{([^}]+)\}',
        lambda m: r'\label{' + prefix + m.group(1) + '}',
        content)
    content = re.sub(
        r'\\ref\{([^}]+)\}',
        lambda m: r'\ref{' + prefix + m.group(1) + '}',
        content)
    content = re.sub(
        r'\\eqref\{([^}]+)\}',
        lambda m: r'\eqref{' + prefix + m.group(1) + '}',
        content)
    return content

def remove_acknowledgments_from_body(body):
    """Remove acknowledgments chapter from HAFF body (will be placed in back matter)."""
    # Remove from \chapter*{Acknowledgments} to end
    idx = body.find(r'\chapter*{Acknowledgments}')
    if idx >= 0:
        return body[:idx].rstrip()
    return body

def fix_double_encoded_utf8(text):
    """Fix double-encoded UTF-8 in HAFF source.

    The HAFF file has corrupted diacritics where proper UTF-8 bytes were
    interpreted as Latin-1 and re-encoded to UTF-8.  We replace these
    with proper LaTeX diacritics.
    """
    # Try to fix via encode/decode round-trip
    try:
        fixed = text.encode('utf-8').decode('utf-8')
    except:
        fixed = text

    # Direct replacements for known corrupted strings
    replacements = [
        # Yogācāra
        ('Yog\u00c4\u0081c\u00c4\u0081ra', r'Yog\=ac\=ara'),
        # Mahāyānasaṃgraha
        ('Asa\u1e45ga', r'Asa\.{n}ga'),
        ('Mah\u00c4\u0081y\u00c4\u0081nasa\u1e43graha', r'Mah\=ay\=anasa\d{m}graha'),
        # Nāgārjuna
        ('N\u00c4\u0081g\u00c4\u0081rjuna', r'N\=ag\=arjuna'),
        # Mūlamadhyamakakārikā
        ('M\u00c5\u00ablamadhyamakak\u00c4\u0081rik\u00c4\u0081', r'M\=ulamadhyamakak\=arik\=a'),
        # Sutta Nipāta
        ('Nip\u00c4\u0081ta', r'Nip\=ata'),
        # É. Lamotte
        ('\u00c3\u0089.', "\\'E."),
        # vijñaptimātratā
        ('vij\u00c3\u00b1aptim\u00c4\u0081trat\u00c4\u0081', "vij\\~naptim\\=atrat\\=a"),
        # Triṃśikā
        ('Tri\u1e43\u015bik\u00c4\u0081', "Tri\\d{m}\\'sik\\=a"),
    ]

    for old, new in replacements:
        fixed = fixed.replace(old, new)

    # Also fix any remaining double-encoded ā (Ä + \x81 pattern)
    # by reading as bytes and doing byte-level fix
    try:
        raw = fixed.encode('utf-8')
        # C3 84 C2 81 → C4 81 (ā → \=a in LaTeX, but let's just fix to proper UTF-8 first)
        # Actually, replace known byte patterns with LaTeX macros
        raw = raw.replace(b'\xc3\x84\xc2\x81', b'\\=a')
        raw = raw.replace(b'\xc3\x85\xc2\xab', b'\\=u')
        raw = raw.replace(b'\xc3\x83\xc2\xb1', b'\\~n')
        # Fix Devanagari sequences (remove garbled Devanagari in Skt. parentheticals)
        import re as _re
        # Match (Skt. <garbled bytes>, and replace with just (Skt.,
        raw = _re.sub(
            rb'\(Skt\.\s*(?:[\xc0-\xff][\x80-\xbf]*|[\xe2][\x80-\xbf][\x80-\xbf])+,',
            b'(Skt.,',
            raw)
        # Also match garbled Devanagari not in (Skt. ...) — e.g. inline in text
        # Pattern: sequences of multi-byte chars from Devanagari block (à¤, à¥)
        raw = _re.sub(
            rb'(?:[\xc3][\xa0][\xc2][\xa4-\xa5][\xc2\xc3\xc5\xe2][\x80-\xbf](?:[\x80-\xbf])?)+',
            b'',
            raw)
        fixed = raw.decode('utf-8', errors='replace')
    except:
        pass

    return fixed

# ---- Read collections ----
haff_tex = read_file(os.path.join(BASE, "HAFF", "Complete collection",
                                   "HAFF_Complete_Collection.tex"))
haff_tex = fix_double_encoded_utf8(haff_tex)
qraif_tex = read_file(os.path.join(BASE, "Q-RAIF",
                                    "Q_RAIF_Complete_Collection.tex"))
tdome_tex = read_file(os.path.join(BASE, "T-Dome",
                                    "T_DOME_Complete_Collection.tex"))

# ---- Extract bodies ----
haff_body = extract_main_body(haff_tex)
qraif_body = extract_main_body(qraif_tex)
tdome_body = extract_main_body(tdome_tex, end_markers=[
    r'% ============================================================================' + '\n'
    + r'% UNIFIED BIBLIOGRAPHY',
    r'\begin{thebibliography}'])

# ---- Extract and remove HAFF acknowledgments ----
haff_ack = extract_acknowledgments(haff_tex)
haff_body = remove_acknowledgments_from_body(haff_body)

# ---- Prefix labels ----
haff_body = prefix_refs(haff_body, "H-")
qraif_body = prefix_refs(qraif_body, "Q-")
tdome_body = prefix_refs(tdome_body, "T-")

# ---- Merge bibliographies ----
bib_haff = extract_bib_entries(haff_tex)
bib_qraif = extract_bib_entries(qraif_tex)
bib_tdome = extract_bib_entries(tdome_tex)

all_bib = {}
seen_order = []
for entries in [bib_haff, bib_qraif, bib_tdome]:
    for key, text in entries.items():
        if key not in all_bib:
            all_bib[key] = text
            seen_order.append(key)

# Sort: Liu papers last, others alphabetically
liu_keys = [k for k in seen_order if k.startswith('Liu2026')]
other_keys = [k for k in seen_order if not k.startswith('Liu2026')]
other_keys.sort()
liu_keys.sort()
ordered_keys = other_keys + liu_keys

unified_bib = '\n\n'.join(all_bib[k] for k in ordered_keys)

# ---- Build the document ----
PREAMBLE = r"""% ============================================================================
% The Genesis Trilogy
% Complete Collected Volume
% ============================================================================

\documentclass[12pt,a4paper,openany]{book}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,amsfonts,amsthm}
\usepackage{physics}
\usepackage{graphicx}
\usepackage[hidelinks]{hyperref}
\usepackage{tikz}
\usepackage{geometry}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{array}
\usepackage{booktabs}
\usepackage{tcolorbox}
\usepackage{xcolor}
\usepackage{caption}
\usepackage{float}

\usetikzlibrary{shapes,arrows}

\geometry{a4paper, margin=1in, headheight=28pt}

\newtheorem{theorem}{Theorem}[chapter]
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{remark}[theorem]{Remark}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{conjecture}[theorem]{Conjecture}
\newtheorem{constraint}[theorem]{Constraint}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[LE,RO]{\thepage}
\fancyhead[RE]{\textit{The Genesis Trilogy}}
\fancyhead[LO]{\textit{\leftmark}}
\renewcommand{\headrulewidth}{0.4pt}

\title{
  \vspace{-2cm}
  {\Huge\textbf{The Genesis Trilogy}}\\[1cm]
  {\Large\textit{Emergent Geometry, Algebraic Persistence,\\
  and the Architecture of the Observer}}\\[2cm]
  {\large Complete Collected Volume}\\[0.5cm]
  {\normalsize HAFF \textperiodcentered{} Q-RAIF \textperiodcentered{} T-DOME}
}

\author{
  \textbf{Sidong Liu, PhD}\\[0.5em]
  iBioStratix Ltd\\[0.3em]
  \texttt{sidongliu@hotmail.com}
}

\date{February 2026}

\begin{document}

\frontmatter
\maketitle

\newpage
\thispagestyle{empty}
\vspace*{\fill}
\begin{center}
\textbf{Publication Record}\\[2em]
\begin{tabular}{lll}
\multicolumn{3}{l}{\textbf{Part~I --- HAFF: The Holographic Alaya-Field Framework}} \\[0.5em]
Paper A & DOI: 10.5281/zenodo.18361707 & Emergent Geometry \\
Paper B & DOI: 10.5281/zenodo.18367061 & Accessibility \& Stability \\
Essay C & DOI: 10.5281/zenodo.18374806 & Causation \& Agency \\
Paper D & DOI: 10.5281/zenodo.18388882 & Gravitational Phenomena \\
Paper E & DOI: 10.5281/zenodo.18400066 & Measurement \\
Paper F & DOI: 10.5281/zenodo.18400426 & Temporal Asymmetry \\
Paper G & DOI: 10.5281/zenodo.18402908 & Structural Limits \\
Postscript & DOI: 10.5281/zenodo.18407368 & Closure of Structure \\[1em]
\multicolumn{3}{l}{\textbf{Part~II --- Q-RAIF: Quantum Reference Algebra for Information Flow}} \\[0.5em]
Paper A & DOI: 10.5281/zenodo.18525877 & Lorentzian Metrics \\
Paper B & DOI: 10.5281/zenodo.18525891 & Thermodynamic Stability \\
Paper C & DOI: 10.5281/zenodo.18528935 & Realizability Bridge \\[1em]
\multicolumn{3}{l}{\textbf{Part~III --- T-DOME: Thermodynamic Dynamics of Observer-Memory Entanglement}} \\[0.5em]
Paper I & DOI: 10.5281/zenodo.18574342 & Non-Markovian Memory \\
Paper II & DOI: 10.5281/zenodo.18579703 & Symmetry Breaking \\
Paper III & DOI: 10.5281/zenodo.18591771 & Self-Referential Calibration \\
\end{tabular}
\\[3em]
\textit{This collected volume compiles previously published works\\
for archival and reference purposes.}
\\[2em]
\copyright{} 2026 Sidong Liu. All rights reserved.
\end{center}
\vspace*{\fill}

\chapter*{Abstract}
\addcontentsline{toc}{chapter}{Abstract}

This volume collects the thirteen papers and postscript comprising the Genesis Trilogy---a programme that investigates, from first principles, the minimal structural conditions under which physical descriptions, persistent agents, and self-referential observers can arise within quantum theory.

The programme is organised in three parts:

\begin{enumerate}
\item \textbf{Part~I: HAFF} (Papers A--G, Essay~C, Postscript).
  The Holographic Alaya-Field Framework treats the tensor factorisation of Hilbert space as a derived, rather than assumed, structure.  Starting from a single global quantum state and coarse-grained observable algebras, it demonstrates that:
  (a)~emergent geometry, locality, and topology follow from algebraic accessibility;
  (b)~gravitational phenomena arise as evolution \emph{of} the accessible algebra;
  (c)~measurement is selection \emph{within} the accessible algebra;
  (d)~temporal asymmetry is a propagation property of informational redundancy; and
  (e)~the framework is structurally incomplete---it cannot self-ground its own starting point.
  This incompleteness motivates Parts~II and~III.

\item \textbf{Part~II: Q-RAIF} (Papers A--C).
  The Quantum Reference Algebra for Information Flow asks what algebraic structure a persistent subsystem must possess if it is to survive within the geometry established by Part~I.  The three papers show that:
  (a)~boundary algebras compatible with emergent Lorentzian geometry must respect three constraints (associativity, metric compatibility, indefinite signature), selecting Clifford algebra $Cl(1,3)$;
  (b)~thermodynamic stability of a non-equilibrium steady state requires a Clifford control algebra $Cl(V,q)$ for Lyapunov-stable channel discrimination; and
  (c)~realizability forces the internal algebra to embed in the environmental algebra, $Cl(V,q) \hookrightarrow Cl(1,3)$---algebraic natural selection.

\item \textbf{Part~III: T-DOME} (Papers I--III).
  The Thermodynamic Dynamics of Observer-Memory Entanglement characterises the internal architecture that makes persistence possible.  The three papers trace an irreversible logic chain:
  (a)~\emph{Memory}: Markovian dynamics impose a survival ceiling; non-Markovian memory (temporal accumulation) is necessary, but creates a memory catastrophe under finite resources;
  (b)~\emph{Ego}: bounded computation forces spontaneous symmetry breaking of the agent's reference frame, enabling tractable processing but introducing systematic bias; and
  (c)~\emph{Loop}: the agent's own prediction-residual stream carries a Fisher-information signal that enables self-referential calibration, with an explicit thermodynamic cost.
\end{enumerate}

Together, the three parts establish a \textbf{Four-Part Structure Proposition}: within the class of agents satisfying the standing assumptions, a sufficient architecture for persistent far-from-equilibrium existence comprises (1)~an external observable geometry, (2)~an internal control algebra, (3)~a self-monitoring Lyapunov function, and (4)~biased non-Markovian memory.

Each layer is the necessary resolution of the previous layer's survival crisis, and simultaneously the source of the next crisis.  The programme terminates when the self-referential loop closes: beyond that point, the framework's own structural incompleteness---identified in HAFF Paper~G---precludes further internal completion.

\bigskip
\noindent\textbf{Keywords}: emergent geometry, accessible algebras, coarse-graining, Clifford algebra, algebraic natural selection, non-Markovian dynamics, spontaneous symmetry breaking, Fisher information, self-referential calibration, Lyapunov stability, thermodynamic cost

\tableofcontents

\mainmatter

"""

PART_HAFF = r"""
% ============================================================================
% PART I: HAFF
% ============================================================================
\part{HAFF: The Holographic Alaya-Field Framework}
\label{part:HAFF}

"""

PART_QRAIF = r"""
% ============================================================================
% PART II: Q-RAIF
% ============================================================================
\part{Q-RAIF: Quantum Reference Algebra for Information Flow}
\label{part:QRAIF}

"""

PART_TDOME = r"""
% ============================================================================
% PART III: T-DOME
% ============================================================================
\part{T-DOME: Thermodynamic Dynamics of Observer-Memory Entanglement}
\label{part:TDOME}

"""

BACKMATTER = r"""

% ============================================================================
% BACK MATTER
% ============================================================================
\backmatter

"""

# Assemble
output = PREAMBLE
output += PART_HAFF + haff_body + '\n'
output += PART_QRAIF + qraif_body + '\n'
output += PART_TDOME + tdome_body + '\n'
output += BACKMATTER

# Acknowledgments
if haff_ack:
    output += haff_ack + '\n\n'

# Bibliography
output += '% ============================================================================\n'
output += '% UNIFIED BIBLIOGRAPHY\n'
output += '% ============================================================================\n'
output += '\\begin{thebibliography}{99}\n\n'
output += unified_bib + '\n\n'
output += '\\end{thebibliography}\n\n'
output += '\\end{document}\n'

# Write output
outpath = os.path.join(BASE, "Genesis_Trilogy.tex")
with open(outpath, 'w') as f:
    f.write(output)

print(f"Written to {outpath}")
print(f"Total lines: {output.count(chr(10))}")
print(f"Bibliography entries: {len(ordered_keys)}")

# Copy figures
for fig in ['fig_survival.pdf', 'fig_paper1_markov_ceiling.pdf',
           'fig_ego.pdf', 'fig_paper2_kstar_scaling.pdf',
           'fig_delusion_trap.pdf', 'fig_phase_with_boundary.pdf',
           'fig_alpha_opt.pdf']:
    src = os.path.join(BASE, "T-Dome", fig)
    dst = os.path.join(BASE, fig)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied {fig}")
