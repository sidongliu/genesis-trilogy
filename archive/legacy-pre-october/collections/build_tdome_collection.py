#!/usr/bin/env python3
"""Build T-DOME Complete Collection from the three standalone papers.

Follows the Q-RAIF Complete Collection template:
  - book class, a4paper, chapter per paper
  - prefixed labels to avoid conflicts
  - unified bibliography
  - front matter with publication record and abstract
"""

import re
import os

BASE = "/Users/sidongliu/Documents/Sidong/Genesis Trilogy/T-Dome"

def read_file(path):
    with open(path) as f:
        return f.read()

def extract_body(tex):
    """Extract content between \\maketitle and \\begin{thebibliography}."""
    # Find start: after \maketitle
    idx = tex.find('\\maketitle')
    if idx >= 0:
        idx = tex.index('\n', idx) + 1
    else:
        idx = 0
    # Find end: before bibliography
    bib_idx = tex.find('\\begin{thebibliography}')
    if bib_idx < 0:
        bib_idx = tex.find('\\end{document}')
    body = tex[idx:bib_idx].strip()
    return body

def extract_bib_entries(tex):
    """Extract individual \\bibitem entries from a .tex file."""
    m = re.search(
        r'\\begin\{thebibliography\}\{99\}\s*(.*?)\\end\{thebibliography\}',
        tex, re.DOTALL)
    if not m:
        return {}
    content = m.group(1)
    entries = {}
    # Split by \bibitem
    parts = re.split(r'(\\bibitem\{[^}]+\})', content)
    for i in range(1, len(parts), 2):
        key_match = re.match(r'\\bibitem\{([^}]+)\}', parts[i])
        if key_match and i+1 < len(parts):
            key = key_match.group(1)
            entry_text = parts[i] + parts[i+1].rstrip()
            entries[key] = entry_text
    return entries

def prefix_refs(content, prefix):
    """Prefix all \\label, \\ref, \\eqref with a paper-specific prefix."""
    content = re.sub(
        r'\\label\{([^}]+)\}',
        lambda m: '\\label{' + prefix + m.group(1) + '}',
        content)
    content = re.sub(
        r'\\ref\{([^}]+)\}',
        lambda m: '\\ref{' + prefix + m.group(1) + '}',
        content)
    content = re.sub(
        r'\\eqref\{([^}]+)\}',
        lambda m: '\\eqref{' + prefix + m.group(1) + '}',
        content)
    return content

def convert_abstract(content):
    """Convert \\begin{abstract}...\\end{abstract} to \\section*{Abstract}..."""
    # Handle the case with comment line before abstract
    content = re.sub(
        r'% =+\s*\n\\begin\{abstract\}',
        r'\\section*{Abstract}',
        content)
    # Fallback: plain \begin{abstract}
    content = content.replace('\\begin{abstract}', '\\section*{Abstract}')
    content = content.replace('\\end{abstract}', '')
    return content

# ---- Read and process each paper ----
paper1 = read_file(os.path.join(BASE, "T_DOME_paper_I_v2.tex"))
paper2 = read_file(os.path.join(BASE, "T_DOME_paper_II_v2.tex"))
paper3 = read_file(os.path.join(BASE, "T_DOME_paper_III_v2.tex"))

body1 = extract_body(paper1)
body2 = extract_body(paper2)
body3 = extract_body(paper3)

# Convert abstract environments
body1 = convert_abstract(body1)
body2 = convert_abstract(body2)
body3 = convert_abstract(body3)

# Prefix labels
body1 = prefix_refs(body1, "I-")
body2 = prefix_refs(body2, "II-")
body3 = prefix_refs(body3, "III-")

# ---- Merge bibliographies (deduplicate by key, preserve order of first occurrence) ----
bib1 = extract_bib_entries(paper1)
bib2 = extract_bib_entries(paper2)
bib3 = extract_bib_entries(paper3)

# Merge: first occurrence wins
all_bib = {}
seen_order = []
for entries in [bib1, bib2, bib3]:
    for key, text in entries.items():
        if key not in all_bib:
            all_bib[key] = text
            seen_order.append(key)

# Group bibliography: Liu papers first, then others alphabetically
liu_keys = [k for k in seen_order if k.startswith('Liu2026')]
other_keys = [k for k in seen_order if not k.startswith('Liu2026')]
other_keys.sort()
liu_keys.sort()
ordered_keys = other_keys + liu_keys

unified_bib = '\n\n'.join(all_bib[k] for k in ordered_keys)

# ---- Add Paper III DOI to its own citation if missing ----
# (Paper III doesn't cite itself, but Papers I and II cite each other)
# We need to ADD a bibitem for Paper III for completeness
if 'Liu2026TDOME_III' not in all_bib:
    unified_bib += """

\\bibitem{Liu2026TDOME_III}
S.~Liu,
\\emph{Fisher Information Geometry and the Thermodynamic Cost
of Self-Referential Calibration},
Zenodo (2026), DOI: 10.5281/zenodo.18591771."""

# ---- Build the collection ----
PREAMBLE = r"""% ============================================================================
% T-DOME: Thermodynamic Dynamics of Observer-Memory Entanglement
% Complete Collected Volume
% ============================================================================

\documentclass[12pt,a4paper,openany]{book}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,amsfonts,amsthm}
\usepackage{physics}
\usepackage{graphicx}
\usepackage[hidelinks]{hyperref}
\usepackage{geometry}
\usepackage{fancyhdr}
\usepackage{array}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{caption}
\usepackage{float}

\geometry{a4paper, margin=1in, headheight=14pt}

\newtheorem{theorem}{Theorem}[chapter]
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{remark}[theorem]{Remark}
\newtheorem{corollary}[theorem]{Corollary}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[LE,RO]{\thepage}
\fancyhead[RE]{\textit{Thermodynamic Dynamics of Observer-Memory Entanglement}}
\fancyhead[LO]{\textit{\leftmark}}
\renewcommand{\headrulewidth}{0.4pt}

\title{
  \vspace{-2cm}
  {\Huge\textbf{Thermodynamic Dynamics of\\Observer-Memory Entanglement}}\\[1cm]
  {\Large\textit{Memory, Ego, and Self-Referential Calibration\\
  in Persistent Far-from-Equilibrium Systems}}\\[2cm]
  {\large Complete Collected Volume}\\[0.5cm]
  {\normalsize Papers I, II, and III}
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
\begin{tabular}{ll}
Paper I & DOI: 10.5281/zenodo.18574342 \\
Paper II & DOI: 10.5281/zenodo.18579703 \\
Paper III & DOI: 10.5281/zenodo.18591771 \\
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

This volume develops the theory of persistent agency under thermodynamic constraints.  The central question is: what minimal internal structure must an open quantum system possess in order to maintain itself far from equilibrium in a changing environment?

The answer is organised as an irreversible logical chain across three papers:

\begin{enumerate}
\item \textbf{Paper~I (Memory):} We prove a Markovian Ceiling---under open-loop Markovian dynamics the survival functional satisfies $\mathcal{S} \leq 0$---and show that non-Markovian memory (system--environment correlations carried forward in time) is necessary for sustained far-from-equilibrium persistence.  Memory, however, creates the Memory Catastrophe: unbounded history under finite resources leads to thermodynamic collapse.

\item \textbf{Paper~II (Ego):} We prove a Computational Ceiling---symmetric processing of a Clifford algebra $Cl(V,q)$ leads to computational paralysis at a finite critical time---and show that the resolution requires spontaneous symmetry breaking of the agent's internal reference frame.  The broken phase (the ``ego'') introduces four systematic bias terms and, under environmental drift, leads to the Delusion Trap: an exponential divergence of prediction error invisible from within the agent's own frame.

\item \textbf{Paper~III (Loop):} We show that the Fisher information of the agent's own prediction-residual stream provides a detectable drift signal (growing quadratically with accumulated drift), derive a Self-Referential Cram\'{e}r--Rao bound on drift estimation, and establish a Lyapunov tracking bound for the natural-gradient calibration loop.  The thermodynamic cost of the complete self-referential architecture is calculated explicitly.
\end{enumerate}

Each resolution creates the precondition for the next crisis: memory enables overload, compression enables bias, and bias demands calibration.  Together, the three papers establish a Four-Part Structure Proposition: within the class of agents satisfying the standing assumptions, a sufficient architecture for persistence comprises (1)~an external observable geometry, (2)~an internal control algebra, (3)~a self-monitoring Lyapunov function, and (4)~biased non-Markovian memory.

The framework builds on the Holographic Alaya-Field Framework (HAFF), which establishes that geometry emerges from observable algebras, and Q-RAIF, which establishes the algebraic constraints on persistent subsystems.  T-DOME completes the programme by characterising the \emph{observer}: the internal architecture that makes persistence possible.

\bigskip
\noindent\textbf{Keywords}: non-Markovian dynamics, open quantum systems, memory kernel, spontaneous symmetry breaking, bounded rationality, Fisher information, information geometry, self-referential calibration, Lyapunov stability, thermodynamic cost

\tableofcontents

\mainmatter

"""

CHAPTER1_HEADER = r"""
% ============================================================================
% PAPER I
% ============================================================================
\chapter{Non-Markovian Memory and the Thermodynamic Necessity of Temporal Accumulation}
\label{chap:paperI}

\begin{center}
\textit{Paper I --- ``The Seed''}\\[0.5em]
Originally published: Zenodo, DOI: 10.5281/zenodo.18574342
\end{center}

\bigskip

"""

CHAPTER2_HEADER = r"""
% ============================================================================
% PAPER II
% ============================================================================
\chapter{Spontaneous Symmetry Breaking of Reference Frames as a Computational Cost Minimization Strategy}
\label{chap:paperII}

\begin{center}
\textit{Paper II --- ``The Ego''}\\[0.5em]
Originally published: Zenodo, DOI: 10.5281/zenodo.18579703
\end{center}

\bigskip

"""

CHAPTER3_HEADER = r"""
% ============================================================================
% PAPER III
% ============================================================================
\chapter{Fisher Information Geometry and the Thermodynamic Cost of Self-Referential Calibration}
\label{chap:paperIII}

\begin{center}
\textit{Paper III --- ``The Loop''}\\[0.5em]
Originally published: Zenodo, DOI: 10.5281/zenodo.18591771
\end{center}

\bigskip

"""

BACKMATTER = r"""

% ============================================================================
% UNIFIED BIBLIOGRAPHY
% ============================================================================
"""

# Assemble
output = PREAMBLE
output += CHAPTER1_HEADER + body1 + '\n'
output += CHAPTER2_HEADER + body2 + '\n'
output += CHAPTER3_HEADER + body3 + '\n'
output += BACKMATTER
output += '\\begin{thebibliography}{99}\n\n'
output += unified_bib + '\n\n'
output += '\\end{thebibliography}\n\n'
output += '\\end{document}\n'

# Write output
outpath = os.path.join(BASE, "T_DOME_Complete_Collection.tex")
with open(outpath, 'w') as f:
    f.write(output)

print(f"Written to {outpath}")
print(f"Total lines: {output.count(chr(10))}")
print(f"Bibliography entries: {len(ordered_keys) + (1 if 'Liu2026TDOME_III' not in all_bib else 0)}")
