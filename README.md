# The Genesis Trilogy

**Author:** Sidong Liu, PhD — iBioStratix Ltd

A unified theoretical framework in three parts — **HAFF**, **Q-RAIF**, and **T-DOME** — investigating how geometry, observers, and irreversible cognition emerge from algebraic structure.

---

## Structure

The trilogy comprises **13 papers** across three frameworks:

### I. HAFF — Holographic Alaya-Field Framework (7 papers + postscript)

*How geometry emerges from observable algebras.*

| Paper | Title |
|-------|-------|
| A | Emergent Geometry from Coarse-Grained Observable Algebras |
| B | Accessibility, Stability, and Emergent Geometry — Conceptual Clarifications |
| C | Causation, Agency, and Existence — Structural Constraints and Interpretive Bridges |
| D | Gravitational Phenomena as Emergent Properties of Observable Algebra Selection |
| E | Measurement as Accessibility — A Structural Analysis of Observable Algebra Selection |
| F | Temporal Asymmetry as Accessibility Propagation |
| G | Structural Limits of Unification — Accessibility, Incompleteness, and the Necessity of a Final Cut |
| PS | Postscript: On the Closure of Structure |

### II. Q-RAIF — Quantum Reality-Algebra Intersection Framework (3 papers)

*Why an observer's control algebra must embed in the environment's Cl(1,3).*

| Paper | Title |
|-------|-------|
| A | The Quantum Reality-Algebra Intersection Framework |
| B | Algebraic Constraints on Observer–Environment Coupling |
| C | The Structural Closure of Reality Intersection |

### III. T-DOME — Thermodynamic Dome (3 papers)

*The irreversible logic chain: Chaos → Time → Self → Sentience.*

| Paper | Title |
|-------|-------|
| I | Memory as Thermodynamic Resource: Markov Ceilings and the Necessity of Non-Markovian Encoding |
| II | Foreground-Background Partitioning Under Finite Attention: The Emergence of Ego-Structure |
| III | Closure Under Calibration: From Ego-Structure to the Self-Monitoring Loop |

---

## Repository Layout

```
genesis-trilogy/
├── papers/
│   ├── HAFF_A/ … HAFF_G/        # Individual HAFF papers
│   ├── HAFF_Postscript/
│   ├── QRAIF_A/ … QRAIF_C/      # Individual Q-RAIF papers
│   ├── TDOME_I/ … TDOME_III/     # Individual T-DOME papers
│   │   ├── manuscript.tex
│   │   ├── manuscript.pdf
│   │   ├── figures/              # (T-DOME only)
│   │   ├── data/                 # (T-DOME only)
│   │   └── demo/                 # (T-DOME only)
├── collections/                   # Complete collected volumes
│   ├── HAFF_Complete_Collection.*
│   ├── Q_RAIF_Complete_Collection.*
│   ├── T_DOME_Complete_Collection.*
│   ├── Genesis_Trilogy.*          # All 13 papers in one volume
│   └── build_*.py                 # Automated build scripts
├── notes/                         # Companion materials
│   ├── mapping_en.md / mapping_zh.md   # Theory ↔ Yogacara mapping
│   ├── reviewer_en.md / reviewer_zh.md # Self-evaluation
│   └── tdome_companion_note_draft.md
├── arxiv.sty                      # Shared LaTeX style
├── LICENSE                        # CC BY 4.0
└── CITATION.cff
```

---

## Zenodo DOIs (Concept DOIs — always resolve to latest version)

| Item | Concept DOI |
|------|-------------|
| **Genesis Trilogy** | [10.5281/zenodo.18608021](https://doi.org/10.5281/zenodo.18608021) |
| | |
| HAFF Paper A — Emergent Geometry | [10.5281/zenodo.18361706](https://doi.org/10.5281/zenodo.18361706) |
| HAFF Paper B — Accessibility, Stability | [10.5281/zenodo.18367060](https://doi.org/10.5281/zenodo.18367060) |
| HAFF Paper C — Causation, Agency, Existence | [10.5281/zenodo.18374805](https://doi.org/10.5281/zenodo.18374805) |
| HAFF Paper D — Gravitational Phenomena | [10.5281/zenodo.18388881](https://doi.org/10.5281/zenodo.18388881) |
| HAFF Paper E — Measurement as Accessibility | [10.5281/zenodo.18400065](https://doi.org/10.5281/zenodo.18400065) |
| HAFF Paper F — Temporal Asymmetry | [10.5281/zenodo.18400425](https://doi.org/10.5281/zenodo.18400425) |
| HAFF Paper G — Structural Limits of Unification | [10.5281/zenodo.18402907](https://doi.org/10.5281/zenodo.18402907) |
| HAFF Postscript | [10.5281/zenodo.18407367](https://doi.org/10.5281/zenodo.18407367) |
| HAFF Abstract | [10.5281/zenodo.18416482](https://doi.org/10.5281/zenodo.18416482) |
| HAFF Complete Collection | [10.5281/zenodo.18452194](https://doi.org/10.5281/zenodo.18452194) |
| | |
| Q-RAIF Paper A — Lorentzian Metrics | [10.5281/zenodo.18525876](https://doi.org/10.5281/zenodo.18525876) |
| Q-RAIF Paper B — Operator Algebra Stability | [10.5281/zenodo.18525890](https://doi.org/10.5281/zenodo.18525890) |
| Q-RAIF Paper C — Realizability Bridge | [10.5281/zenodo.18528934](https://doi.org/10.5281/zenodo.18528934) |
| Q-RAIF Complete Collection | [10.5281/zenodo.18548704](https://doi.org/10.5281/zenodo.18548704) |
| | |
| T-DOME Paper I — Non-Markovian Memory | [10.5281/zenodo.18574342](https://doi.org/10.5281/zenodo.18574342) |
| T-DOME Paper II — Reference Frame SSB | [10.5281/zenodo.18579703](https://doi.org/10.5281/zenodo.18579703) |
| T-DOME Paper III — Self-Referential Calibration | [10.5281/zenodo.18591771](https://doi.org/10.5281/zenodo.18591771) |
| T-DOME Complete Collection | [10.5281/zenodo.18593180](https://doi.org/10.5281/zenodo.18593180) |

---

## Reproducing Numerical Demonstrations

T-DOME Papers I–III include numerical demonstrations. Each demo script is self-contained and requires only **NumPy** and **Matplotlib**:

```bash
# Paper I — Markov ceiling demo
cd papers/TDOME_I/demo
python paper1_markov_ceiling_demo.py

# Paper II — k* staircase demo
cd papers/TDOME_II/demo
python paper2_kstar_scaling_demo.py

# Paper III — phase diagram, survival, ego, loop demos
cd papers/TDOME_III/demo
python tdome_demo.py
```

Each script produces deterministic output (fixed seeds) and generates PDF figures with confidence intervals.

---

## Building Collections

The collected volumes can be rebuilt from individual papers:

```bash
cd collections
python build_tdome_collection.py    # → T_DOME_Complete_Collection.tex
python build_genesis_trilogy.py     # → Genesis_Trilogy.tex
# Then compile with pdflatex (3 passes for cross-references)
```

---

## Archive / Long-term Preservation

| Layer | Platform | Identifier |
|-------|----------|------------|
| Academic citation | **Zenodo** | [10.5281/zenodo.18608021](https://doi.org/10.5281/zenodo.18608021) |
| Canonical source | **GitHub** | [github.com/sidongliu/genesis-trilogy](https://github.com/sidongliu/genesis-trilogy) |
| Software Heritage | **SWH** | [`swh:1:snp:76ebc4c8e9cd85e7f9e97a0b7c0237ca17696f83`](https://archive.softwareheritage.org/swh:1:snp:76ebc4c8e9cd85e7f9e97a0b7c0237ca17696f83) |

This repository contains the LaTeX sources, numerical demonstrations, and build scripts needed to reproduce every result in the Genesis Trilogy. The Zenodo DOIs serve as the academic citation entry point; the GitHub repository is the canonical, versioned source; and the Software Heritage archive provides content-addressed, platform-independent long-term preservation.

---

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
