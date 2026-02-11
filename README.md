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

| Item | DOI |
|------|-----|
| T-DOME Paper I | [10.5281/zenodo.18574342](https://doi.org/10.5281/zenodo.18574342) |
| T-DOME Paper II | [10.5281/zenodo.18579703](https://doi.org/10.5281/zenodo.18579703) |
| T-DOME Paper III | [10.5281/zenodo.18591771](https://doi.org/10.5281/zenodo.18591771) |
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

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
