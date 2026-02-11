# T-DOME Companion Note (Draft) — Papers I–II Minimal Demonstrations
*Status: draft spec + expected signatures (no figures yet). Purpose is to “park” the plan until the plots reach Paper III-level hardness.*

---

## 0) Purpose and scope

This companion note is intended to **supplement (not modify)** T-DOME Papers I–II with **minimal computational demonstrations**.

- **Does show (when implemented):** existence and reproducibility of the qualitative mechanisms claimed in Papers I–II under minimal assumptions.
- **Does not show:** universality across environments/architectures, nor tightness of analytic bounds.
- **Why not publish yet:** until the signatures are produced at “final-review hardness” (seeds + CI + boundary curves), uploading this note may dilute the overall credibility.

**Hardness target (Paper III benchmark):**
- At least **one quantitative boundary/curve** per paper (like \(\alpha_{\mathrm{opt}}(\Lambda)\) in Paper III),
- **6–10 seeds** with **CI**,
- A short parameter table,
- A strict “does show / does not show” list.

---

## 1) Paper I (The Seed): Memory surpasses a Markov ceiling

### 1.1 Claim (Paper I)
In partially observed environments, any agent constrained to a finite-order Markov representation of observations exhibits a **performance ceiling** as the relevant temporal dependence exceeds the Markov order. Introducing an explicit memory state (belief state / recurrent state) yields a systematic improvement.

### 1.2 Minimal environment (HMM / POMDP with aliasing)
Use a two-hidden-state HMM with aliased observations so that optimal action requires temporal integration.

- Hidden state: \(s_t \in \{0,1\}\), with transition
  - \(p(s_{t+1}=s_t)=1-\epsilon\),
  - \(p(s_{t+1}\neq s_t)=\epsilon\).
  - Correlation length proxy: \(\ell \sim 1/\epsilon\).

- Observation: \(o_t \in \{A,B\}\) with **aliasing** (non-invertible), e.g.
  - Max aliasing: \(p(o_t=A\mid s_t=0)=p(o_t=A\mid s_t=1)=1/2\),
  - Or slight asymmetry for controllability.

- Reward: depends on hidden state and action, e.g.
  - \(r_t=\mathbb{1}[a_t=s_t]\) (or signed reward).

This is a minimal setting where \(o_t\) alone is insufficient.

### 1.3 Agents (three baselines)
Compare three agents:

1) **Markov-0:** policy depends only on current observation \(o_t\).  
2) **Markov-k:** policy depends on window \((o_{t-k+1},\dots,o_t)\).  
3) **Memory:** maintains internal belief \(\hat b_t \approx p(s_t=1\mid o_{1:t})\) via Bayes filter (or a small RNN/GRU).

### 1.4 Metrics
- Primary: long-horizon average reward \(\bar R(T)\).
- Secondary: predictive log-loss for \(s_t\) (or action regret).

### 1.5 Expected signatures (what should be plotted)

**Signature I-A (Markov ceiling boundary):**
- For small \(\epsilon\) (long correlation), Markov-k saturates for fixed \(k\), while memory continues to improve.
- Plot: \(\bar R\) vs \(\ell\sim 1/\epsilon\) (or vs \(k\)) with CI.
- Extract a boundary like \(\ell_c(k)\) where Markov-k drops below memory by a fixed margin.

**Signature I-B (gap scaling):**
- \(\Delta\bar R(\ell)=\bar R_{\mathrm{mem}}-\bar R_{\mathrm{Markov-}k}\) increases beyond the ceiling regime.

### 1.6 “Final-review hardness” checklist (Paper I)
- One figure: **ceiling boundary curve** (preferably \(\ell_c(k)\) or \(\Delta\bar R(\ell)\) with CI).
- Seeds: ≥ 6 (prefer 10).
- Parameter table (5–8 rows): \(T\), \(\epsilon\) grid, \(k\) set, reward definition, emission probabilities.
- “Does show / does not show” bullet list (see §3).

### 1.7 Reproducibility packaging (recommended)
- Script: `paper1_markov_ceiling_demo.py`
- Output: `fig_paper1_markov_ceiling.pdf`
- Defaults:
  - \(T=10^5\), seeds ≥ 10
  - \(k\in\{1,2,4,8\}\)
  - \(\epsilon\in[10^{-3},10^{-1}]\) (log-grid)

---

## 2) Paper II (The Ego): Budget forces symmetry breaking (gauge fixing) and induces bias

### 2.1 Claims (Paper II)
Under a processing/representation budget, a symmetric “equal-fidelity over all degrees of freedom” regime is not sustainable. The system must select a privileged subspace/frame (foreground), producing unavoidable bias and (under drift) the possibility of delusion.

This companion note focuses on **upstream validation**:
- budget \(\Rightarrow\) emergent selection/SSB,
- scaling \(k^* \sim \lfloor \mathcal C_{\mathrm{budget}}/h_\mu \rfloor\) (staircase).

### 2.2 Minimal environment (multi-channel linear prediction)
Let \(x_t\sim\mathcal N(0,I_D)\) and
\[
y_t = \mathbf w^*(t)^\top x_t + \xi_t,
\]
with drift/noise controlling an entropy-rate surrogate.

Define a **budget constraint**: only \(k\) coordinates (or a rank-\(k\) projection) can be processed/updated per step.

### 2.3 Agents (minimal)
1) **Symmetric baseline:** spreads updates uniformly across \(D\) coordinates (or uses full \(D\) but with reduced per-coordinate compute).  
2) **Budgeted selector (SSB):** selects a foreground subset/projection of size \(k\) and updates only there. Selection is survival-weighted or predictive-gain-weighted.

### 2.4 Metrics
#### (A) Emergent symmetry breaking via attention entropy
Let \(p_i\) be normalized update/energy share for coordinate \(i\). Define
\[
H_{\mathrm{attn}}=-\sum_{i=1}^D p_i\log p_i.
\]
**Signature II-A:** as budget tightens, \(H_{\mathrm{attn}}\) collapses from \(\log D\) (symmetric) toward \(\log k\) (selected).

**Figure II-1:** \(H_{\mathrm{attn}}\) vs budget ratio (or surrogate \(k/D\)), with CI.

#### (B) Scaling law for empirical \(k^*\)
Scan \(\mathcal C_{\mathrm{budget}}\) and an \(h_\mu\) surrogate (via drift/noise). Find empirical optimal \(k\) under survival-weighted objective.

**Signature II-B:** \(k^*\) tracks \(\lfloor \mathcal C_{\mathrm{budget}}/h_\mu \rfloor\) up to integer effects and finite-time constants.

**Figure II-2:** empirical \(k^*\) vs \(\mathcal C_{\mathrm{budget}}/h_\mu\) (staircase), with CI.

### 2.5 “Final-review hardness” checklist (Paper II)
- Minimum: **one staircase/boundary curve** (prefer \(k^*(\mathcal C/h_\mu)\) with CI).
- Seeds: ≥ 6 (prefer 10).
- Parameter table: \(D\), drift/noise grid, objective definition, scan range for \(k\).
- “Does show / does not show” bullet list (see §3).

### 2.6 Reproducibility packaging (recommended)
- `paper2_ssb_attention_entropy_demo.py` → `fig_paper2_ssb_entropy.pdf`
- `paper2_kstar_scaling_demo.py` → `fig_paper2_kstar_scaling.pdf`
- Suggested defaults:
  - \(D=32\) or \(64\)
  - \(T=5{,}000\)–\(20{,}000\)
  - seeds ≥ 10
  - scan \(k\in[1,D]\)
  - vary drift/noise to sweep \(h_\mu\) surrogate

---

## 3) What this companion note does / does not claim

### Does show (once the plots are produced)
- **Paper I:** a reproducible regime where finite-order Markov agents plateau while a memory/belief agent improves (a “Markov ceiling” signature).
- **Paper II:** under budget constraints, symmetry breaking emerges as a measurable collapse of attention entropy/active dimension; and \(k^*\) scales with \(\mathcal C_{\mathrm{budget}}/h_\mu\) (staircase signature).

### Does not show
- Universality across environments, objectives, or architectures.
- Tight constants or optimality of analytic bounds.
- The full delusion–correction closed loop (addressed in Paper III v2 numerical demonstration).

---

## 4) Archival packaging (Zenodo-ready, when hard enough)
Recommended assets to publish *together*:

- `companion_note.pdf` (this note, rendered)
- `fig_paper1_markov_ceiling.pdf`
- `fig_paper2_kstar_scaling.pdf`
- *(optional)* `fig_paper2_ssb_entropy.pdf`
- `code/` with scripts + `README.md` specifying:
  - python version
  - dependencies (NumPy, Matplotlib)
  - seeds + parameter grids
  - expected runtime

Suggested citation language for Paper I/II descriptions:
> “Minimal computational demonstrations supporting key mechanisms in Papers I–II are provided in a companion note (Zenodo record: …).”

---

## 5) Publish-or-wait rule (recommended)
Do **not** upload until at least **two** of the following are satisfied:

1) Paper I includes a boundary/curve (\(\ell_c(k)\) or \(\Delta\bar R(\ell)\)) with CI.  
2) Paper II includes a staircase/boundary curve (\(k^*(\mathcal C/h_\mu)\)) with CI.  
3) Parameter tables + seeds/CI are explicitly stated (reproducibility-grade).  
4) “Does show / does not show” is included verbatim (credit-safe).

Until then, keep this note as an internal spec.
