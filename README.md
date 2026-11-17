# SMASH-HCM-ECG

Analysis code for the study

> **Normal-anchored and age-conditioned ECG manifolds reveal directional electrical evolution in hypertrophic cardiomyopathy**
>
> Pietro Cerveri, Marion Taconnè, Valentina Corino, Luca Mainardi

developed at the University of Pavia and Politecnico di Milano within the EU-funded **SMASH-HCM** project.

---

## Study in brief

Hypertrophic cardiomyopathy (HCM) produces heterogeneous electrical phenotypes whose continuous
organization, longitudinal evolution and clinical relevance are poorly defined. Supervised AI-ECG
models detect HCM but do not describe how electrical phenotypes evolve. This repository implements an
**unsupervised framework of three variational autoencoders** learned from 12-lead median-beat
morphology alone, with both static manifolds referenced to an **age- and sex-conditioned normative
model**.

| Component | Role |
|---|---|
| **Normal-anchored β-VAE** (16 latents) | Physiological reference manifold learned only from strictly normal PTB-XL ECGs; measures deviation of HCM ECGs from normal morphology ($d_N$). |
| **HCM-oriented β-VAE** (8 latents) | Continuous map of electrical heterogeneity within HCM, interpreted with decoder traversals and ECG features; linked to the normal manifold by cross-manifold analysis. |
| **Normative model** | Expresses every frozen latent coordinate of both manifolds as a calibrated deviation from normal ECGs of the same age and sex, removing the ageing confound by construction. |
| **Longitudinal trajectories** | Serial HCM ECGs projected through the frozen encoders; mixed-effects and within-subject slopes, compared with repeated normal ECGs and decomposed into drift along and orthogonal to the axis of departure from normal. |
| **Longitudinal Delta-VAE** (4 latents) | Expresses the waveform difference between two visits as physiologically annotated modes, indicating which morphological component changed. |
| **External evaluation** | Frozen models applied, without retraining or rescaling, to an independent MIMIC-IV HCM cohort. |

### Cohorts

| Cohort | Content |
|---|---|
| PTB-XL normal reference | 5,194 ECGs (one per subject; NORM, human-validated, 18–90 years) |
| PTB-XL repeated normals | 131 subjects with repeated normal ECGs (longitudinal normal control) |
| SHaRe HCM | 2,594 valid ECG visits from 650 subjects; longitudinal cohort of 493 subjects / 2,437 visits |
| MIMIC-IV HCM | 5,564 eligible ECGs from 335 subjects (external evaluation) |

### Main findings

- HCM ECGs overlap the physiological manifold but extend towards its periphery; the HCM-oriented
  manifold largely reorganizes electrical modes already present in normal ECGs.
- The normative model met prespecified calibration criteria in held-out normal ECGs for 23 of 24
  latent coordinates.
- A lateral/frontal QRS-force coordinate ($z_{06}^{HCM}$) declines by 0.053 normative SD per year
  beyond the age- and sex-specific expectation in event-free patients, unlike repeated normal ECGs,
  and the decline is replicated in MIMIC-IV.
- Distance from normal morphology does not increase: 94% of the systematic drift is orthogonal to the
  axis of departure from normal, indicating directional rather than generalized electrical evolution.
- Lower $z_{06}^{HCM}$ is associated with subsequent heart failure (HR 1.38 per SD decrease,
  95% CI 1.12–1.71, adjusted for age and sex); the association is largely shared with left atrial
  size and ejection fraction.
- The Delta-VAE identifies four interpretable change modes, for example repolarization change after
  pacing and depolarization–ST/T change after septal reduction.
- Sarcomeric genotype is associated with phenotype position but not with trajectory rate.

These are associations from an observational analysis; $z_{06}^{HCM}$ is not a validated prognostic
marker.

---

## Pipeline

Scripts are numbered and run in order from the project root. Each script writes to
`Results/<step>_<name>/`.

| Steps | Stage |
|---|---|
| `00`–`02` | PTB-XL normal reference: download, validation, harmonization (first 4 s at 500 Hz) |
| `03`–`10` | Median beats (0.5–40 Hz, −350/+650 ms, ≥2 beats), Normal-anchored β-VAE training, traversals, interactive explorer, ECG features, residual analysis |
| `11`–`14` | HCM-oriented β-VAE: preparation, training, traversals, ECG features |
| `15` | Cross-manifold analysis |
| `16`–`17` | Frozen longitudinal projection and mixed-effects trajectories |
| `18` | Genotype × time interactions |
| `19`–`22` | Delta-VAE: training, interpretation (`20`, rate model `20b`), cluster-aware validation, integration with trajectories |
| `23` | Event-censoring sensitivity (SRT, AF, pacing) |
| `24`–`27` | Clinical-risk analyses: feasibility, baseline Cox, time-updated Cox (`26`, bootstrap `26b`), post-SRT landmark |
| `28`–`31` | MIMIC-IV: median beats, near-duplicate calibration, review and deduplication |
| `32` | Interactive viewer of patient positions in the HCM latent space |
| `33`–`36` | MIMIC-IV external evaluation: frozen projection, manifold transport, longitudinal and subject-balanced replication |
| `37` | Ageing sensitivity: PTB-XL age reference and age decomposition |
| `38` | Piecewise trajectory and event-censored cohorts |
| `39` | Robustness of the heart-failure association (SRT, AF, pacing, age handling, genotype) |
| `40` | Adjustment for echocardiographic and clinical markers |
| `41` | Exploratory ECG-only composite markers |
| `42` | Projection of an additional external cohort (`.mat` files) through the frozen VAEs |
| `43` | Longitudinal normal control from repeated PTB-XL ECGs |
| `44` | Age- and sex-conditioned normative model of both manifolds, calibration and normative trajectories |
| `45` | Benchmark of the Delta-VAE against simpler change measures and ablations of time conditioning |
| `46` | Directional decomposition of longitudinal drift in the whitened normative space |
| `47` | Latent dimensionality ablation (reconstruction error versus correlation between latent coordinates) |

Steps 37–47 are post-freeze analyses; their decision rules are documented in each script header.

---

## Data

**No patient data are included in this repository.**

| Source | Access |
|---|---|
| PTB-XL | Public, [PhysioNet](https://physionet.org/content/ptb-xl/) (`ptbxl_database.csv` and WFDB records) |
| MIMIC-IV-ECG / MIMIC-IV | Credentialed access through PhysioNet |
| SHaRe | Restricted; available through the Sarcomeric Human Cardiomyopathy Registry under its data-use governance |

Expected local layout (not tracked by git):

```
_SMASH-HCM/
├── Data/
│   ├── PTBXL_Normal/records500/...
│   ├── ptbxl_database.csv
│   ├── PTBXL_normal_reference_manifest.csv
│   ├── SHaReData-FLO.xlsx
│   ├── MIMIC-IV/...
│   └── ...
├── Results/
└── Scripts/
```

---

## Environment

Analyses were run with:

```
Python 3.12.10
PyTorch 2.13.0+cu132
NumPy 2.5.3
SciPy 1.18.1
pandas 3.0.6
statsmodels 0.15.0
scikit-learn 1.9.1
```

Additional packages: `wfdb`, `h5py`, `matplotlib`.

## Usage

```
cd _SMASH-HCM
python Scripts/00_download_ptbxl_normal_reference.py
...
python Scripts/47_latent_dimension_ablation.py
```

Paths are resolved relative to the project root; most scripts accept `--root` and input/output
arguments (`--help`).

---

## Citation

The citation will be added upon publication.

## Licence

Apache License 2.0 (see `LICENSE`).

## Funding

This work was funded by the European Union's Horizon Europe Research and Innovation Programme under
grant agreement No. 101137115 (SMASH-HCM).
