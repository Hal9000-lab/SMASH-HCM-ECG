# SMASH-HCM-ECG

Analysis code for the study

> **Interpretable ECG manifolds link longitudinal electrical phenotypes to heart-failure risk in hypertrophic cardiomyopathy**
> <!-- TODO: authors, preprint DOI / journal reference -->

developed at Politecnico di Milano (Milan, Italy) and Università di Pavia (Pavia, Italy) within the EU-funded **SMASH-HCM** project.

---

## Study in brief

Hypertrophic cardiomyopathy (HCM) produces heterogeneous electrical phenotypes whose continuous
organization, longitudinal evolution and clinical relevance are poorly defined. This repository
implements an **ECG-first, unsupervised latent-manifold framework**, based on β-Variational Autoencoders, built on 12-lead median beats.

| Component | Role |
|---|---|
| **Normal-anchored β-VAE** (16 latents) | Physiological reference manifold learned from PTB-XL normal ECGs; measures deviation of HCM ECGs from normal morphology ($d_N$). |
| **HCM-oriented static β-VAE** (8 latents) | Continuous map of electrical heterogeneity within HCM, interpreted with decoder traversals and ECG features. |
| **Frozen longitudinal projection** | Serial HCM ECGs projected through both frozen encoders; trajectories modelled with linear mixed-effects models. |
| **Delta-VAE** (4 latents, time-conditioned) | Learns recurrent modes of within-subject waveform change between consecutive visits. |
| **External evaluation** | Frozen models applied, without retraining or rescaling, to an independent MIMIC-IV HCM cohort. |

### Cohorts

| Cohort | Content |
|---|---|
| PTB-XL normal reference | 5,194 ECGs (one per subject; NORM, human-validated, 18–90 years) |
| SHaRe HCM | 2,594 valid ECG visits from 650 subjects; longitudinal cohort of 493 subjects / 2,437 visits |
| MIMIC-IV HCM | 5,564 eligible ECGs from 335 subjects (external evaluation) |

### Main findings

- HCM ECGs overlap the physiological manifold but extend towards its periphery; the HCM-oriented
  manifold largely reorganizes electrical modes already present in normal ECGs.
- A lateral/frontal QRS-force coordinate ($z_{06}^{HCM}$) declines over time, also after censoring
  septal reduction therapy, atrial fibrillation and pacing, and the decline is replicated in MIMIC-IV.
  Repeated normal PTB-XL ECGs show no comparable drift.
- Lower $z_{06}^{HCM}$ is associated with heart failure (HR 1.38 per SD decrease, 95% CI 1.12–1.71,
  adjusted for age and sex); the association is robust to clinical transitions and age but largely
  shared with left atrial size and ejection fraction.
- The Delta-VAE identifies four interpretable modes of within-subject change aligned with
  longer-term trajectories.
- Sarcomeric genotype is associated with phenotype position but not with trajectory rate.

These are associations from an exploratory, observational analysis; $z_{06}^{HCM}$ is not a
validated prognostic marker.

---

## Pipeline

Scripts are numbered and run in order from the project root. Each script writes to
`Results/<step>_<name>/`.

| Steps | Stage |
|---|---|
| `00`–`02` | PTB-XL normal reference: download, validation, harmonization (first 4 s at 500 Hz) |
| `03`–`10` | Median beats (0.5–40 Hz, −350/+650 ms, ≥2 beats), Normal-anchored β-VAE training, traversals, ECG features, residual analysis |
| `11`–`14` | HCM-oriented static β-VAE: preparation, training, traversals, ECG features |
| `15`–`22` | Cross-manifold integration, frozen longitudinal projection, mixed-effects trajectories, genetics, Delta-VAE and change-mode interpretation <!-- TODO: verify step-by-step names --> |
| `23` | Event-censoring sensitivity (SRT, AF, pacing) |
| `24`–`26` | Clinical-risk analyses: event feasibility, baseline Cox, time-updated Cox |
| `27`–`36` | External evaluation in MIMIC-IV: median beats, frozen projection, transport and longitudinal replication <!-- TODO: verify step-by-step names --> |
| `37` | Ageing sensitivity: PTB-XL age reference and age decomposition |
| `38` | Piecewise trajectory and event-censored cohorts |
| `39` | Robustness of the heart-failure association (SRT, AF, pacing, age handling, genotype) |
| `40` | Adjustment for echocardiographic and clinical markers |
| `41` | Exploratory ECG-only composite markers |
| `42` | Projection of an additional external cohort (`.mat` files) through the frozen VAEs |
| `43` | Longitudinal normal control from repeated PTB-XL ECGs |

Steps 37–43 are post-freeze sensitivity analyses; their decision rules are documented in each
script header.

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

Additional packages: `wfdb`, `h5py`, `matplotlib` <!-- TODO: pin versions in requirements.txt -->.

## Usage

```
cd _SMASH-HCM
python Scripts/00_download_ptbxl_normal_reference.py
...
python Scripts/43_ptbxl_normal_longitudinal_control.py
```

Paths are resolved relative to the project root; most scripts accept `--root` and input/output
arguments (`--help`).

---

## Citation

<!-- TODO: add citation once available -->

## Licence

<!-- TODO: choose a licence (e.g. MIT, BSD-3-Clause) -->

## Funding

This work was carried out within the SMASH-HCM project, funded by the European Union.
<!-- TODO: grant agreement number -->
