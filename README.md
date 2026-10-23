# OrthoNet: A Neural Network Model of Retrieval Difficulty in Chinese Handwriting

This repository contains the data, model code, and analysis scripts for OrthoNet, a feed-forward neural network model of orthographic retrieval difficulty in Chinese handwriting. The model maps semantic and phonological representations to character outputs. Shannon entropy over the output distribution provides a continuous measure of retrieval uncertainty, evaluated against human tip-of-the-pen (TOP) rates and writing latencies from dictation experiments.

## Model Overview

OrthoNet combines a 768-dimensional semantic representation from Chinese RoBERTa with a 768-dimensional phonological representation from ChineseBERT's pinyin encoder. These inputs pass through two hidden layers to produce a softmax distribution over 4,677 Chinese characters.

The training data are derived from the Chinese Lexical Database (CLD), covering approximately 48,000 words and 106,844 character-in-word instances. Training weights are proportional to word frequency raised to the power of 0.75.

The selected configuration has **22 units per hidden layer and 11 training epochs**. Model measures are averaged over ten random seeds: 7, 42, 88, 123, 200, 314, 456, 555, 789, and 999. The repository also includes the one-hidden-layer comparison model and meaning-only and sound-only ablation models.

Alongside entropy, the analyses examine relative target support: the target logit minus either the mean or maximum non-target logit. Each difference is standardized within seed before averaging repeated positions within a Character-Word context and then averaging across seeds.

## Human Behavioral Datasets

Model predictions are evaluated against three datasets:

- **Huang et al. (2021):** 200 characters, with 64 participants. TOP judgments were collected before the correct answer was shown.
- **Wang Small:** the 200 characters from Wang et al. (2020) that overlap with the Huang materials, tested in a different participant sample.
- **Wang Full:** all 1,600 characters in Wang et al. (2020), including the materials used for hyperparameter selection. Model scores are available for 1,592 designated Character-Word contexts; the eight unmatched contexts retain missing values.

Hyperparameters were selected using the Wang materials outside the shared 200-character set, with 1,392 validation items retained after corpus filtering. The selection criterion was the mean seed-wise partial correlation between entropy and log writing latency, controlling for lexical covariates. The Huang and Wang Small materials were excluded from this selection. File names containing `1400` retain the original naming convention.

## Repository Structure

```text
OrthoNet/
├── data/
│   ├── raw/                       # Original behavioral datasets and CLD
│   ├── processed/                 # Human trials, training tokens, and word dictionary
│   ├── ablation/                  # Full, meaning-only, and sound-only model entropy
│   ├── logit_measures/             # Character-Word and position-level model measures
│   ├── analysis_inputs/            # Parameter grids, sensitivity trials, and model summaries
│   └── embeddings/                # Input tensors supplied in the companion Zenodo deposit
├── preprocessing/
│   ├── before_training/           # Corpus, embedding, and behavioral-data preparation
│   └── after_training/            # Model-measure aggregation and trial-data merging
├── models/
│   ├── twolayers_10-random-seeds/  # Main two-hidden-layer model
│   ├── onelayer_10-random-seeds/   # One-hidden-layer comparison
│   └── ablation_10-random-seeds/   # Meaning-only and sound-only models
├── analysis/
│   ├── top_latency_correlation/   # Human descriptives and TOP-latency associations
│   ├── mixed_effects_entropy/     # Entropy mixed models and behavioral correlations
│   ├── ablation/                  # Full and ablated model comparisons
│   ├── crowdedness_analysis/      # Embedding-space crowdedness
│   ├── mixed_effects_logits/      # Relative-support models and behavioral correlations
│   ├── logit_measures/            # Correlations among uncertainty measures
│   ├── error_analysis/            # Model-human divergence and group comparisons
│   ├── lexical_covariates/        # Frequency-AoA associations and VIF
│   ├── model_behavior/            # Entropy learning trajectories and item summaries
│   ├── sensitivity_*_random10/    # Parameter selection for each architecture
│   └── glmer_sensitivity_*_averaged10seeds/  # Mixed-model sensitivity
├── requirements.txt              # Python dependencies
├── renv.lock                     # R package versions
├── tip_of_pen.Rproj
└── README.md
```

## Reproducing the Analyses

Run commands from the repository root. The supplied processed data are the starting point for the statistical analyses; retraining the network is optional. Scripts save their numerical results in the corresponding analysis folders. Existing analysis outputs and fitted objects are not distributed.

### Main analyses

| Analysis | Entry point |
|---|---|
| Human descriptives and TOP-latency correlations | `analysis/top_latency_correlation/01_top_latency_correlation.R` |
| Entropy mixed models | `analysis/mixed_effects_entropy/01_fit_models.R` |
| Entropy-behavior correlations | `analysis/mixed_effects_entropy/02_correlate_behavior.R` |
| Ablation comparisons | `analysis/ablation/01_fit_models.R` |
| Crowdedness mixed models | `analysis/crowdedness_analysis/06_fit_models.R` |
| Relative-support mixed models | `analysis/mixed_effects_logits/02_fit_standardized_measures.R` |
| Relative-support behavioral correlations | `analysis/mixed_effects_logits/03_correlate_behavior.R` |
| Correlations among uncertainty measures | `analysis/logit_measures/correlate_uncertainty_measures.R` |
| Model-human difficulty classification | `analysis/error_analysis/01_classify_and_rank.R` |
| Divergence group comparisons | `analysis/error_analysis/02_compare_group_means.R` |
| Frequency-AoA associations and VIF | `analysis/lexical_covariates/audit_covariates.R` |

For example, fit the entropy models with:

```text
Rscript --vanilla analysis/mixed_effects_entropy/01_fit_models.R
```

Run the entropy models before the VIF analysis, which reads the fitted objects generated by that step. Run the difficulty classification before the divergence group comparisons. Those comparisons use two-sided one-sample t-tests against the overall item mean, treated as fixed. The TOP-latency analysis includes a 10,000-replicate bootstrap.

Model scores are matched to behavioral trials on **both Character and Word**. TOP analyses use `top_eligible` and `top_binary`; latency analyses use `rt_eligible`, a 10,000 ms cutoff, and natural-log latency. Trial files ending in `_with_entropy`, `_with_entropy_crowdedness`, and `_with_logit_measures` contain the inputs for the corresponding analyses.

### Parameter selection and sensitivity

After adding the companion data described below, run:

```text
python analysis/sensitivity_twolayers_random10/03_compute_parameter_selection.py
python analysis/sensitivity_onelayer_random10/03_compute_parameter_selection.py
```

These scripts read the supplied parameter grids and save seed-wise correlations, averaged grids, and selected parameters. Selection uses the largest mean partial correlation across seeds, without accuracy filtering.

For mixed-model sensitivity, run `02_sensitivity_glmer_averaged.R` in either `analysis/glmer_sensitivity_twolayers_averaged10seeds/` or `analysis/glmer_sensitivity_onelayer_averaged10seeds/`. The required trial-by-parameter data are supplied in `data/analysis_inputs/`. Newly rebuilt grids or sensitivity trials in an analysis output folder take precedence over the supplied inputs.

Learning trajectories and selected-epoch summaries are supplied in `data/analysis_inputs/model_behavior/`. Regenerating them with `analysis/model_behavior/01_summarize_model_behavior.py` requires the per-epoch model evaluation files.

## Data Preprocessing and Model Training

To rebuild the model inputs, run scripts 01-06 in `preprocessing/before_training/` in numbered order. These extract CLD training instances, build the word dictionary, obtain semantic and phonological embeddings, and prepare the behavioral trials.

Training and evaluation code is in `models/`. For example, train and evaluate one two-hidden-layer model with:

```text
python models/twolayers_10-random-seeds/training/train_twolayers.py --hidden_dim 22 --epochs 25 --seed 42
python models/twolayers_10-random-seeds/evaluate_twolayers.py --hidden_dim 22 --seed 42
```

The 25-epoch run produces the learning trajectory and includes the selected epoch 11. Batch launchers cover all ten seeds and can run sequentially or submit jobs to a configured SLURM cluster. Training produces `.pth` checkpoints; evaluation produces per-epoch CSVs. See [models/README.md](models/README.md) for the model commands and output locations.

The numbered extraction scripts in the parameter-selection and mixed-model sensitivity folders rebuild their inputs from model evaluations. To rebuild crowdedness, run scripts 01-06 in `analysis/crowdedness_analysis/`, including `04_select_lambda.py`; the distance computation requires the companion embeddings. This is optional when using the supplied crowdedness trial files.

`preprocessing/after_training/merge_logit_measures.py` rebuilds the relative-measure trial files and preparation records from the supplied model measures. The fitting script also accepts the existing records in `data/analysis_inputs/relative_measure_preparation/`. `average_ablation_entropy.py` updates existing ablation data from new model evaluations; it is not needed to analyze the supplied ablation data.

## Large Files and Model Checkpoints

Code repository: [weihaolin-psych/OrthoNet](https://github.com/weihaolin-psych/OrthoNet).

- **Code archive:** [Zenodo DOI to be added].
- **Companion data:** [Zenodo DOI to be added].

The following five large files are supplied separately in the companion data deposit:

```text
data/embeddings/word_meaning_embeddings.pt
data/embeddings/word_pinyin_embeddings.pt
data/analysis_inputs/sensitivity_onelayer_random10/wang_1400_entropy_with_covariates.csv
data/analysis_inputs/sensitivity_twolayers_random10/wang_1400_entropy_with_covariates.csv
data/logit_measures/position_metrics_by_seed.csv
```

Copy the companion deposit's `data/` folder into the repository root, preserving these paths and merging directories. Embeddings are needed for training and crowdedness reconstruction, grids for parameter selection, and position-level metrics for rebuilding relative-measure trial files. The main mixed-model analyses can use the processed trials directly.

**Trained model checkpoints are available from the first author on request.** Trained weights, per-epoch evaluation CSVs, and complete-logit arrays are not included in either deposit. Rebuilding their derived measures requires generating the upstream files or obtaining the necessary saved model files. Manuscript figure/table production files are outside the scope of this repository.

## Software Requirements

- **Python:** PyTorch, Transformers, NumPy, pandas, SciPy, scikit-learn, tqdm, pyprojroot, and openpyxl; see `requirements.txt` for the recorded dependencies.
- **R:** here, dplyr, lme4, lmerTest, and performance; package versions are recorded in `renv.lock`.
- **Hardware:** a GPU is recommended for model training. Select a PyTorch installation appropriate for your hardware. Behavioral statistical analyses do not require a GPU.

Install dependencies before running the scripts. `Rscript --vanilla` runs the analysis without automatically activating the local renv environment.

## Data Sources

- **Human handwriting data:** [Huang et al. (2021)](https://doi.org/10.1177/17470218211004385), with original data on [OSF](https://osf.io/zfg5v/); [Wang et al. (2020)](https://doi.org/10.3758/s13428-019-01206-4).
- **Training corpus:** the [Chinese Lexical Database (Sun et al., 2018)](https://doi.org/10.3758/s13428-018-1038-3).
- **Pretrained representations:** semantic embeddings from `hfl/chinese-roberta-wwm-ext`; phonological embeddings from `iioSnail/ChineseBERT-base`.

Please cite the accompanying article and the original resources when using these materials. Third-party datasets and pretrained resources retain their original terms of use.
