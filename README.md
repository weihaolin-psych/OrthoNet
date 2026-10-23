# OrthoNet: data and analysis code

This repository accompanies *A Neural Network Model of Retrieval Difficulty in
Chinese Handwriting*. OrthoNet maps semantic and phonological representations
to Chinese characters. The analyses relate model entropy and relative target
support to human tip-of-the-pen (TOP) responses and writing latencies.

## Files and companion data

| Directory | Contents |
|---|---|
| data/raw/ | Source behavioral workbooks and the Chinese Lexical Database |
| data/processed/ | Training tokens, word dictionary and canonical human trial files |
| data/ablation/ | Full, meaning-only and sound-only model entropy |
| data/logit_measures/ | Position-level and Character-Word relative-support measures |
| data/analysis_inputs/ | Supplied parameter grids, sensitivity trials, learning summaries and preparation records |
| preprocessing/ | Data preparation and model-measure aggregation |
| models/ | One-layer, two-layer and ablation training/evaluation code |
| analysis/ | Numerical analyses; runtime results are written to ignored output folders |

The companion Zenodo data deposit contains these five files, with the exact
relative paths listed below:

```text
data/embeddings/word_meaning_embeddings.pt
data/embeddings/word_pinyin_embeddings.pt
data/analysis_inputs/sensitivity_onelayer_random10/wang_1400_entropy_with_covariates.csv
data/analysis_inputs/sensitivity_twolayers_random10/wang_1400_entropy_with_covariates.csv
data/logit_measures/position_metrics_by_seed.csv
```

Copy the companion deposit's `data/` directory into this repository root,
merging directories. The resulting paths must match the list above. Ordinary
trial-based mixed models can start from the supplied processed trials without
the embeddings or parameter grids. Rebuilding crowdedness requires embeddings;
parameter selection requires the grids; rebuilding relative-measure trial files
requires the position-level CSV.

Repository URL: **to be added before publication**.
Code-version Zenodo DOI: **to be added before publication**.
Companion-data Zenodo DOI: **to be added before publication**.

Trained model checkpoints, per-epoch evaluation CSVs and complete-logit arrays
are not included. For access to trained model checkpoints, please contact the
first author of the accompanying article. The supplied model measures and
summaries support downstream analyses without loading the trained checkpoints.
Recreating all upstream outputs requires training/evaluation or access to the
corresponding saved model files. Learning trajectories are supplied summaries;
their underlying per-epoch evaluations are not distributed.

Manuscript production files, figure/table layout code, and previously generated
statistical results are outside this release. Numerical calculations formerly
combined with display code are provided as analysis scripts.

## Environment

Run commands from the repository root. `requirements.txt` records Python
dependencies; `renv.lock` records R packages. The numerical R scripts principally
use here, dplyr, lme4, lmerTest and performance. The performance version used for
the archived mixed-model results was 0.18.2. Choose a PyTorch build appropriate
for CPU or GPU hardware; training is more practical on a GPU. Missing packages
must be installed explicitly; analysis scripts do not install them automatically.

The release has been checked statically. Training and full statistical analyses
have not been rerun as part of release preparation, and installation on a new
machine has not been validated. The package records do not constitute a tested
environment restoration.

## Data conventions

Huang and Wang Small contain different participant samples for the same 200
materials. Wang Full contains 1,600 items, with model outputs for 1,592 designated
Character-Word contexts. Eight unmatched contexts retain missing model values.
Match model scores on both Character and Word, rather than averaging over words.

`top_eligible` selects TOP-analysis trials and `top_binary` codes TOP as 1 and
correct responses as 0; other response categories are outside that denominator.
Latency analyses use `rt_eligible`, the 10,000 ms cutoff and natural-log `log_rt`.
The lexical covariates include frequency, age of acquisition, context-word
familiarity, phonetic radical order, spelling regularity and stroke count.

The main trial families are `*_trials.csv`, `*_trials_with_entropy.csv`,
`*_trials_with_entropy_crowdedness.csv`, and `*_trials_with_logit_measures.csv`.
The last family retains three standardized model measures for audit compatibility;
the two relative measures are target-minus-mean and target-minus-maximum
non-target logits. Differences are standardized within seed before averaging
repeated positions within Character-Word and then averaging ten seeds.

The seeds are 7, 42, 88, 123, 200, 314, 456, 555, 789 and 999. The reported two-layer
configuration is hidden dimension 22, epoch 11. File names containing `1400` are
historical: actual validation coverage follows the data and exclusion rules.

## Numerical analyses

Scripts write fresh numerical results and may replace earlier runtime outputs.
No saved output is needed for the independent entries below. The exceptions and
their required run order are stated explicitly.

| Analysis | Command from the repository root |
|---|---|
| Human TOP-latency associations and descriptives | `Rscript --vanilla analysis/top_latency_correlation/01_top_latency_correlation.R` |
| Entropy mixed models | `Rscript --vanilla analysis/mixed_effects_entropy/01_fit_models.R` |
| Entropy-behavior correlations | `Rscript --vanilla analysis/mixed_effects_entropy/02_correlate_behavior.R` |
| Ablation mixed models | `Rscript --vanilla analysis/ablation/01_fit_models.R` |
| Crowdedness mixed models from prepared trials | `Rscript --vanilla analysis/crowdedness_analysis/06_fit_models.R` |
| Relative-support mixed models | `Rscript --vanilla analysis/mixed_effects_logits/02_fit_standardized_measures.R` |
| Relative-support behavior correlations | `Rscript --vanilla analysis/mixed_effects_logits/03_correlate_behavior.R` |
| Pairwise uncertainty correlations | `Rscript --vanilla analysis/logit_measures/correlate_uncertainty_measures.R` |
| Model-human difficulty classification | `Rscript --vanilla analysis/error_analysis/01_classify_and_rank.R` |
| Divergence group-mean tests | `Rscript --vanilla analysis/error_analysis/02_compare_group_means.R` |
| Frequency-AoA associations and VIF | `Rscript --vanilla analysis/lexical_covariates/audit_covariates.R` |

Run entropy mixed models before the VIF audit: that audit reads the newly
generated fitted objects. Run the classification before the group-mean tests.
The group tests preserve the original two-sided one-sample comparison of each
selected group against the overall item mean, treated as fixed, including the
selected items in that overall mean. The TOP-latency script includes a
10,000-replicate bootstrap and may take appreciable time.

Relative-support mixed models read the supplied preparation records in
`data/analysis_inputs/relative_measure_preparation/`. These records verify the
trial-file contents. To rebuild the trial files and their records, run
`python preprocessing/after_training/merge_logit_measures.py` after installing
the companion data. Run entropy mixed models first if the optional historical
base-model comparison is desired; it is not required to fit relative models.

### Parameter selection and sensitivity

Run `python analysis/sensitivity_onelayer_random10/03_compute_parameter_selection.py`
and `python analysis/sensitivity_twolayers_random10/03_compute_parameter_selection.py`
after adding the companion grids. These scripts preserve complete-case filtering,
residualization against the six configured covariates, arithmetic averaging of
seed-wise Pearson r, sample SD across seeds, and maximum-mean-r selection without
an accuracy threshold. They write numerical grids and selected-cell results.

The numbered 01/02 scripts in those folders are upstream extraction tools.
They require per-epoch evaluation CSVs to rebuild grids. They are not prerequisites
for using the supplied grids. A freshly rebuilt grid in the analysis output
folder takes precedence over the supplied grid.

For mixed-model sensitivity, run `Rscript --vanilla` followed by
`analysis/glmer_sensitivity_onelayer_averaged10seeds/02_sensitivity_glmer_averaged.R`
or `analysis/glmer_sensitivity_twolayers_averaged10seeds/02_sensitivity_glmer_averaged.R`.
Both use the four supplied trial-by-parameter input files under
`data/analysis_inputs/`. Their 01 extraction scripts require per-epoch evaluations;
freshly extracted trial files in output/ take precedence over supplied trials.

The five files under `data/analysis_inputs/model_behavior/` preserve the learning
trajectory, selected-epoch scores, dataset summaries and provenance. The script
`analysis/model_behavior/01_summarize_model_behavior.py` regenerates these from
per-epoch evaluations if those are available; it writes to output/ and does not
overwrite the supplied inputs.

### Rebuilding crowdedness and model inputs

To rebuild crowdedness, run the scripts in `analysis/crowdedness_analysis/` in
order: 01 validation selection, 02 type frequency, 03 distances/crowdedness,
04_select_lambda.py, 05 merge, and 06 mixed models. The 03 computation requires
the companion embeddings and can be expensive. This route is optional when
starting from the supplied crowdedness trial files.

`preprocessing/before_training/` contains corpus extraction, dictionary building,
embedding extraction and behavioral preparation. The supplied data provide a
direct analysis entry point; regenerating embeddings also depends on the
availability and versions of the pretrained resources below.

For one model run:

```text
python models/twolayers_10-random-seeds/training/train_twolayers.py --hidden_dim 22 --epochs 25 --seed 42
python models/twolayers_10-random-seeds/evaluate_twolayers.py --hidden_dim 22 --seed 42
```

The model batch launchers cover all ten seeds. See models/README.md for model
variants and checkpoint/evaluation locations. `average_ablation_entropy.py`
updates and checks existing ablation CSVs against new model evaluations; it is
not a stand-alone first-time bootstrap of all entropy trial files. Statistical
analyses should start from the supplied canonical trials unless deliberately
rebuilding upstream materials.

## Original sources and attribution

Huang, S., Lin, W., Xu, M., Wang, R., & Cai, Z. G. (2021). On the tip of the pen:
Effects of character-level lexical variables and handwriter-level individual
differences on orthographic retrieval difficulties in Chinese handwriting.
*Quarterly Journal of Experimental Psychology, 74*(9), 1497-1511.
https://doi.org/10.1177/17470218211004385 . Original data: https://osf.io/zfg5v/ .
Retained workbooks are under `data/raw/huang2021/`.

Wang, R., Huang, S., Zhou, Y., & Cai, Z. G. (2020). Chinese character handwriting:
A large-scale behavioral study and a database. *Behavior Research Methods,
52*(1), 82-96. https://doi.org/10.3758/s13428-019-01206-4 . Retained workbooks
are under `data/raw/wang2020/`.

Sun, C. C., Hendrix, P., Ma, J., & Baayen, R. H. (2018). Chinese Lexical Database
(CLD): A large-scale lexical database for simplified Mandarin Chinese.
*Behavior Research Methods, 50*(6), 2606-2629.
https://doi.org/10.3758/s13428-018-1038-3 . The retained corpus is under
`data/raw/CLD_chineselexicaldatabase/`.

Semantic embeddings use `hfl/chinese-roberta-wwm-ext`; phonological embeddings
use `iioSnail/ChineseBERT-base`. Cite the article and original resources when
using these materials. Third-party resources retain their original terms.
No new blanket license is assigned to third-party materials by this release.
