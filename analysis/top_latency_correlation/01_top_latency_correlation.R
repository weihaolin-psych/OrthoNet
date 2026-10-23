#!/usr/bin/env Rscript
# Character-level descriptive statistics and TOP-latency correlations.
# Open tip_of_pen.Rproj and source this file, or run from the repository root:
# Rscript analysis/top_latency_correlation/01_top_latency_correlation.R
# Reads canonical processed trial data with entropy for the three datasets.
# Outputs: character/descriptive/correlation/audit CSVs,
# configuration and session info. Re-running also repeats the existing bootstrap.

# ---- Configuration ----------------------------------------------------------

# NULL uses here::here(); supply an absolute repository path to run elsewhere.
PROJECT_ROOT <- NULL
DATA_FILES <- c(
  Huang = "data/processed/huang_trials_with_entropy.csv",
  "Wang Small" = "data/processed/wang_small_trials_with_entropy.csv",
  "Wang Full" = "data/processed/wang_full_trials_with_entropy.csv"
)
DATASET_ORDER <- names(DATA_FILES)
OUTPUT_FILES <- c(
  characters = "analysis/top_latency_correlation/output/character_summary.csv",
  descriptive = "analysis/top_latency_correlation/output/descriptive_statistics.csv",
  correlations = "analysis/top_latency_correlation/output/correlations.csv",
  screening = "analysis/top_latency_correlation/output/screening_summary.csv",
  session = "analysis/top_latency_correlation/output/session_info.txt",
  configuration = "analysis/top_latency_correlation/output/configuration.txt"
)
RT_MAX_MS <- 10000
BOOTSTRAP_REPLICATES <- 10000L
RANDOM_SEED <- 20260917L
RNG_SETTINGS <- list(kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
CONFIDENCE_LEVEL <- 0.95
COEFFICIENT_DIGITS <- 2L
TOP_DESCRIPTIVE_DIGITS <- 2L
LATENCY_DESCRIPTIVE_DIGITS <- 1L
P_DIGITS <- 3L
P_THRESHOLD <- 0.001
# Adjust exact zero/one TOP proportions before the logit transform.
ZERO_FRACTION <- 0.5
ONE_GAP_FRACTION <- 0.5

# These initial read-only results trigger warnings, never replace calculations.
REFERENCE_N <- c(Huang = 200L, "Wang Small" = 199L, "Wang Full" = 1591L)
REFERENCE_PEARSON <- c(Huang = .5629, "Wang Small" = .4019, "Wang Full" = .6542)
REFERENCE_SPEARMAN <- c(Huang = .5232, "Wang Small" = .3709, "Wang Full" = .6356)
REFERENCE_TOLERANCE <- 0.0001

REQUIRED_PACKAGES <- c("here", "dplyr")

# ---- Dependency and configuration checks ------------------------------------

missing_packages <- REQUIRED_PACKAGES[!vapply(
  REQUIRED_PACKAGES, requireNamespace, logical(1), quietly = TRUE
)]
if (length(missing_packages)) {
  stop("Missing R packages: ", paste(missing_packages, collapse = ", "),
       ". Install these yourself before running this script.", call. = FALSE)
}
suppressPackageStartupMessages(library(dplyr))
if (is.null(PROJECT_ROOT)) PROJECT_ROOT <- here::here()
PROJECT_ROOT <- normalizePath(PROJECT_ROOT, winslash = "/", mustWork = TRUE)
if (!file.exists(file.path(PROJECT_ROOT, "tip_of_pen.Rproj"))) {
  stop("Repository root not found. Open tip_of_pen.Rproj or set PROJECT_ROOT.")
}
stopifnot(
  is.finite(RT_MAX_MS), RT_MAX_MS > 0,
  BOOTSTRAP_REPLICATES >= 2L,
  BOOTSTRAP_REPLICATES == as.integer(BOOTSTRAP_REPLICATES),
  CONFIDENCE_LEVEL > 0, CONFIDENCE_LEVEL < 1,
  ZERO_FRACTION > 0, ZERO_FRACTION < 1,
  ONE_GAP_FRACTION > 0, ONE_GAP_FRACTION < 1,
  setequal(DATASET_ORDER, names(DATA_FILES)), !anyDuplicated(DATASET_ORDER)
)
input_paths <- setNames(file.path(PROJECT_ROOT, DATA_FILES), names(DATA_FILES))
output_paths <- setNames(file.path(PROJECT_ROOT, OUTPUT_FILES), names(OUTPUT_FILES))
if (any(!file.exists(input_paths))) {
  stop("Missing input files: ", paste(input_paths[!file.exists(input_paths)], collapse = ", "))
}

# ---- Helper functions -------------------------------------------------------

parse_eligibility <- function(values, column, dataset) {
  text <- tolower(trimws(as.character(values)))
  invalid <- !is.na(values) & !text %in% c("true", "false", "1", "0")
  if (any(invalid)) stop(dataset, ": unrecognized values in ", column)
  # Missing eligibility is excluded, matching filter() in Figure 4.
  !is.na(values) & text %in% c("true", "1")
}

aggregate_dataset <- function(path, dataset_name) {
  trials <- read.csv(path, fileEncoding = "UTF-8", stringsAsFactors = FALSE)
  required <- c("Character", "Word", "top_binary", "top_eligible", "rt",
                "log_rt", "rt_eligible", "entropy")
  if (!all(required %in% names(trials))) {
    stop(dataset_name, ": missing columns: ",
         paste(setdiff(required, names(trials)), collapse = ", "))
  }
  if (anyNA(trials[c("Character", "Word")]) ||
      any(trials$Character == "" | trials$Word == "")) {
    stop(dataset_name, ": missing Character or Word identifiers.")
  }
  for (column in c("top_binary", "rt", "log_rt", "entropy")) {
    if (!is.numeric(trials[[column]])) stop(dataset_name, ": ", column, " must be numeric.")
  }
  trials$use_top <- parse_eligibility(trials$top_eligible, "top_eligible", dataset_name)
  trials$use_rt <- parse_eligibility(trials$rt_eligible, "rt_eligible", dataset_name)
  trials$keep_rt <- trials$use_rt & !is.na(trials$rt) & trials$rt <= RT_MAX_MS
  if (any(!trials$top_binary[trials$use_top] %in% c(0, 1))) {
    stop(dataset_name, ": eligible TOP trials must have non-missing binary outcomes.")
  }
  if (any(!is.finite(trials$rt[trials$keep_rt])) ||
      any(trials$rt[trials$keep_rt] <= 0) ||
      any(!is.finite(trials$log_rt[trials$keep_rt]))) {
    stop(dataset_name, ": retained latency trials require positive RT and finite log RT.")
  }

  # TOP and latency have different trial denominators; preserve both counts.
  top <- trials %>%
    filter(use_top) %>% group_by(Character, Word) %>%
    summarise(n_top_trials = n(), n_top_events = sum(top_binary),
              top_rate = mean(top_binary), entropy = first(entropy), .groups = "drop")
  latency <- trials %>%
    filter(keep_rt) %>% group_by(Character, Word) %>%
    summarise(n_latency_trials = n(), mean_log_rt = mean(log_rt),
              mean_rt = mean(rt), .groups = "drop")
  characters <- trials %>% distinct(Character, Word) %>%
    left_join(top, by = c("Character", "Word")) %>%
    left_join(latency, by = c("Character", "Word")) %>%
    mutate(dataset = dataset_name,
           retained_for_transform = !is.na(top_rate) & !is.na(entropy))

  # Match Figure 4: determine boundary replacements before pairing with latency.
  rates <- characters$top_rate[characters$retained_for_transform]
  if (!length(rates) || any(!is.finite(rates)) || any(rates < 0 | rates > 1)) {
    stop(dataset_name, ": no valid character-level TOP rates for transformation.")
  }
  if (!any(rates > 0) || !any(rates < 1)) {
    stop(dataset_name, ": TOP rates are all zero or all one; correlation is undefined.")
  }
  zero_replacement <- min(rates[rates > 0]) * ZERO_FRACTION
  largest_below_one <- max(rates[rates < 1])
  one_replacement <- largest_below_one + (1 - largest_below_one) * ONE_GAP_FRACTION
  characters <- characters %>%
    mutate(
      adjusted_top_rate = case_when(
        !retained_for_transform ~ NA_real_,
        top_rate == 0 ~ zero_replacement,
        top_rate == 1 ~ one_replacement,
        TRUE ~ top_rate
      ),
      logit_top = qlogis(adjusted_top_rate),
      included = retained_for_transform & is.finite(logit_top) & is.finite(mean_log_rt),
      exclusion_reason = case_when(
        is.na(n_top_trials) ~ "no_TOP_eligible_trials",
        is.na(entropy) ~ "missing_entropy",
        !is.finite(mean_log_rt) ~ "no_finite_mean_log_latency",
        !is.finite(logit_top) ~ "nonfinite_logit_TOP",
        TRUE ~ "included"
      )
    )
  paired <- characters %>% filter(included)
  if (nrow(paired) < 4L || sd(paired$logit_top) == 0 || sd(paired$mean_log_rt) == 0) {
    stop(dataset_name, ": need at least four paired characters and variation in both measures.")
  }
  screening <- data.frame(
    dataset = dataset_name, input_trials = nrow(trials), input_characters = nrow(characters),
    top_eligible_trials = sum(trials$use_top), rt_eligible_trials = sum(trials$use_rt),
    rt_trials_above_limit = sum(trials$use_rt & trials$rt > RT_MAX_MS, na.rm = TRUE),
    rt_eligible_missing_rt = sum(trials$use_rt & is.na(trials$rt)),
    retained_latency_trials = sum(trials$keep_rt),
    no_top_trials = sum(is.na(characters$n_top_trials)),
    missing_entropy_with_top = sum(!is.na(characters$n_top_trials) & is.na(characters$entropy)),
    transform_characters = length(rates),
    excluded_after_transform = length(rates) - nrow(paired),
    paired_characters = nrow(paired), paired_zero_rates = sum(paired$top_rate == 0),
    paired_one_rates = sum(paired$top_rate == 1),
    zero_replacement = zero_replacement, one_replacement = one_replacement
  )
  list(characters = characters, screening = screening)
}

correlation_row <- function(x, y, dataset_name, measure, method) {
  if (method == "pearson") {
    test <- cor.test(x, y, method = "pearson", conf.level = CONFIDENCE_LEVEL)
    interval <- unname(test$conf.int)
    ci_method <- "Fisher z"
  } else {
    test <- cor.test(x, y, method = "spearman", exact = FALSE)
    # Resample entire character pairs and rerank within each bootstrap sample.
    bootstrap_rho <- replicate(BOOTSTRAP_REPLICATES, {
      indices <- sample.int(length(x), length(x), replace = TRUE)
      suppressWarnings(cor(x[indices], y[indices], method = "spearman"))
    })
    if (any(!is.finite(bootstrap_rho))) {
      stop(dataset_name, ": a bootstrap sample had zero variance; cannot report this CI.")
    }
    tail_probability <- (1 - CONFIDENCE_LEVEL) / 2
    interval <- unname(quantile(bootstrap_rho, c(tail_probability, 1 - tail_probability)))
    ci_method <- "Paired-character percentile bootstrap"
  }
  data.frame(
    dataset = dataset_name, top_measure = measure, latency_measure = "mean_log_rt",
    method = method, n = length(x), estimate = unname(test$estimate),
    ci_lower = interval[1], ci_upper = interval[2], p_value = test$p.value,
    confidence_level = CONFIDENCE_LEVEL, ci_method = ci_method,
    bootstrap_replicates = if (method == "spearman") BOOTSTRAP_REPLICATES else NA_integer_
  )
}

# Descriptive SDs reflect variation between character means, not between trials.
describe_dataset <- function(characters, dataset_name) {
  values <- characters %>% filter(dataset == dataset_name, !is.na(entropy))
  top_rates <- values$top_rate[is.finite(values$top_rate)]
  latencies <- values$mean_rt[is.finite(values$mean_rt)]
  if (length(top_rates) < 2L || length(latencies) < 2L) {
    stop(dataset_name, ": need at least two characters per measure for descriptive SDs.")
  }
  # Use each measure's available characters; do not require a paired latency for TOP.
  data.frame(
    dataset = dataset_name,
    n_top_characters = length(top_rates), top_rate_mean = mean(top_rates),
    top_rate_sd = sd(top_rates),
    n_latency_characters = length(latencies), latency_mean_ms = mean(latencies),
    latency_sd_ms = sd(latencies)
  )
}

# ---- Read and aggregate -----------------------------------------------------

aggregated <- lapply(DATASET_ORDER, function(dataset_name) {
  message("Aggregating: ", dataset_name)
  aggregate_dataset(input_paths[[dataset_name]], dataset_name)
})
all_characters <- bind_rows(lapply(aggregated, function(x) x$characters))
screening_summary <- bind_rows(lapply(aggregated, function(x) x$screening))
paired_characters <- all_characters %>% filter(included)

# ---- Descriptive statistics on the original measurement scales ---------------

# Each character-word pair receives equal weight, irrespective of trial count.
# Raw TOP proportions include 0 and 1; latency uses arithmetic means in ms.
descriptive_statistics <- bind_rows(lapply(DATASET_ORDER, function(dataset_name) {
  describe_dataset(all_characters, dataset_name)
}))

# ---- Correlation analysis ---------------------------------------------------

# Dataset order and seed fix the sequence of bootstrap draws.
do.call(RNGkind, RNG_SETTINGS)
set.seed(RANDOM_SEED)
correlation_results <- bind_rows(lapply(DATASET_ORDER, function(dataset_name) {
  message("Correlations and bootstrap CI: ", dataset_name)
  values <- paired_characters %>% filter(dataset == dataset_name)
  bind_rows(
    correlation_row(values$logit_top, values$mean_log_rt, dataset_name, "logit_top", "pearson"),
    correlation_row(values$logit_top, values$mean_log_rt, dataset_name, "logit_top", "spearman"),
    correlation_row(values$top_rate, values$mean_log_rt, dataset_name, "raw_top_rate", "pearson")
  )
}))
main_results <- correlation_results %>% filter(top_measure == "logit_top")
for (dataset_name in DATASET_ORDER) {
  pearson <- main_results %>% filter(dataset == dataset_name, method == "pearson")
  spearman <- main_results %>% filter(dataset == dataset_name, method == "spearman")
  if (pearson$n != REFERENCE_N[[dataset_name]] ||
      abs(pearson$estimate - REFERENCE_PEARSON[[dataset_name]]) > REFERENCE_TOLERANCE ||
      abs(spearman$estimate - REFERENCE_SPEARMAN[[dataset_name]]) > REFERENCE_TOLERANCE) {
    warning(dataset_name, ": results differ from the initial check. Inspect the audit CSVs.")
  }
}

# ---- Create runtime output directories --------------------------------------
for (directory in unique(dirname(output_paths))) {
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
}

# ---- Audit data and run information -----------------------------------------

write.csv(all_characters, output_paths[["characters"]], row.names = FALSE, fileEncoding = "UTF-8")
write.csv(descriptive_statistics, output_paths[["descriptive"]], row.names = FALSE)
write.csv(correlation_results, output_paths[["correlations"]], row.names = FALSE)
write.csv(screening_summary, output_paths[["screening"]], row.names = FALSE)
configuration <- list(
  project_root = PROJECT_ROOT, input_paths = input_paths, output_paths = output_paths,
  input_md5 = tools::md5sum(input_paths), dataset_order = DATASET_ORDER,
  rt_max_ms = RT_MAX_MS, bootstrap_replicates = BOOTSTRAP_REPLICATES,
  random_seed = RANDOM_SEED, rng_kind = RNGkind(), confidence_level = CONFIDENCE_LEVEL,
  zero_fraction = ZERO_FRACTION, one_gap_fraction = ONE_GAP_FRACTION,
  coefficient_digits = COEFFICIENT_DIGITS, p_digits = P_DIGITS, p_threshold = P_THRESHOLD,
  top_descriptive_digits = TOP_DESCRIPTIVE_DIGITS,
  latency_descriptive_digits = LATENCY_DESCRIPTIVE_DIGITS
)
writeLines(capture.output(dput(configuration)), output_paths[["configuration"]])
writeLines(capture.output(sessionInfo()), output_paths[["session"]])
print(main_results)
message("Finished. Outputs:\n", paste(output_paths, collapse = "\n"))
