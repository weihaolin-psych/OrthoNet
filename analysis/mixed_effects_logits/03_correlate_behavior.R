#!/usr/bin/env Rscript
# Correlate relative target support with context-matched human difficulty.
# Run: Rscript --vanilla analysis/mixed_effects_logits/03_correlate_behavior.R
# Reads prepared trials; writes context summaries and Pearson correlations only.

# ---- Configuration ----------------------------------------------------------
REQUIRED_PACKAGES <- c("here", "dplyr")
missing <- REQUIRED_PACKAGES[!vapply(REQUIRED_PACKAGES, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("Missing packages: ", paste(missing, collapse = ", "), ". Install manually.")
library(dplyr)
DATA_FILES <- c(
  Huang = here::here("data", "processed", "huang_trials_with_logit_measures.csv"),
  `Wang Small` = here::here("data", "processed", "wang_small_trials_with_logit_measures.csv"),
  `Wang Full` = here::here("data", "processed", "wang_full_trials_with_logit_measures.csv"))
EXPECTED_CONTEXTS <- c(Huang = 200L, `Wang Small` = 200L, `Wang Full` = 1600L)
EXPECTED_MATCHED <- c(Huang = 200L, `Wang Small` = 200L, `Wang Full` = 1592L)
MEASURES <- c("deficit_measure", "blocking_measure")
RT_CUTOFF_MS <- 10000
CONFIDENCE_LEVEL <- 0.95
OUTPUT_DIR <- here::here("analysis", "mixed_effects_logits", "output_standardized_measures")

# ---- Verify model scores and aggregate the original trial samples -----------
read_trials <- function(path, dataset_name) {
  if (!file.exists(path)) stop("Missing prepared trials: ", path)
  trials <- read.csv(path, stringsAsFactors = FALSE, fileEncoding = "UTF-8",
                     check.names = FALSE)
  required <- c("Character", "Word", "top_binary", "top_eligible", "rt",
                "log_rt", "rt_eligible", MEASURES)
  absent <- setdiff(required, names(trials))
  if (length(absent)) stop(dataset_name, ": missing columns: ", paste(absent, collapse = ", "))
  if (anyNA(trials[c("Character", "Word")]) ||
      any(!nzchar(trials$Character) | !nzchar(trials$Word))) stop("Missing context keys")
  for (flag in c("top_eligible", "rt_eligible")) {
    trials[[flag]] <- as.logical(trials[[flag]])
    if (anyNA(trials[[flag]])) stop(dataset_name, ": invalid eligibility flag: ", flag)
  }
  if (!identical(is.na(trials[[MEASURES[1]]]), is.na(trials[[MEASURES[2]]]))) {
    stop(dataset_name, ": measures have different missing contexts")
  }
  for (measure in MEASURES) {
    values <- trials[[measure]]
    if (!is.numeric(values) || any(!is.finite(values[!is.na(values)]))) {
      stop(dataset_name, ": invalid model values: ", measure)
    }
  }
  contexts <- trials %>% group_by(Character, Word) %>%
    summarise(across(all_of(MEASURES), ~ n_distinct(.x, na.rm = FALSE)), .groups = "drop")
  if (any(as.matrix(contexts[MEASURES]) != 1L)) {
    stop(dataset_name, ": model scores vary within a Character-Word")
  }
  matched <- trials %>% distinct(Character, Word, across(all_of(MEASURES))) %>%
    filter(!is.na(.data[[MEASURES[1]]]))
  if (nrow(contexts) != EXPECTED_CONTEXTS[[dataset_name]] ||
      nrow(matched) != EXPECTED_MATCHED[[dataset_name]]) stop(dataset_name, ": unexpected coverage")
  top <- trials[trials$top_eligible, ]
  if (anyNA(top$top_binary) || any(!top$top_binary %in% c(0, 1))) {
    stop(dataset_name, ": eligible TOP outcomes must be binary and observed")
  }
  rt <- trials[trials$rt_eligible, ]
  if (any(!is.finite(rt$rt)) || any(rt$rt <= 0) || any(!is.finite(rt$log_rt)) ||
      any(abs(log(rt$rt) - rt$log_rt) > 1e-10)) stop(dataset_name, ": invalid eligible latency")
  message(dataset_name, ": matched contexts ", nrow(matched), "/", nrow(contexts))
  trials
}

# ---- Aggregate behavioral trials within their designated word context --------
aggregate_contexts <- function(trials, dataset_name) {
  top <- trials %>% filter(top_eligible) %>% group_by(Character, Word) %>%
    summarise(n_top = sum(top_binary), n_eligible = n(),
              top_rate = mean(top_binary), across(all_of(MEASURES), first), .groups = "drop")
  if (any(top$n_top < 0 | top$n_top > top$n_eligible) ||
      any(top$top_rate != top$n_top / top$n_eligible)) stop("Invalid TOP aggregation")
  rt <- trials %>% filter(rt_eligible, rt <= RT_CUTOFF_MS) %>%
    group_by(Character, Word) %>%
    summarise(mean_log_rt = mean(log_rt), n_rt = n(), .groups = "drop")
  # Match the entropy workflow: the TOP-eligible context table anchors both panels.
  top %>% left_join(rt, by = c("Character", "Word")) %>%
    filter(!is.na(.data[[MEASURES[1]]])) %>%
    mutate(dataset = factor(dataset_name, levels = names(DATA_FILES)))
}

# ---- Match the entropy figure's dataset-specific endpoint adjustment ---------
logit_transform_vec <- function(p) {
  finite <- p[is.finite(p)]
  if (!length(finite) || any(finite < 0 | finite > 1) ||
      !any(finite > 0) || !any(finite < 1)) {
    stop("TOP rates do not permit the entropy figure's endpoint adjustment")
  }
  floor_value <- min(finite[finite > 0]) / 2
  ceiling_value <- (max(finite[finite < 1]) + 1) / 2
  adjusted <- ifelse(p == 0, floor_value, ifelse(p == 1, ceiling_value, p))
  log(adjusted / (1 - adjusted))
}


# ---- Adjust endpoints within dataset, after the original context screening --
data <- bind_rows(lapply(names(DATA_FILES), function(dataset)
  aggregate_contexts(read_trials(DATA_FILES[[dataset]], dataset), dataset)))
data <- data %>% group_by(dataset) %>%
  mutate(logit_top = logit_transform_vec(top_rate)) %>% ungroup()

# ---- Both relative measures must retain the same contexts for each outcome --
for (outcome in c("logit_top", "mean_log_rt")) {
  keys <- lapply(MEASURES, function(measure) data %>%
    filter(is.finite(.data[[measure]]), is.finite(.data[[outcome]])) %>%
    select(dataset, Character, Word))
  if (!identical(keys[[1]], keys[[2]])) stop("Relative measures retain different contexts")
}

# ---- Save the Pearson estimates used to describe the behavioral relations ---
rows <- list()
for (dataset in names(DATA_FILES)) {
  for (measure in MEASURES) {
    for (outcome in c("logit_top", "mean_log_rt")) {
      values <- data[data$dataset == dataset, ]
      keep <- is.finite(values[[measure]]) & is.finite(values[[outcome]])
      if (sum(keep) < 3L) stop("Insufficient pairs: ", dataset, " / ", outcome)
      test <- cor.test(values[[measure]][keep], values[[outcome]][keep], method = "pearson",
                       conf.level = CONFIDENCE_LEVEL)
      rows[[length(rows) + 1L]] <- data.frame(dataset = dataset, measure = measure,
        outcome = outcome, n = sum(keep), r = unname(test$estimate), p = test$p.value,
        ci_lower = test$conf.int[1], ci_upper = test$conf.int[2])
    }
  }
}
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
write.csv(data, file.path(OUTPUT_DIR, "behavior_context_summaries.csv"), row.names = FALSE)
write.csv(bind_rows(rows), file.path(OUTPUT_DIR, "behavior_correlations.csv"), row.names = FALSE)
