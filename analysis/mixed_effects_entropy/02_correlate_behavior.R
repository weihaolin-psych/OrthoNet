#!/usr/bin/env Rscript
# Correlate model entropy with context-matched human retrieval difficulty.
# Run: Rscript --vanilla analysis/mixed_effects_entropy/02_correlate_behavior.R
# Reads canonical trials and writes context summaries and Pearson correlations.
# No plotting or mixed-effects fitting.

# ---- Configuration ----------------------------------------------------------
REQUIRED_PACKAGES <- c("here", "dplyr")
missing <- REQUIRED_PACKAGES[!vapply(REQUIRED_PACKAGES, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("Missing packages: ", paste(missing, collapse = ", "), ". Install manually.")
library(dplyr)
here <- here::here
DATA_FILES <- list(
  Huang = here("data", "processed", "huang_trials_with_entropy.csv"),
  `Wang Small` = here("data", "processed", "wang_small_trials_with_entropy.csv"),
  `Wang Full` = here("data", "processed", "wang_full_trials_with_entropy.csv"))
OUTPUT_DIR <- here("analysis", "mixed_effects_entropy", "output")
RT_CUTOFF_MS <- 10000
CONFIDENCE_LEVEL <- 0.95

# ---- Average trials within their designated Character-Word context ---------
aggregate_characters <- function(trial_file, dataset_name) {
  df <- read.csv(trial_file)
  
  top_agg <- df %>%
    filter(as.logical(top_eligible)) %>%
    group_by(Character, Word) %>%
    summarise(top_rate = mean(top_binary, na.rm = TRUE),
              entropy  = first(entropy), .groups = "drop")
  
  rt_agg <- df %>%
    filter(as.logical(rt_eligible), rt <= RT_CUTOFF_MS) %>%
    group_by(Character, Word) %>%
    summarise(mean_log_rt = mean(log_rt, na.rm = TRUE), .groups = "drop")
  
  top_agg %>%
    left_join(rt_agg, by = c("Character", "Word")) %>%
    filter(!is.na(entropy)) %>%
    mutate(dataset = factor(dataset_name,
                            levels = c("Huang", "Wang Small", "Wang Full")))
}

logit_transform_vec <- function(p) {
  # Replace exact 0s with half the smallest non-zero value in this group
  # Replace exact 1s with midpoint between largest non-one value and 1
  p_nonzero <- p[p > 0 & !is.na(p)]
  p_nonone  <- p[p < 1 & !is.na(p)]
  floor_val <- min(p_nonzero) / 2
  ceil_val  <- (max(p_nonone) + 1) / 2
  p_adj <- ifelse(p == 0, floor_val, ifelse(p == 1, ceil_val, p))
  log(p_adj / (1 - p_adj))
}


# ---- Apply the original dataset-specific endpoint adjustment ----------------
data <- bind_rows(lapply(names(DATA_FILES), function(dataset)
  aggregate_characters(DATA_FILES[[dataset]], dataset)))
data <- data %>% group_by(dataset) %>%
  mutate(logit_top = ifelse(!is.na(top_rate), logit_transform_vec(top_rate), NA)) %>% ungroup()

# ---- Report associations on exactly the original complete pairs -------------
rows <- list()
for (dataset in names(DATA_FILES)) {
  for (outcome in c("logit_top", "mean_log_rt")) {
    values <- data[data$dataset == dataset, ]
    keep <- complete.cases(values[c("entropy", outcome)])
    test <- cor.test(values$entropy[keep], values[[outcome]][keep], method = "pearson",
                     conf.level = CONFIDENCE_LEVEL)
    rows[[length(rows) + 1L]] <- data.frame(dataset = dataset, outcome = outcome,
      n = sum(keep), r = unname(test$estimate), p = test$p.value,
      ci_lower = test$conf.int[1], ci_upper = test$conf.int[2])
  }
}
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
write.csv(data, file.path(OUTPUT_DIR, "behavior_context_summaries.csv"), row.names = FALSE)
write.csv(bind_rows(rows), file.path(OUTPUT_DIR, "behavior_correlations.csv"), row.names = FALSE)
