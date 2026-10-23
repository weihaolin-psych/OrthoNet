#!/usr/bin/env Rscript
# ============================================================================
# 01_classify_and_rank.R
#
# Purpose: Classify every character into one of four model-human types using a
#          MEDIAN split on entropy (model difficulty) and TOP rate (human
#          difficulty), then SELECT the K characters per type that sit most
#          clearly in their quadrant, using a joint top-K-percentile rule.
#
#   Q1  both easy               (model easy, human easy)
#   Q2  model easy, human hard  <- under-predicted (the interesting one)
#   Q3  model hard, human easy  <- over-predicted
#   Q4  both hard
#
# Selection rule (replaces the old z-score SUM ranking):
#   The median split fixes the SIGN (which quadrant a character is in). Within a
#   quadrant, score each character by how far it sits into BOTH extremes. Convert
#   each axis to a percentile, take it in the direction the quadrant cares about
#   (hard side uses p, easy side uses 1 - p), and score by the MIN of the two
#   directional percentiles. The min is the binding axis, i.e. "must be in the
#   top-K% on BOTH measures". Rank by that score (high = clearly in the quadrant)
#   and keep the top K. Because the quadrant gate already enforces the sign, a
#   lopsided point cannot leak across a boundary, and within a quadrant the min
#   correctly ranks balanced exemplars above one-sided ones.
#
# Baseline for the later tests is "selected K vs the rest": for each quadrant Q,
# script 02 compares (selected & type == Q) against every OTHER character in the
# dataset. This script only carries the `selected` flag, the `type`, and the
# covariates; the t-tests live in 02.
#
# Input : *_trials_with_entropy.csv  (averaged optimal entropy already merged)
# Output: classified_<dataset>.csv   (one row per character; all rows carried)
# ============================================================================

library(here)
library(dplyr)

# -- Configuration -----------------------------------------------------------
DATA_FILES <- list(
  huang      = here("data", "processed", "huang_trials_with_entropy.csv"),
  wang_small = here("data", "processed", "wang_small_trials_with_entropy.csv"),
  wang_full  = here("data", "processed", "wang_full_trials_with_entropy.csv")
)

OUTPUT_DIR <- here("analysis", "error_analysis")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Lexical variables carried through for the later "selected vs rest" tests
LEX_VARS <- c("log_freq", "aoa", "context_familiarity",
              "phonetic_radical_order", "regularity", "n_stroke")

K       <- 30   # characters selected per quadrant for the tests
TOP_FIG <- 10   # of the selected, how many to label in the figure

# -- Aggregate trials -> character level -------------------------------------
aggregate_to_character <- function(trial_file) {
  df <- read.csv(trial_file)
  df$top_eligible <- as.logical(df$top_eligible)

  top_agg <- df %>%
    filter(top_eligible == TRUE) %>%
    group_by(Character, Word) %>%
    summarise(top_rate = mean(top_binary), n_top = n(), .groups = "drop")

  props <- df %>%
    distinct(Character, Word, .keep_all = TRUE) %>%
    select(Character, Word, entropy, all_of(LEX_VARS))

  top_agg %>%
    left_join(props, by = c("Character", "Word")) %>%
    filter(!is.na(entropy), !is.na(top_rate))
}

# -- Median-split classification + joint top-K selection ---------------------
classify_and_select <- function(char_df, k = K) {
  n <- nrow(char_df)

  df <- char_df %>%
    mutate(
      z_entropy  = as.numeric(scale(entropy)),
      z_top_rate = as.numeric(scale(top_rate)),
      # Percentiles on each axis; average ties so the Wang Full TOP = 0 pile
      # shares one human percentile rather than getting an arbitrary order.
      p_model = rank(entropy,  ties.method = "average") / n,  # high = model hard
      p_human = rank(top_rate, ties.method = "average") / n   # high = human hard
    )

  med_e <- median(df$z_entropy)
  med_t <- median(df$z_top_rate)

  # Sign / quadrant from a strict median split. ">" for hard and "<=" for easy
  # keeps a clean partition and parks the TOP = 0 tie mass on the easy side.
  df <- df %>%
    mutate(
      model_hard = z_entropy  > med_e,
      human_hard = z_top_rate > med_t,
      type = case_when(
        !model_hard & !human_hard ~ "Q1",
        !model_hard &  human_hard ~ "Q2",
         model_hard & !human_hard ~ "Q3",
         model_hard &  human_hard ~ "Q4"
      ),
      # Directional percentile per axis (all land in [.5, 1] inside a quadrant),
      # then the joint score is the binding (min) axis. Higher = clearer member.
      dir_model   = if_else(model_hard, p_model, 1 - p_model),
      dir_human   = if_else(human_hard, p_human, 1 - p_human),
      joint_score = pmin(dir_model, dir_human)
    ) %>%
    group_by(type) %>%
    arrange(desc(joint_score), .by_group = TRUE) %>%
    mutate(
      rank_in_type = row_number(),
      selected     = rank_in_type <= k,      # caps automatically if quadrant < k
      top_fig      = rank_in_type <= TOP_FIG
    ) %>%
    ungroup()

  df
}

TYPE_LABEL <- c(Q1 = "Q1 both easy",
                Q2 = "Q2 model easy, human hard",
                Q3 = "Q3 model hard, human easy",
                Q4 = "Q4 both hard")

# -- MAIN --------------------------------------------------------------------
for (dataset in names(DATA_FILES)) {
  f <- DATA_FILES[[dataset]]
  if (!file.exists(f)) { cat("Skipping", dataset, "- not found:", f, "\n"); next }

  char_df <- aggregate_to_character(f)
  cls <- classify_and_select(char_df)

  cat("\n==============================================\n")
  cat(" ", dataset, "  (", nrow(cls), "characters )\n")
  cat("==============================================\n")

  # Quadrant sizes: confirm each cell can supply K before trusting selection.
  sizes <- cls %>% count(type)
  cat("  Quadrant sizes (need >= ", K, " each):\n", sep = "")
  short <- FALSE
  for (q in c("Q1","Q2","Q3","Q4")) {
    nq <- sizes$n[sizes$type == q]; if (length(nq) == 0) nq <- 0
    flag <- if (nq < K) { short <- TRUE; "  << below K" } else ""
    cat(sprintf("    %-28s n = %3d%s\n", TYPE_LABEL[q], nq, flag))
  }
  if (short) {
    cat("  WARNING: a quadrant has fewer than K =", K, "characters.\n")
    cat("           Those cells select all available rows. Consider lowering K\n")
    cat("           globally so every cell uses the same n before reporting.\n")
  }

  # Eyeball the figure set (top TOP_FIG of each quadrant).
  for (q in c("Q1","Q2","Q3","Q4")) {
    cat("\n  --", TYPE_LABEL[q], "(top", TOP_FIG, "of", K, "selected ) --\n")
    cls %>%
      filter(type == q, top_fig) %>%
      mutate(across(c(entropy, top_rate, p_model, p_human, joint_score),
                    ~ round(.x, 3))) %>%
      select(rank_in_type, Character, Word, entropy, top_rate,
             p_model, p_human, joint_score) %>%
      as.data.frame() %>%
      print(row.names = FALSE)
  }

  # Save EVERY character: script 02 needs the unselected rows as "the rest".
  out <- cls %>%
    arrange(type, rank_in_type) %>%
    select(Character, Word, type, rank_in_type, selected, top_fig,
           entropy, top_rate, z_entropy, z_top_rate,
           p_model, p_human, joint_score,
           all_of(LEX_VARS))
  out_path <- file.path(OUTPUT_DIR, paste0("classified_", dataset, ".csv"))
  write.csv(out, out_path, row.names = FALSE)
  cat("\n  -> saved", out_path, "\n")
}

cat("\nDone. In 02, for each quadrant Q compare (selected & type == Q) against",
    "\nall other characters in the dataset. Use top_fig == TRUE for figure labels.\n")
