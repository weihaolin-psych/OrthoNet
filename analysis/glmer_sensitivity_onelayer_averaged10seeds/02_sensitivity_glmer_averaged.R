# ============================================================================
# 02_sensitivity_glmer_averaged.R
#
# Run mixed-effects models across all (hidden_dim, epoch) combinations
# using averaged entropy (mean across 10 random seeds).
#
# DVs: TOP rate (glmer, binomial) and Writing Latency (lmer, log_rt)
#
# Input:
#   - huang_trials_averaged.csv
#   - wang_small_trials_averaged.csv
#
# Output:
#   - sensitivity_glmer_results_averaged.csv
#
# Usage:
#   Rscript 02_sensitivity_glmer_averaged.R
# ============================================================================

library(lme4)
library(lmerTest)
library(dplyr)

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_DIR <- file.path("data", "analysis_inputs", "glmer_sensitivity_onelayer_averaged10seeds")
OUTPUT_DIR <- file.path("analysis", "glmer_sensitivity_onelayer_averaged10seeds", "output")

TRIAL_FILES <- list(
  Huang      = file.path(DATA_DIR, "huang_trials_averaged.csv"),
  Wang_Small = file.path(DATA_DIR, "wang_small_trials_averaged.csv")
)

DVS <- c("top", "rt")

GLMER_CONTROL <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
LMER_CONTROL  <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

# ── Data preparation ──────────────────────────────────────────────────────────

prepare_data <- function(df, dv, entropy_col) {
  # Filter by DV eligibility
  if (dv == "top") {
    df <- df[as.logical(df$top_eligible), ]
  } else {
    df <- df[as.logical(df$rt_eligible) & df$rt <= 10000, ]
  }

  # Center covariates
  for (v in c("log_freq", "aoa", "context_familiarity", "regularity", "n_stroke")) {
    if (v %in% names(df))
      df[[paste0(v, "_c")]] <- df[[v]] - mean(df[[v]], na.rm = TRUE)
  }

  # Center entropy
  df$entropy_c <- df[[entropy_col]] - mean(df[[entropy_col]], na.rm = TRUE)

  # Contrast-code radical order
  df$radical_order_c <- ifelse(df$phonetic_radical_order == 1, 0.5, -0.5)

  df$subject <- as.factor(df$subject)
  df$item    <- as.factor(df$item)

  # Drop incomplete cases
  required <- c("log_freq_c", "aoa_c", "context_familiarity_c",
                "radical_order_c", "regularity_c", "n_stroke_c", "entropy_c")
  if (dv == "rt") required <- c("log_rt", required)
  df <- df[complete.cases(df[, required]), ]

  return(df)
}

# ── Model fitting ─────────────────────────────────────────────────────────────

fit_base_model <- function(df, dv) {
  base_formula <- if (dv == "top") {
    top_binary ~ log_freq_c + aoa_c + context_familiarity_c +
      radical_order_c + regularity_c + n_stroke_c +
      (1 | subject) + (1 | item)
  } else {
    log_rt ~ log_freq_c + aoa_c + context_familiarity_c +
      radical_order_c + regularity_c + n_stroke_c +
      (1 | subject) + (1 | item)
  }

  if (dv == "top") {
    glmer(base_formula, data = df, family = binomial, control = GLMER_CONTROL)
  } else {
    lmer(base_formula, data = df, REML = FALSE, control = LMER_CONTROL)
  }
}

fit_entropy_model <- function(df, dv) {
  entropy_formula <- if (dv == "top") {
    top_binary ~ log_freq_c + aoa_c + context_familiarity_c +
      radical_order_c + regularity_c + n_stroke_c +
      entropy_c +
      (1 | subject) + (1 | item)
  } else {
    log_rt ~ log_freq_c + aoa_c + context_familiarity_c +
      radical_order_c + regularity_c + n_stroke_c +
      entropy_c +
      (1 | subject) + (1 | item)
  }

  if (dv == "top") {
    glmer(entropy_formula, data = df, family = binomial, control = GLMER_CONTROL)
  } else {
    lmer(entropy_formula, data = df, REML = FALSE, control = LMER_CONTROL)
  }
}

# ── Extract results ───────────────────────────────────────────────────────────

extract_entropy_results <- function(entropy_model, base_model, dv) {
  s <- summary(entropy_model)$coefficients
  stat_col <- if (dv == "top") "z value" else "t value"
  p_col    <- if (dv == "top") "Pr(>|z|)" else "Pr(>|t|)"

  ent_row <- s["entropy_c", ]
  beta <- ent_row["Estimate"]
  se   <- ent_row["Std. Error"]
  stat <- ent_row[stat_col]
  p    <- ent_row[p_col]

  # Confidence interval (Wald method)
  ci <- confint(entropy_model, parm = "entropy_c", method = "Wald")
  ci_lower <- ci[1]
  ci_upper <- ci[2]

  # LRT
  lrt <- anova(base_model, entropy_model)
  lrt_chisq <- lrt$Chisq[2]
  lrt_p     <- lrt$`Pr(>Chisq)`[2]

  data.frame(
    beta = beta, se = se, stat = stat, p = p,
    ci_lower = ci_lower, ci_upper = ci_upper,
    lrt_chisq = lrt_chisq, lrt_p = lrt_p,
    stringsAsFactors = FALSE, row.names = NULL
  )
}

# ── Main ──────────────────────────────────────────────────────────────────────

cat("=== Sensitivity Analysis: Averaged Entropy ===\n\n")

all_results <- list()

for (dataset_name in names(TRIAL_FILES)) {

  cat(sprintf("\nLoading %s...\n", dataset_name))
  # Prefer newly extracted trials when explicitly rebuilding from evaluations.
  rebuilt <- file.path(OUTPUT_DIR, basename(TRIAL_FILES[[dataset_name]]))
  input <- if (file.exists(rebuilt)) rebuilt else TRIAL_FILES[[dataset_name]]
  df_raw <- read.csv(input)

  entropy_cols <- grep("^entropy_h\\d+_ep\\d+$", names(df_raw), value = TRUE)
  cat(sprintf("  Found %d entropy columns\n", length(entropy_cols)))

  param_grid <- data.frame(
    col_name = entropy_cols,
    hidden_dim = as.integer(gsub(".*_h(\\d+)_ep.*", "\\1", entropy_cols)),
    epoch      = as.integer(gsub(".*_ep(\\d+)$", "\\1", entropy_cols)),
    stringsAsFactors = FALSE
  )
  param_grid <- param_grid[order(param_grid$hidden_dim, param_grid$epoch), ]

  for (dv in DVS) {

    cat(sprintf("\n  %s | DV: %s | %d combinations\n",
                dataset_name, toupper(dv), nrow(param_grid)))

    for (i in seq_len(nrow(param_grid))) {
      h   <- param_grid$hidden_dim[i]
      ep  <- param_grid$epoch[i]
      col <- param_grid$col_name[i]

      cat(sprintf("    [%3d/%d] h=%2d, ep=%2d, DV=%s ... ",
                  i, nrow(param_grid), h, ep, toupper(dv)))

      tryCatch({
        df_prep <- prepare_data(df_raw, dv, col)

        base_model    <- fit_base_model(df_prep, dv)
        entropy_model <- fit_entropy_model(df_prep, dv)

        res <- extract_entropy_results(entropy_model, base_model, dv)

        sig <- if (res$p < .001) "***" else if (res$p < .01) " **" else if (res$p < .05) "  *" else "   "
        ci_includes_zero <- res$ci_lower <= 0 & res$ci_upper >= 0
        cat(sprintf("beta=%.3f [%.3f, %.3f]%s p=%.4f | LRT chi2=%.2f p=%.4f\n",
                    res$beta, res$ci_lower, res$ci_upper,
                    ifelse(ci_includes_zero, " (CI incl 0)", ""),
                    res$p, res$lrt_chisq, res$lrt_p))

        all_results[[length(all_results) + 1]] <- data.frame(
          dataset = dataset_name, dv = toupper(dv),
          hidden_dim = h, epoch = ep,
          beta = res$beta, se = res$se,
          ci_lower = res$ci_lower, ci_upper = res$ci_upper,
          stat = res$stat, p = res$p,
          lrt_chisq = res$lrt_chisq, lrt_p = res$lrt_p,
          sig = res$p < .05,
          ci_excludes_zero = !(res$ci_lower <= 0 & res$ci_upper >= 0),
          lrt_sig = res$lrt_p < .05,
          stringsAsFactors = FALSE
        )

      }, error = function(e) {
        cat(sprintf("ERROR: %s\n", e$message))
        all_results[[length(all_results) + 1]] <<- data.frame(
          dataset = dataset_name, dv = toupper(dv),
          hidden_dim = h, epoch = ep,
          beta = NA, se = NA,
          ci_lower = NA, ci_upper = NA,
          stat = NA, p = NA,
          lrt_chisq = NA, lrt_p = NA,
          sig = NA, ci_excludes_zero = NA, lrt_sig = NA,
          stringsAsFactors = FALSE
        )
      })
    }
  }
}

# ── Save results ──
df_results <- bind_rows(all_results)

out_csv <- file.path(OUTPUT_DIR, "sensitivity_glmer_results_averaged.csv")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
write.csv(df_results, out_csv, row.names = FALSE)
cat(sprintf("\n✓ Results saved to %s\n", out_csv))

# ── Print summary ──
cat(sprintf("\n%s\n", strrep("=", 60)))
cat("  SUMMARY (Averaged Entropy)\n")
cat(sprintf("%s\n", strrep("=", 60)))

for (ds in unique(df_results$dataset)) {
  for (dv in unique(df_results$dv)) {
    sub <- df_results[df_results$dataset == ds & df_results$dv == dv, ]
    n_total <- nrow(sub)
    n_sig   <- sum(sub$sig, na.rm = TRUE)
    n_ci    <- sum(sub$ci_excludes_zero, na.rm = TRUE)
    n_pos   <- sum(sub$beta > 0, na.rm = TRUE)

    cat(sprintf("\n  %s | %s:\n", ds, dv))
    cat(sprintf("    Total combinations:        %d\n", n_total))
    cat(sprintf("    p < .05:                   %d / %d (%.0f%%)\n",
                n_sig, n_total, 100 * n_sig / n_total))
    cat(sprintf("    CI excludes zero:          %d / %d (%.0f%%)\n",
                n_ci, n_total, 100 * n_ci / n_total))
    cat(sprintf("    Positive beta:             %d / %d (%.0f%%)\n",
                n_pos, n_total, 100 * n_pos / n_total))
    cat(sprintf("    Beta range:                [%.3f, %.3f]\n",
                min(sub$beta, na.rm = TRUE), max(sub$beta, na.rm = TRUE)))
  }
}

cat("\nDone!\n")
