#!/usr/bin/env Rscript
# Fit lexical and entropy mixed-effects models of human retrieval difficulty.
# Run from the repository root: Rscript analysis/mixed_effects_entropy/01_fit_models.R
# Reads data/processed/*_trials_with_entropy.csv; saves CSV summaries and RDS
# models in output/. Run the separate manuscript table script after completion.

# ---- Configuration ----------------------------------------------------------
REQUIRED_PACKAGES <- c("here", "lme4", "lmerTest", "dplyr", "performance")
missing_packages <- REQUIRED_PACKAGES[!vapply(REQUIRED_PACKAGES, requireNamespace,
                                            logical(1), quietly = TRUE)]
if (length(missing_packages)) stop("Missing R packages: ",
  paste(missing_packages, collapse = ", "), ". Install them before running; nothing was installed.")
library(dplyr)
library(lme4)
library(lmerTest)
DATA_FILES <- setNames(
  file.path(here::here("data", "processed"),
            paste0(c("huang", "wang_small", "wang_full"), "_trials_with_entropy.csv")),
  c("huang", "wang_small", "wang_full"))
OUTPUT_DIR <- here::here("analysis", "mixed_effects_entropy", "output")
RT_MAX_MS <- 10000
OPTIMIZER <- "bobyqa"
MAX_FUNCTION_EVALUATIONS <- 2e5
SINGULARITY_TOLERANCE <- 1e-4
R2_TOLERANCE <- 1e-8
GLMER_CONTROL <- glmerControl(optimizer = OPTIMIZER,
                             optCtrl = list(maxfun = MAX_FUNCTION_EVALUATIONS))
LMER_CONTROL <- lmerControl(optimizer = OPTIMIZER,
                           optCtrl = list(maxfun = MAX_FUNCTION_EVALUATIONS))
FIXED_TERMS <- c("log_freq_c", "aoa_c", "context_familiarity_c",
                 "radical_order_c", "regularity_c", "n_stroke_c")

# ---- Data preparation -------------------------------------------------------
# Preserve centering before complete-case filtering, as in the original analysis.
prepare_data <- function(df, dv = c("top", "rt")) {
  dv <- match.arg(dv)

  if (dv == "top") {
    df <- df %>% filter(as.logical(top_eligible))
  } else {
    df <- df %>% filter(as.logical(rt_eligible), rt <= RT_MAX_MS)
  }

  # Center numeric predictors
  for (v in c("log_freq", "aoa", "context_familiarity",
              "regularity", "n_stroke", "entropy")) {
    if (v %in% names(df))
      df[[paste0(v, "_c")]] <- df[[v]] - mean(df[[v]], na.rm = TRUE)
  }

  # Contrast-code phonetic radical order (±0.5)
  df$radical_order_c <- ifelse(df$phonetic_radical_order == 1, 0.5, -0.5)

  df$subject <- as.factor(df$subject)
  df$item    <- as.factor(df$item)

  # Drop rows with missing predictors
  required <- c("log_freq_c", "aoa_c", "context_familiarity_c",
                "radical_order_c", "regularity_c", "n_stroke_c", "entropy_c")
  if (dv == "rt") required <- c("log_rt", required)
  df <- df %>% filter(if_all(all_of(required), ~ !is.na(.)))

  return(df)
}


# ---- Fit and retain diagnostics without changing model specifications --------
capture_conditions <- function(expr) {
  warnings <- messages <- character()
  error <- ""
  value <- tryCatch(withCallingHandlers(expr,
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    },
    message = function(m) {
      messages <<- c(messages, conditionMessage(m))
      invokeRestart("muffleMessage")
    }), error = function(e) { error <<- conditionMessage(e); NULL })
  list(value = value, warnings = paste(unique(warnings), collapse = " | "),
       messages = paste(unique(messages), collapse = " | "), error = error)
}

fit_one_model <- function(df, dv, model_name) {
  terms <- c(FIXED_TERMS, if (model_name == "entropy") "entropy_c",
             "(1 | subject)", "(1 | item)")
  response <- if (dv == "top") "top_binary" else "log_rt"
  formula <- as.formula(paste(response, "~", paste(terms, collapse = " + ")))
  if (dv == "top") {
    glmer(formula, data = df, family = binomial, control = GLMER_CONTROL)
  } else {
    lmer(formula, data = df, REML = FALSE, control = LMER_CONTROL)
  }
}

extract_coefficients <- function(model, dv) {
  s <- summary(model)$coefficients
  data.frame(term = rownames(s), beta = s[, "Estimate"],
    se = s[, "Std. Error"],
    stat = s[, if (dv == "top") "z value" else "t value"],
    statistic_type = if (dv == "top") "z" else "t",
    df = if (dv == "rt") s[, "df"] else NA_real_,
    p = s[, if (dv == "top") "Pr(>|z|)" else "Pr(>|t|)"], row.names = NULL)
}

extract_r2 <- function(model) {
  # Binomial models use the theoretical distribution-specific variance by default.
  result <- capture_conditions(performance::r2_nakagawa(
    model, tolerance = R2_TOLERANCE, ci = NULL, verbose = TRUE))
  get_value <- function(name) {
    value <- if (is.list(result$value)) result$value[[name]] else NULL
    if (length(value) == 1L && is.finite(value)) as.numeric(value) else NA_real_
  }
  marginal <- get_value("R2_marginal")
  conditional <- get_value("R2_conditional")
  data.frame(r2_marginal = marginal, r2_conditional = conditional,
    r2_status = if (nzchar(result$error)) "error" else if (
      anyNA(c(marginal, conditional))) "unavailable" else "ok",
    r2_warnings = result$warnings, r2_messages = result$messages,
    r2_error = result$error)
}

extract_model_summary <- function(model, fit, r2) {
  convergence_messages <- paste(unlist(model@optinfo$conv$lme4$messages), collapse = " | ")
  optimizer_code <- model@optinfo$conv$opt
  frame <- model.frame(model)
  cbind(data.frame(
    n_obs = nobs(model), n_subjects = length(unique(frame$subject)),
    n_items = length(unique(frame$item)), log_likelihood = as.numeric(logLik(model)),
    aic = AIC(model), bic = BIC(model),
    converged = if (is.null(optimizer_code)) NA else
      all(optimizer_code == 0) && !nzchar(convergence_messages),
    optimizer_code = paste(optimizer_code, collapse = ","),
    convergence_messages = convergence_messages,
    singular = isSingular(model, tol = SINGULARITY_TOLERANCE),
    fit_warnings = fit$warnings, fit_messages = fit$messages), r2)
}

# ---- Run analysis and save reusable results ---------------------------------
main <- function() {
  missing_files <- DATA_FILES[!file.exists(DATA_FILES)]
  if (length(missing_files)) stop("Missing input files: ", paste(missing_files, collapse = ", "))
  dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
  # A shared identifier prevents the table from combining different analysis runs.
  run_id <- format(Sys.time(), "%Y%m%dT%H%M%OS6")
  coefficients <- summaries <- random_effects <- comparisons <- list()
  diagnostic_file <- file.path(OUTPUT_DIR, "diagnostics.log")
  writeLines(paste("Analysis run", run_id), diagnostic_file)
  add_keys <- function(x, dataset, dv, model_name = NULL) {
    keys <- data.frame(run_id = run_id, dataset = dataset, dv = dv)
    if (!is.null(model_name)) keys$model <- model_name
    cbind(keys[rep(1L, nrow(x)), , drop = FALSE], x, row.names = NULL)
  }
  for (dataset in names(DATA_FILES)) {
    raw <- read.csv(DATA_FILES[[dataset]])
    for (dv in c("top", "rt")) {
      df <- prepare_data(raw, dv)
      models <- pair_summaries <- list()
      for (model_name in c("base", "entropy")) {
        key <- paste(dataset, dv, model_name, sep = "_")
        message("Fitting ", key)
        fit <- capture_conditions(fit_one_model(df, dv, model_name))
        cat(key, fit$warnings, fit$messages, fit$error, "\n",
            file = diagnostic_file, append = TRUE)
        if (is.null(fit$value)) stop("Model fitting failed: ", key, ". ", fit$error,
          " See diagnostics.log; previously completed RDS models are retained.")
        model <- fit$value
        # Save immediately: later extraction failures must not discard an expensive fit.
        saveRDS(model, file.path(OUTPUT_DIR, paste0(key, ".rds")))
        models[[model_name]] <- model
        r2 <- extract_r2(model)
        summary_row <- extract_model_summary(model, fit, r2)
        cat(key, r2$r2_status, r2$r2_warnings, r2$r2_messages, r2$r2_error, "\n",
            file = diagnostic_file, append = TRUE)
        if (nzchar(fit$warnings) || nzchar(fit$messages) ||
            r2$r2_status != "ok" || nzchar(r2$r2_warnings)) {
          message("  Diagnostics recorded for ", key, "; inspect model_summary.csv and diagnostics.log.")
        }
        coefficients[[key]] <- add_keys(extract_coefficients(model, dv), dataset, dv, model_name)
        summaries[[key]] <- add_keys(summary_row, dataset, dv, model_name)
        random_effects[[key]] <- add_keys(as.data.frame(VarCorr(model)), dataset, dv, model_name)
        pair_summaries[[model_name]] <- summary_row
      }
      # Identical row identities, not just counts, are required for nested comparison.
      if (!identical(rownames(model.frame(models$base)),
                     rownames(model.frame(models$entropy)))) {
        stop("Base and entropy model samples differ: ", dataset, " / ", dv)
      }
      comparison <- capture_conditions(anova(models$base, models$entropy, refit = FALSE))
      cat(dataset, dv, comparison$warnings, comparison$messages, comparison$error,
          "\n", file = diagnostic_file, append = TRUE)
      if (is.null(comparison$value)) stop("LRT failed: ", comparison$error)
      lrt <- comparison$value
      comparisons[[paste(dataset, dv, sep = "_")]] <- add_keys(data.frame(
        base_model = "base", added_model = "entropy", n_obs = nobs(models$base),
        chisq = lrt$Chisq[2], df = lrt$Df[2], p = lrt$`Pr(>Chisq)`[2],
        delta_r2_marginal = pair_summaries$entropy$r2_marginal - pair_summaries$base$r2_marginal,
        lrt_warnings = comparison$warnings, lrt_messages = comparison$messages), dataset, dv)
    }
  }
  outputs <- list(coefficients = bind_rows(coefficients), model_summary = bind_rows(summaries),
                  random_effects = bind_rows(random_effects), model_comparisons = bind_rows(comparisons))
  stopifnot(nrow(outputs$model_summary) == 12L, nrow(outputs$model_comparisons) == 6L)
  for (name in names(outputs)) {
    write.csv(outputs[[name]], file.path(OUTPUT_DIR, paste0(name, ".csv")), row.names = FALSE)
  }
  # Provenance supports later checks against the input data and package versions.
  write.csv(data.frame(run_id = run_id, dataset = names(DATA_FILES),
                      input = unname(DATA_FILES), md5 = unname(tools::md5sum(DATA_FILES))),
            file.path(OUTPUT_DIR, "input_manifest.csv"), row.names = FALSE)
  writeLines(c(paste("Run:", run_id), capture.output(sessionInfo())),
             file.path(OUTPUT_DIR, "session_info.txt"))
  message("Saved 12 models and 6 comparisons to ", OUTPUT_DIR,
          ". Inspect diagnostics before generating the table.")
}

main()
