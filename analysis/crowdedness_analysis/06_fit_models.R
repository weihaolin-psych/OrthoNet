#!/usr/bin/env Rscript
# Fit lexical, crowdedness, and entropy models of human retrieval difficulty.
# Run from the repository root: Rscript analysis/crowdedness_analysis/06_fit_models.R
# Reads data/processed/*_trials_with_entropy_crowdedness.csv; writes statistical
# CSVs, RDS models, and diagnostics to output/. See README.md for the table step.

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
            paste0(c("huang", "wang_small", "wang_full"), "_trials_with_entropy_crowdedness.csv")),
  c("huang", "wang_small", "wang_full"))
OUTPUT_DIR <- here::here("analysis", "crowdedness_analysis", "output")
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
                 "radical_order_c", "regularity_c", "n_stroke_c", "log_type_frequency_z")

CROWD_TYPES <- c("raw", "freq")
OUTCOMES <- c("top", "rt")
MODELS <- c("base", "crowd", "crowd_ent")
CROWD_TERMS <- c("log_sem_crowdedness_z", "log_sound_crowdedness_z",
                 "log_sem_crowdedness_z:log_sound_crowdedness_z")
COMPARISONS <- data.frame(base_model = c("base", "crowd"),
                          added_model = c("crowd", "crowd_ent"), df = c(3L, 1L))

# ---- Data preparation: preserve original scaling and screening order --------
prepare_data <- function(df, dv = c("top", "rt"),
                         crowd_type = c("raw", "freq")) {
  dv <- match.arg(dv)
  crowd_type <- match.arg(crowd_type)

  df$.trial_row <- seq_len(nrow(df))

  if (dv == "top") {
    df <- df %>% filter(as.logical(top_eligible))
  } else {
    df <- df %>% filter(as.logical(rt_eligible), rt <= RT_MAX_MS)
  }

  # For freq-weighted: overwrite standard crowdedness columns with _freq versions
  if (crowd_type == "freq") {
    df <- df %>%
      mutate(log_sem_crowdedness   = log_sem_crowdedness_freq,
             log_sound_crowdedness = log_sound_crowdedness_freq)
  }

  # Center numeric predictors
  for (v in c("log_freq", "aoa", "context_familiarity",
              "regularity", "n_stroke", "entropy")) {
    if (v %in% names(df))
      df[[paste0(v, "_c")]] <- df[[v]] - mean(df[[v]], na.rm = TRUE)
  }

  # Z-score crowdedness and type_frequency
  for (v in c("log_sem_crowdedness", "log_sound_crowdedness", "log_type_frequency")) {
    if (v %in% names(df))
      df[[paste0(v, "_z")]] <- as.numeric(scale(df[[v]]))
  }

  df$radical_order_c <- ifelse(df$phonetic_radical_order == 1, 0.5, -0.5)
  df$subject <- as.factor(df$subject)
  df$item    <- as.factor(df$item)

  # Complete cases
  required <- c("log_freq_c", "aoa_c", "context_familiarity_c",
                "radical_order_c", "regularity_c", "n_stroke_c",
                "entropy_c",
                "log_type_frequency_z", "log_sem_crowdedness_z",
                "log_sound_crowdedness_z")
  if (dv == "rt") required <- c("log_rt", required)
  df <- df %>% filter(if_all(all_of(required), ~ !is.na(.)))

  # Keep original trial identities for strict sample checks after model fitting.
  rownames(df) <- as.character(df$.trial_row)
  response <- if (dv == "top") "top_binary" else "log_rt"
  if (!nrow(df) || anyNA(df[, c(response, "subject", "item")]) ||
      any(!is.finite(as.matrix(df[, c(required, response)])))) {
    stop("Empty sample, missing outcome/group, or non-finite model values: ", dv, " / ", crowd_type)
  }
  if (dv == "top" && !all(df$top_binary %in% c(0, 1))) stop("TOP outcome must be binary.")
  return(df)
}

# ---- Model fitting, full coefficients, and diagnostics -----------------------
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
  terms <- c(FIXED_TERMS, if (model_name != "base") CROWD_TERMS,
             if (model_name == "crowd_ent") "entropy_c",
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

# ---- Run analysis and retain reusable results -------------------------------
main <- function() {
  missing_files <- DATA_FILES[!file.exists(DATA_FILES)]
  if (length(missing_files)) stop("Missing input files: ", paste(missing_files, collapse = ", "))
  dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
  run_id <- format(Sys.time(), "%Y%m%dT%H%M%OS6")
  coefficients <- summaries <- random_effects <- comparisons <- list()
  diagnostic_file <- file.path(OUTPUT_DIR, "diagnostics.log")
  writeLines(paste("Analysis run", run_id), diagnostic_file)
  # Save provenance at the start, including when a later fit fails.
  write.csv(data.frame(run_id = run_id, dataset = names(DATA_FILES),
                      input = unname(DATA_FILES), md5 = unname(tools::md5sum(DATA_FILES))),
            file.path(OUTPUT_DIR, "input_manifest.csv"), row.names = FALSE)
  writeLines(c(paste("Run:", run_id), capture.output(sessionInfo())),
             file.path(OUTPUT_DIR, "session_info.txt"))
  add_keys <- function(x, dataset, dv, crowd_type, model_name = NULL) {
    keys <- data.frame(run_id = run_id, dataset = dataset, dv = dv, crowd_type = crowd_type)
    if (!is.null(model_name)) keys$model <- model_name
    cbind(keys[rep(1L, nrow(x)), , drop = FALSE], x, row.names = NULL)
  }
  for (dataset in names(DATA_FILES)) {
    raw <- read.csv(DATA_FILES[[dataset]])
    required_columns <- c("subject", "item", "top_binary", "top_eligible", "rt", "log_rt",
      "rt_eligible", "log_freq", "aoa", "context_familiarity", "regularity", "n_stroke",
      "entropy", "phonetic_radical_order", "log_type_frequency", "log_sem_crowdedness",
      "log_sound_crowdedness", "log_sem_crowdedness_freq", "log_sound_crowdedness_freq")
    missing_columns <- setdiff(required_columns, names(raw))
    if (length(missing_columns)) stop(dataset, " missing columns: ", paste(missing_columns, collapse = ", "))
    for (crowd_type in CROWD_TYPES) {
      for (dv in OUTCOMES) {
        df <- prepare_data(raw, dv, crowd_type)
        models <- model_summaries <- list()
        for (model_name in MODELS) {
          key <- paste(dataset, dv, crowd_type, model_name, sep = "_")
          message("Fitting ", key)
          fit <- capture_conditions(fit_one_model(df, dv, model_name))
          cat(key, fit$warnings, fit$messages, fit$error, "\n", file = diagnostic_file, append = TRUE)
          if (is.null(fit$value)) stop("Model fitting failed: ", key, ". ", fit$error,
            " See diagnostics.log; completed RDS models are retained.")
          model <- fit$value
          # Retain each expensive fit before extracting coefficients or diagnostics.
          saveRDS(model, file.path(OUTPUT_DIR, paste0(key, ".rds")))
          if (!identical(rownames(model.frame(model)), rownames(df))) {
            stop("Model changed trial identities: ", key)
          }
          models[[model_name]] <- model
          r2 <- extract_r2(model)
          summary_row <- extract_model_summary(model, fit, r2)
          cat(key, r2$r2_status, r2$r2_warnings, r2$r2_messages, r2$r2_error, "\n",
              file = diagnostic_file, append = TRUE)
          if (nzchar(fit$warnings) || nzchar(fit$messages) || r2$r2_status != "ok" ||
              nzchar(r2$r2_warnings) || !isTRUE(summary_row$converged) || summary_row$singular) {
            message("  Diagnostics require review: ", key)
          }
          coefficients[[key]] <- add_keys(extract_coefficients(model, dv), dataset, dv, crowd_type, model_name)
          summaries[[key]] <- add_keys(summary_row, dataset, dv, crowd_type, model_name)
          random_effects[[key]] <- add_keys(as.data.frame(VarCorr(model)), dataset, dv, crowd_type, model_name)
          model_summaries[[model_name]] <- summary_row
        }
        # All three models use the same complete-case data; verify actual fitted rows.
        for (j in seq_len(nrow(COMPARISONS))) {
          small <- COMPARISONS$base_model[j]
          large <- COMPARISONS$added_model[j]
          if (!identical(rownames(model.frame(models[[small]])),
                         rownames(model.frame(models[[large]])))) stop("Model samples differ.")
          comparison <- capture_conditions(anova(models[[small]], models[[large]], refit = FALSE))
          key <- paste(dataset, dv, crowd_type, small, large, sep = "_")
          cat(key, comparison$warnings, comparison$messages, comparison$error, "\n",
              file = diagnostic_file, append = TRUE)
          if (is.null(comparison$value)) stop("LRT failed: ", key, ": ", comparison$error)
          lrt <- comparison$value
          if (is.na(lrt$Df[2]) || lrt$Df[2] != COMPARISONS$df[j]) {
            stop("Unexpected LRT degrees of freedom (check rank deficiency): ", key)
          }
          comparisons[[key]] <- add_keys(data.frame(
            base_model = small, added_model = large, n_obs = nobs(models[[small]]),
            chisq = lrt$Chisq[2], df = lrt$Df[2], p = lrt$`Pr(>Chisq)`[2],
            delta_r2_marginal = model_summaries[[large]]$r2_marginal - model_summaries[[small]]$r2_marginal,
            lrt_warnings = comparison$warnings, lrt_messages = comparison$messages), dataset, dv, crowd_type)
        }
      }
    }
  }
  outputs <- list(coefficients = bind_rows(coefficients), model_summary = bind_rows(summaries),
                  random_effects = bind_rows(random_effects), model_comparisons = bind_rows(comparisons))
  expected_groups <- length(DATA_FILES) * length(CROWD_TYPES) * length(OUTCOMES)
  expected_coefficients <- expected_groups * sum(c(1 + length(FIXED_TERMS),
    1 + length(FIXED_TERMS) + length(CROWD_TERMS), 2 + length(FIXED_TERMS) + length(CROWD_TERMS)))
  stopifnot(nrow(outputs$model_summary) == expected_groups * length(MODELS),
            nrow(outputs$model_comparisons) == expected_groups * nrow(COMPARISONS),
            nrow(outputs$coefficients) == expected_coefficients)
  for (name in names(outputs)) {
    write.csv(outputs[[name]], file.path(OUTPUT_DIR, paste0(name, ".csv")), row.names = FALSE)
  }
  # Completion marker prevents a failed rerun from silently reusing older summaries.
  writeLines(run_id, file.path(OUTPUT_DIR, "completed_run.txt"))
  message("Saved 36 model records and 24 comparisons to ", OUTPUT_DIR,
          ". Inspect diagnostics before generating the table.")
}

main()
