#!/usr/bin/env Rscript
# Fit lexical and seed-standardized target-support models of retrieval difficulty.
# Run from the repository root: Rscript analysis/mixed_effects_logits/02_fit_standardized_measures.R
# Reads data/processed/*_trials_with_logit_measures.csv; saves CSV summaries and RDS
# models in output_standardized_measures/. Inspect diagnostics before interpretation.

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
            paste0(c("huang", "wang_small", "wang_full"), "_trials_with_logit_measures.csv")),
  c("huang", "wang_small", "wang_full"))
OUTPUT_DIR <- here::here("analysis", "mixed_effects_logits", "output_standardized_measures")
PREPARATION_DIR <- here::here("data", "analysis_inputs", "relative_measure_preparation")
OLD_BASE_DIR <- here::here("analysis", "mixed_effects_entropy", "output")
CANONICAL_DIR <- here::here("data", "processed")
DATASETS <- names(DATA_FILES)
OUTCOMES <- c("top", "rt")
METRICS <- c("target_logit_measure", "deficit_measure", "blocking_measure")
CENTERED_METRICS <- paste0(METRICS, "_c")
MODEL_TERMS <- c(base = "", setNames(CENTERED_METRICS, METRICS))
EXPECTED_ITEMS <- c(huang = 200L, wang_small = 200L, wang_full = 1592L)
NUMERIC_TOLERANCE <- 1e-10
BASE_COMPARISON_TOLERANCE <- 1e-6
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

# ---- Verify that all three trial files belong to one completed preparation ---
validate_preparation <- function() {
  paths <- file.path(PREPARATION_DIR, c("completed_run.txt", "output_manifest.csv", "run_metadata.json"))
  if (!all(file.exists(paths))) stop("Missing completed preprocessing audit; run merge_logit_measures.py first")
  run_id <- readLines(paths[1], warn = FALSE, encoding = "UTF-8")
  manifest <- read.csv(paths[2], fileEncoding = "UTF-8-BOM", stringsAsFactors = FALSE)
  required <- c("run_id", "dataset", "path", "md5", "sha256")
  if (length(run_id) != 1L || !nzchar(run_id) || !all(required %in% names(manifest)) ||
      nrow(manifest) != length(DATASETS) || anyDuplicated(manifest$dataset) ||
      !setequal(manifest$dataset, DATASETS) || anyNA(manifest[required]) ||
      !all(manifest$run_id == run_id)) stop("Invalid preprocessing manifest/run identifier")
  for (dataset in DATASETS) {
    row <- manifest[manifest$dataset == dataset, , drop = FALSE]
    # Historical manifests retain original locations. Validate this local copy
    # by dataset filename and content hash without accessing the old computer.
    recorded_name <- basename(gsub("\\\\", "/", row$path))
    if (recorded_name != basename(DATA_FILES[[dataset]]) ||
        unname(tools::md5sum(DATA_FILES[[dataset]])) != row$md5) {
      stop("Prepared trial file differs from completed manifest: ", dataset)
    }
  }
  list(run_id = run_id, paths = paths)
}

# ---- Validate trial identity and prepare one common sample -------------------
validate_input <- function(df, dataset) {
  path <- file.path(CANONICAL_DIR, paste0(dataset, "_trials_with_entropy.csv"))
  if (!file.exists(path)) stop("Missing canonical input: ", path)
  canonical <- read.csv(path, fileEncoding = "UTF-8-BOM", check.names = FALSE)
  required <- c(names(canonical), METRICS)
  if (!identical(names(df), required) ||
      !isTRUE(all.equal(df[names(canonical)], canonical, tolerance = 0))) {
    stop("Prepared input changed canonical columns or row order: ", dataset)
  }
  keys <- c("Character", "Word")
  if (anyNA(df[keys]) || any(as.matrix(df[keys]) == "")) stop("Missing character/context keys")
  if (!all(vapply(df[METRICS], is.numeric, logical(1)))) stop("Non-numeric measure columns")
  present <- rowSums(!is.na(df[METRICS]))
  if (any(!present %in% c(0L, length(METRICS)))) stop("Partially missing measure columns")
  used <- present == length(METRICS)
  if (!all(is.finite(as.matrix(df[used, METRICS]))) ||
      !identical(is.na(df$entropy), !used)) stop("Invalid measure values/context coverage: ", dataset)
  inventory <- unique(df[c(keys, METRICS)])
  if (anyDuplicated(inventory[keys]) ||
      sum(!is.na(inventory[[METRICS[1]]])) != EXPECTED_ITEMS[[dataset]]) {
    stop("Unexpected Character-Word metric coverage: ", dataset)
  }
  # Original row numbers persist through filtering and lme4 model frames.
  df$.source_row <- seq_len(nrow(df))
  df
}

prepare_data <- function(df, dv) {
  if (dv == "top") {
    df <- df %>% filter(as.logical(top_eligible))
  } else {
    df <- df %>% filter(as.logical(rt_eligible), rt <= RT_MAX_MS)
  }
  # Subtract means only. The model measures already contain seed-wise z-scores;
  # do not divide them by a new dataset/trial-level SD.
  for (v in c("log_freq", "aoa", "context_familiarity", "regularity", "n_stroke",
              METRICS)) {
    df[[paste0(v, "_c")]] <- df[[v]] - mean(df[[v]], na.rm = TRUE)
  }
  df$radical_order_c <- ifelse(df$phonetic_radical_order == 1, 0.5, -0.5)
  df$subject <- as.factor(df$subject)
  df$item <- as.factor(df$item)
  response <- if (dv == "top") "top_binary" else "log_rt"
  required <- c(FIXED_TERMS, CENTERED_METRICS, response, "subject", "item")
  df <- df %>% filter(if_all(all_of(required), ~ !is.na(.)))
  if (!nrow(df) || !all(is.finite(as.matrix(df[c(FIXED_TERMS,
      CENTERED_METRICS, response)])))) stop("Empty or nonfinite model sample")
  if (dv == "top" && !all(df$top_binary %in% c(0, 1))) stop("Invalid TOP outcomes")
  rownames(df) <- as.character(df$.source_row)
  df
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
  terms <- c(FIXED_TERMS, if (nzchar(MODEL_TERMS[[model_name]])) MODEL_TERMS[[model_name]],
             "(1 | subject)", "(1 | item)")
  response <- if (dv == "top") "top_binary" else "log_rt"
  formula <- as.formula(paste(response, "~", paste(terms, collapse = " + ")))
  if (dv == "top") {
    glmer(formula, data = df, family = binomial, control = GLMER_CONTROL, na.action = na.fail)
  } else {
    lmer(formula, data = df, REML = FALSE, control = LMER_CONTROL, na.action = na.fail)
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

# ---- Audit the new base against the saved entropy-analysis base --------------
compare_old_base <- function(model, dataset, dv) {
  path <- file.path(OLD_BASE_DIR, paste0(dataset, "_", dv, "_base.rds"))
  result <- data.frame(old_base_available = file.exists(path), same_sample = NA,
                       same_predictors = NA, same_specification = NA,
                       max_abs_beta_difference = NA_real_, loglik_difference = NA_real_,
                       equivalent_within_tolerance = NA, status = "old_base_missing")
  if (!file.exists(path)) {
    warning("No historical base available for comparison: ", path)
    return(result)
  }
  old <- readRDS(path)
  previous <- model.frame(old)
  current <- model.frame(model)
  # Compare ordered responses, subject/item identities and all base predictors.
  result$same_sample <- identical(names(previous), names(current)) &&
    nrow(previous) == nrow(current) &&
    all(vapply(c(names(current)[1], "subject", "item"), function(v)
      identical(as.character(previous[[v]]), as.character(current[[v]])), logical(1)))
  result$same_predictors <- result$same_sample &&
    isTRUE(all.equal(model.matrix(old), model.matrix(model),
                     check.attributes = FALSE, tolerance = NUMERIC_TOLERANCE))
  result$same_specification <- identical(deparse(formula(old)), deparse(formula(model))) &&
    identical(family(old)$family, family(model)$family) &&
    identical(family(old)$link, family(model)$link) && identical(isREML(old), isREML(model))
  if (identical(names(fixef(old)), names(fixef(model)))) {
    result$max_abs_beta_difference <- max(abs(fixef(model) - fixef(old)))
  }
  result$loglik_difference <- as.numeric(logLik(model) - logLik(old))
  result$equivalent_within_tolerance <- isTRUE(result$same_sample) &&
    isTRUE(result$same_predictors) && isTRUE(result$same_specification) &&
    is.finite(result$max_abs_beta_difference) &&
    result$max_abs_beta_difference <= BASE_COMPARISON_TOLERANCE &&
    abs(result$loglik_difference) <= BASE_COMPARISON_TOLERANCE
  result$status <- if (result$equivalent_within_tolerance) "matches" else "review_difference"
  if (result$status != "matches") warning("Historical base differs: ", dataset, " / ", dv,
    ". Review base_model_audit.csv; new LRTs use the newly fitted common-sample base.")
  result
}

# ---- Run analysis and save reusable results ---------------------------------
main <- function() {
  missing_files <- DATA_FILES[!file.exists(DATA_FILES)]
  if (length(missing_files)) stop("Missing input files: ", paste(missing_files, collapse = ", "))
  dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
  # A shared identifier prevents the table from combining different analysis runs.
  run_id <- format(Sys.time(), "%Y%m%dT%H%M%OS6")
  coefficients <- summaries <- random_effects <- comparisons <- base_audits <- sample_rows <- list()
  diagnostic_file <- file.path(OUTPUT_DIR, "diagnostics.log")
  writeLines(paste("Analysis run", run_id), diagnostic_file)
  # The table rejects stale summaries if this run fails partway through.
  completion_file <- file.path(OUTPUT_DIR, "completed_run.txt")
  if (file.exists(completion_file)) unlink(completion_file)
  preparation <- validate_preparation()
  all_inputs <- c(unname(DATA_FILES), preparation$paths, file.path(CANONICAL_DIR,
    paste0(DATASETS, "_trials_with_entropy.csv")))
  input_hashes <- tools::md5sum(all_inputs)
  cat("Preprocessing run", preparation$run_id, "\n", file = diagnostic_file, append = TRUE)
  add_keys <- function(x, dataset, dv, model_name = NULL) {
    keys <- data.frame(run_id = run_id, dataset = dataset, dv = dv)
    if (!is.null(model_name)) keys$model <- model_name
    cbind(keys[rep(1L, nrow(x)), , drop = FALSE], x, row.names = NULL)
  }
  for (dataset in names(DATA_FILES)) {
    raw <- validate_input(read.csv(DATA_FILES[[dataset]], fileEncoding = "UTF-8-BOM",
                                   check.names = FALSE), dataset)
    for (dv in OUTCOMES) {
      df <- prepare_data(raw, dv)
      models <- pair_summaries <- list()
      sample_rows[[paste(dataset, dv, sep = "_")]] <- add_keys(
        df[c(".source_row", "subject", "item", "Character", "Word")], dataset, dv)
      for (model_name in names(MODEL_TERMS)) {
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
      base_audits[[paste(dataset, dv, sep = "_")]] <- add_keys(
        compare_old_base(models$base, dataset, dv), dataset, dv)
      # Actual model-frame row identities must match the prepared common sample.
      for (model_name in names(MODEL_TERMS)) {
        if (!identical(rownames(model.frame(models[[model_name]])), rownames(df))) {
          stop("Model sample differs: ", dataset, " / ", dv, " / ", model_name)
        }
      }
      for (added in setdiff(names(MODEL_TERMS), "base")) {
        comparison <- capture_conditions(anova(models$base, models[[added]], refit = FALSE))
        cat(dataset, dv, added, comparison$warnings, comparison$messages, comparison$error,
            "\n", file = diagnostic_file, append = TRUE)
        if (is.null(comparison$value)) stop("LRT failed: ", comparison$error)
        lrt <- comparison$value
        if (lrt$Df[2] != 1L || !is.finite(lrt$Chisq[2]) ||
            !is.finite(lrt$`Pr(>Chisq)`[2])) stop("Invalid one-parameter LRT")
        comparisons[[paste(dataset, dv, added, sep = "_")]] <- add_keys(data.frame(
          base_model = "base", added_model = added, n_obs = nobs(models$base),
          chisq = lrt$Chisq[2], df = lrt$Df[2], p = lrt$`Pr(>Chisq)`[2],
          delta_r2_marginal = pair_summaries[[added]]$r2_marginal -
            pair_summaries$base$r2_marginal,
          lrt_warnings = comparison$warnings, lrt_messages = comparison$messages), dataset, dv)
      }
    }
  }
  outputs <- list(coefficients = bind_rows(coefficients), model_summary = bind_rows(summaries),
                  random_effects = bind_rows(random_effects), model_comparisons = bind_rows(comparisons),
                  base_model_audit = bind_rows(base_audits), sample_rows = bind_rows(sample_rows))
  expected_groups <- length(DATASETS) * length(OUTCOMES)
  expected_models <- expected_groups * length(MODEL_TERMS)
  expected_comparisons <- expected_groups * length(METRICS)
  expected_coefficients <- expected_groups * (
    length(FIXED_TERMS) + 1L + length(METRICS) * (length(FIXED_TERMS) + 2L))
  stopifnot(expected_models == 24L, expected_comparisons == 18L,
            nrow(outputs$coefficients) == expected_coefficients,
            nrow(outputs$model_summary) == expected_models,
            nrow(outputs$model_comparisons) == expected_comparisons,
            nrow(outputs$base_model_audit) == expected_groups)
  if (!identical(input_hashes, tools::md5sum(all_inputs))) stop("Inputs changed during fitting")
  for (name in names(outputs)) {
    write.csv(outputs[[name]], file.path(OUTPUT_DIR, paste0(name, ".csv")), row.names = FALSE)
  }
  # Provenance supports later checks against the input data and package versions.
  write.csv(data.frame(run_id = run_id, input = all_inputs, md5 = unname(input_hashes)),
            file.path(OUTPUT_DIR, "input_manifest.csv"), row.names = FALSE)
  writeLines(c(paste("Run:", run_id), capture.output(sessionInfo())),
             file.path(OUTPUT_DIR, "session_info.txt"))
  writeLines(run_id, completion_file)
  message("Saved 24 models and 18 comparisons to ", OUTPUT_DIR,
          ". Inspect diagnostics and Base audits before interpretation.")
}

if (sys.nframe() == 0L) main()
