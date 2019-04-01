#' Given a time series, return a list
#'  where each element is a vector representing a window
#'  of the time series determined by the offsets
make_series <- function(v, start_offset, end_offset, na_pad = 99999) {
  prepad_mask <- function(v, l = 11) {
    length_diff <- l - length(v)
    if (length_diff > 0) {
      c(rep(na_pad, length_diff), v)
    } else {
      v
    }
  }

  purrr::map(
    seq_along(v),
    function(x) {
        start <- max(0, x + start_offset)
        end <- max(0, x + end_offset)
        out <- v[start:end]
        ifelse(is.na(out), na_pad, out)
      } %>%
        prepad_mask()
  )
}

convert_to_tensors <- function(x) {
  x %>%
    lapply(function(e) {
      if (is.null(dim(e))) {
        k_constant(e, shape = c(length(e), 1))
      } else {
        k_constant(e)
      }
    }) %>%
    unname()
}

compute_mack_ultimate <- function(df, claim_ids) {
  triangle_data <- df %>%
    group_by(AY, year) %>%
    summarize(paid = sum(Pay)) %>%
    mutate(cumulative_paid = cumsum(paid))

  atas <- triangle_data %>%
    arrange(year) %>%
    group_by(year) %>%
    summarize(
      year_sum = sum(cumulative_paid),
      year_sum_less_latest = cumulative_paid %>%
        head(-1) %>%
        sum()
    ) %>%
    mutate(ata = year_sum / lag(year_sum_less_latest))

  ldfs <- atas %>%
    mutate(
      ldf = ata %>%
        rev() %>%
        cumprod() %>%
        rev()
    )

  df %>%
    inner_join(claim_ids, by = c("ClNr")) %>%
    group_by(AY, year) %>%
    summarize(paid = sum(Pay)) %>%
    mutate(cumulative_paid = cumsum(paid)) %>%
    group_by(AY) %>%
    summarize(latest_year = max(year), latest_paid = last(cumulative_paid)) %>%
    left_join(
      ldfs %>%
        mutate(ldf = lead(ldf)),
      by = c(latest_year = "year")
    ) %>%
    mutate(ultimate = latest_paid * ldf) %>%
    summarize(total_ultimate = sum(ultimate)) %>%
    pull(total_ultimate)
}

prep_keras_data <- function(x, mask = 99999) {
  ind_lags <- x$claim_open_indicator_lags %>%
    simplify2array(higher = FALSE) %>%
    t()

  ind_lags_flipped <- apply(ind_lags, 1, function(v) {
    masked <- v[v == mask]
    nonmasked <- v[v != mask]
    c(masked, ifelse(nonmasked == 1, 0, 1))
  }) %>% t()

  claim_open_indicator_lags <- cbind(ind_lags, ind_lags_flipped) %>%
    array(dim = c(nrow(.), ncol(.) / 2, 2))

  list(
    x = list(
      paid_loss_lags_ = x$paid_loss_lags %>%
        simplify2array(higher = FALSE) %>%
        t() %>%
        reticulate::array_reshape(c(nrow(.), ncol(.), 1)),
      recovery_lags_ = x$recovery_lags %>%
        simplify2array(higher = FALSE) %>%
        t() %>%
        reticulate::array_reshape(c(nrow(.), ncol(.), 1)),
      claim_open_indicator_lags_ = claim_open_indicator_lags,
      scaled_dev_year_ = x$scaled_dev_year,
      lob_ = x$lob,
      claim_code_ = x$claim_code,
      age_ = x$age,
      injured_part_ = x$injured_part # ,
      # paid_loss_target_ = x$paid_loss_target %>%
      #   simplify2array(higher = FALSE) %>%
      #   t() %>%
      #   reticulate::array_reshape(c(nrow(.), ncol(.), 1))
    ),
    y = list(
      paid_loss_target_ = x$paid_loss_target %>%
        simplify2array(higher = FALSE) %>%
        t() %>%
        reticulate::array_reshape(c(nrow(.), ncol(.), 1)),
      recovery_target_ = x$recovery_target %>%
        simplify2array(higher = FALSE) %>%
        t() %>%
        reticulate::array_reshape(c(nrow(.), ncol(.), 1))
    )
    # y = x$cumulative_paid_loss_target %>%
    #   simplify2array(higher = FALSE) %>%
    #   t() %>%
    #   reticulate::array_reshape(c(nrow(.), ncol(.), 1))
  )
}

mutate_sequences <- function(data, recipe, timesteps, training = TRUE) {
  output <- data %>%
    bake(recipe, .) %>%
    group_by(ClNr) %>%
    mutate(
      claim_open_indicator_lags = make_series(claim_open_indicator, -timesteps, -1),
      paid_loss_lags = make_series(paid_loss, -timesteps, -1),
      recovery_lags = make_series(recovery, -timesteps, -1)
    )

  if (training) {
    output <- output %>%
      mutate(
        # claim_open_indicator_target = make_series(claim_open_indicator, 0, timesteps - 1),
        paid_loss_target = make_series(paid_loss_original, 0, timesteps - 1),
        recovery_target = make_series(recovery_original, 0, timesteps - 1)
        # cumulative_paid_loss_target = make_series(cumulative_paid_loss, 0, timesteps - 1)
      )
  }

  output
}

plot_forecasts <- function(forecasts, actual) {

  # get first dev year in forecast
  first_predicted_dev_year <- min(forecasts$development_year)

  # get subset of actual
  actual_history <- actual %>%
    filter(development_year %in% 0:(first_predicted_dev_year - 1))

  # concatenate the data frames
  forecasts <- forecasts %>%
    group_by(sample) %>%
    group_map(~ bind_rows(
      actual_history %>%
        mutate(type = "actual"),
      .x
    ))

  forecasts %>%
    mutate(cumulative_paid = cumsum(paid_loss)) %>%
    # mutate(cumulative_paid = cumulative_paid_loss) %>%
    filter(type == "predicted") %>%
    ggplot(aes(x = development_year, y = cumulative_paid, color = type)) +
    geom_path(alpha = 0.8, size = 0.3) +
    geom_line(aes(color = type), data = mutate(actual, cumulative_paid = cumsum(paid_loss), type = "actual"), alpha = 0.3) +
    # geom_line(aes(color = type), data = mutate(actual, cumulative_paid = cumulative_paid_loss, type = "actual"), alpha = 0.3) +
    theme_bw() +
    scale_color_brewer(palette = "Dark2") +
    ylim(-10, NA)
}

prep_datasets <- function(simulated_cashflows, n, timesteps = 11, record_year_cutoff = 2005) {
  claim_ids <- simulated_cashflows %>%
    distinct(ClNr) %>%
    sample_n(!!n)

  cashflow_history <- simulated_cashflows %>%
    inner_join(claim_ids, by = "ClNr")

  training_data <- cashflow_history %>%
    filter(record_year <= !!record_year_cutoff)

  rec <- recipe(training_data, ~.) %>%
    step_integer(lob, claim_code, injured_part, zero_based = TRUE) %>%
    step_center(age, paid_loss, recovery) %>%
    step_scale(age, paid_loss, recovery) %>%
    step_mutate(scaled_dev_year = year / 11) %>%
    prep(training_data)

  #' Capture mean/sd for paids so we can recover after prediction
  mean_paid <- rec$step[[2]]$means[["paid_loss"]]
  sd_paid <- rec$steps[[3]]$sds[["paid_loss"]]

  training_data <- training_data %>%
    mutate_sequences(rec, timesteps)

  dev_year_zero_records <- training_data %>%
    filter(year == 0)

  training_data <- training_data %>%
    filter(year > 0)

  # training_weight <- training_data %>%
  # #   mutate(
  # #   training_weight = map_dbl(paid_loss_target, ~ sum(.x != 99999))
  # # ) %>%
  #   pull(training_weight)

  list(
    training_data = training_data,
    dev_year_zero_records = dev_year_zero_records,
    # training_weight = training_weight,
    cashflow_history = cashflow_history,
    mean_paid = mean_paid,
    sd_paid = sd_paid
  )
}

compute_tidy_forecasts <- function(data, decoder_model, num_draws, mean_paid, sd_paid, num_categories = 10L, num_latent_distributions = 10) {
  records_to_score <- data %>%
    mutate(
      paid_loss_lags = map2(paid_loss_lags, paid_loss, ~ c(.x[-1], .y)),
      claim_open_indicator_lags = map2(claim_open_indicator_lags, claim_open_indicator, ~ c(.x[-1], .y)),
      scaled_dev_year = scaled_dev_year + 1 / 11
    ) %>%
    slice(rep(1, num_draws))

  draws <- tfprobability::tfd_one_hot_categorical(
    logits = array(
      rep(1, num_draws * num_categories * num_latent_distributions),
      dim = c(num_draws, num_categories, num_latent_distributions)
    ),
    dtype = k_floatx()
  ) %>%
    tfd_sample() %>%
    as.array()

  draws <- rray::rray_tile(draws, nrow(data))

  scoring_data_keras <- records_to_score %>%
    prep_keras_data()

  scoring_data_keras <- c(list(draw_input_ = draws), scoring_data_keras$x)

  preds <- predict(decoder_model, scoring_data_keras, batch_size = 2048)

  dev_years <- records_to_score %>%
    distinct(ClNr, year) %>%
    transmute(
      development_year = list(year + 1:11)
    ) %>%
    slice(rep(1, num_draws)) %>%
    unnest() %>%
    mutate(
      sample = paste0("sample_", rep(1:num_draws, 11)),
      type = "predicted"
    )

  preds[, , 1] %>%
    as_tibble(.name_repair = ~ paste0("V", 1:11)) %>%
    gather() %>%
    bind_cols(dev_years) %>%
    mutate(paid_loss = value * sd_paid + mean_paid) %>%
    filter(development_year <= 11) %>%
    select(-key)
}

masked_negloglik <- function(mask_value) {
  function(y_true, y_pred) {
    keep_value <- k_cast(k_not_equal(
      tf$squeeze(y_true),
      # y_true,
      mask_value), k_floatx())
    
    logprob <- y_pred$distribution$log_prob(
      tf$squeeze(y_true)
      # y_true
    )
    -k_sum(keep_value * logprob, axis = 2)
  }
}

cust_loss <- function(x, rv_x) masked_negloglik(99999)(x, rv_x)