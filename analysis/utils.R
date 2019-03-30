#' Given a time series, return a list
#'  where each element is a vector representing a window
#'  of the time series determined by the offsets
make_series <- function(v, start_offset, end_offset, na_pad = -99) {
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

masked_mse <- function(mask_value) {
  function(y_true, y_pred) {
    keep_value <- k_cast(k_not_equal(y_true, mask_value), k_floatx())
    sum_squared_error <- k_sum(
      k_square(keep_value * (y_true - y_pred)),
      axis = 2
    )
    mse <- sum_squared_error / k_sum(keep_value, axis = 2)
    
    mse
  }
}

masked_future_paid_error <- function(mask_value) {
  function(y_true, y_pred) {
    keep_value <- k_cast(k_not_equal(y_true, mask_value), k_floatx())
    
    squared_error_future_paid <- k_square(
      k_sum(keep_value * y_true, axis = 2) - k_sum(keep_value * y_pred, axis = 2)
    )
    
    squared_error_future_paid
  }
}

masked_mse2 <- function(mask_value) {
  function(y_true, y_pred) {
    keep_value <- k_cast(k_not_equal(y_true, mask_value), k_floatx())
    sum_squared_error <- k_sum(
      k_square(keep_value * (y_true - y_pred) / (y_true + y_pred)),
      axis = 2
    )
    sum_squared_error / k_sum(keep_value, axis = 2)
  }
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

prep_keras_data <- function(x, mask = -99) {
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
      paid_loss_lags_ =  x$paid_loss_lags %>%
        simplify2array(higher = FALSE) %>%
        t() %>%
        reticulate::array_reshape(c(nrow(.), ncol(.), 1)),
      claim_open_indicator_lags_ = claim_open_indicator_lags,
      lob_ = x$lob,
      claim_code_ = x$claim_code,
      age_ = x$age,
      injured_part_ = x$injured_part,
      paid_loss_target_ = x$paid_loss_target %>%
        simplify2array(higher = FALSE) %>%
        t() %>%
        reticulate::array_reshape(c(nrow(.), ncol(.), 1))
    ),
    y = x$paid_loss_target %>%
      simplify2array(higher = FALSE) %>%
      t() %>%
      reticulate::array_reshape(c(nrow(.), ncol(.), 1))
  )
}

mutate_sequences <- function(data, recipe, timesteps, training = TRUE) {
  output <- data %>% 
    bake(recipe, .) %>%
    group_by(ClNr) %>%
    mutate(
      claim_open_indicator_lags = make_series(claim_open_indicator, -timesteps, -1),
      paid_loss_lags = make_series(paid_loss, -timesteps, -1)#,
    ) 
  
  if (training) {
    output <- output %>%
      mutate(
        claim_open_indicator_target = make_series(claim_open_indicator, 0, timesteps - 1),
        paid_loss_target = make_series(paid_loss, 0, timesteps - 1)#,
      )
  }
  
  output
}

mutate_scoring_tensors <- function(data) {
  data_keras <- data %>%
    mutate(
      paid_loss_lags = map2(paid_loss_lags, paid_loss, ~ c(.x[-1], .y)),
      claim_open_indicator_lags = map2(claim_open_indicator_lags, claim_open_indicator, ~c(.x[-1], .y))
    ) %>%
    prep_keras_data()
  
  n_claims <- data_keras$x$paid_loss_lags %>% 
    dim() %>% 
    first()
  
  tensors_for_scoring <- map(seq_len(n_claims), function(i) {
    x <- data_keras$x 
    list(
      paid_loss_lags = x$paid_loss_lags[i,,,drop = FALSE],
      claim_open_indicator_lags = x$claim_open_indicator_lags[i,,,drop = FALSE],
      lob = x$lob[[i]],
      claim_code = x$claim_code[[i]],
      age = x$age[[i]],
      injured_part = x$injured_part[[i]]
    ) %>%
      convert_to_tensors()
  })
  
  bind_cols(
    data,
    tibble(scoring_tensor = tensors_for_scoring)
  )
}

compute_forecasts <- function(data, num_draws, decoder, num_categories = 10L, num_latent_distributions = 20L) {
  scoring_tensors <- data$scoring_tensor %>%
    map(function(tensors) {
      map(tensors, ~k_repeat_elements(.x, num_draws, 1))
    }) %>%
    purrr::transpose() %>%
    purrr::map(k_concatenate, axis = 1)
  
  random_draws <- tfprobability::tfd_one_hot_categorical(
    logits = array(
      rep(1, nrow(data) * num_draws * num_latent_distributions * num_categories), 
      dim = c(nrow(data) * num_draws, num_latent_distributions, num_categories)
    ),
    dtype = k_floatx()
  ) %>%
    tfd_sample()
  
  predictions <- predict(decoder, c(random_draws, scoring_tensors))
  
  predictions <- seq_len(nrow(data)) %>% 
    map(~ (.x - 1) * num_draws + 1:num_draws) %>%
    map(~ predictions[.x,,])
  
  bind_cols(
    data,
    tibble(forecasts = predictions)
  )
}

tidy_forecasts <- function(data, mean_paid_loss, sd_paid_loss) {
  data %>%
    group_by(ClNr) %>%
    group_map(function(df, ...) {
      
      dev_year <- df$year + 1:11
      
      num_draws <- dim(data$forecasts[[1]])[[1]]
      
      df$forecasts[[1]] %>%
        apply(1, function(x) tibble(development_year = dev_year, paid_loss = x, type = "predicted")) %>%
        set_names(paste0("sample_", seq_len(num_draws))) %>%
        bind_rows(.id = "sample") %>%
        mutate(paid_loss = paid_loss * !!sd_paid_loss + !!mean_paid_loss) %>%
        filter(development_year <= 11)
    }) %>%
    ungroup()
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
    filter(type == "predicted")  %>%
    ggplot(aes(x = development_year, y = cumulative_paid, color = type)) + 
    geom_path(alpha = 0.8, size = 0.3)  +
    geom_line(aes(color = type), data = mutate(actual, cumulative_paid = cumsum(paid_loss), type = "actual"), alpha = 0.3) +
    theme_bw() + 
    scale_color_brewer(palette="Dark2") +
    ylim(0, NA)
}
