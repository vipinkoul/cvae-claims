source("analysis/initialize-environment.R")
source("analysis/data-prep.R")
source("analysis/model.R")

results <- purrr::map_df(1:10, function(i) {
  dataset <- prep_datasets(simulated_cashflows, 10000)
  train_data_keras <- prep_keras_data(filter(dataset$training_data, year > 0))
  
  cvae <- keras_model_cvae(num_categories = 5L, num_latent_distributions = 5L, temperature = 0.5, beta = 10,
                           mean_paid = dataset$mean_paid, sd_paid = dataset$sd_paid)
  
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
  
  vae_loss <- function(x, rv_x) masked_negloglik(-99)(x, rv_x)
  
  cvae$cvae %>%
    compile(
      loss = vae_loss,
      optimizer = tf$keras$optimizers$Adam(1e-3)
    )
  
  history <- cvae$cvae %>%
    fit(
      x = train_data_keras$x,
      y = train_data_keras$y,
      sample_weight = array(dataset$training_weight, length(dataset$training_weight)),
      batch_size = 2048,
      epochs = 1000,
      view_metrics = FALSE,
      verbose = 0
    )
  
  scoring_data <- bind_rows(dataset$training_data, dataset$dev_year_zero_records) %>%
    arrange(ClNr, year)
  
  tidied <- scoring_data %>% 
    group_by(ClNr) %>%
    # get the latest valuation of claim
    slice(n()) %>%
    # don't forecast claims already at maturity 12 (which we assume to be ultimate)
    filter(year < 11) %>%
    compute_tidy_forecasts(cvae$predictor, 100, mean_paid = dataset$mean_paid, sd_paid = dataset$sd_paid,
                           num_categories = 5L, num_latent_distributions = 5L)
  
  # claims not already at maturity
  claim_ids_for_comparison <- tidied %>%
    distinct(ClNr)
  
  actual_ultimate <- dataset$cashflow_history %>%
    inner_join(claim_ids_for_comparison, by = "ClNr") %>%
    summarize(ultimate = sum(Pay)) %>%
    pull(ultimate)
    
  mack_ultimate <- compute_mack_ultimate(scoring_data, claim_ids_for_comparison)
  
  predicted_future_paid <- tidied %>%
    group_by(ClNr, sample) %>%
    summarize(predicted_future_paid = sum(paid_loss)) %>%
    group_by(ClNr) %>%
    summarize(mean_predicted_future_paid = mean(predicted_future_paid))
  
  predicted_df <- scoring_data %>% 
    group_by(ClNr) %>%
    summarize(paid = sum(Pay)) %>%
    right_join(predicted_future_paid, by = "ClNr") %>%
    mutate(ultimate = paid + ifelse(is.na(mean_predicted_future_paid), 0, mean_predicted_future_paid))
  
  nn_ultimate <- sum(predicted_df$ultimate)
  
  mack_ultimate / actual_ultimate - 1
  nn_ultimate / actual_ultimate - 1
  
  list(
    forecasts = list(tidied),
    actual_ultimate = actual_ultimate,
    mack_ultimate = mack_ultimate,
    nn_ultimate = nn_ultimate,
    resample = i
  )
})

