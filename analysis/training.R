source("analysis/initialize-environment.R")
source("analysis/data-prep.R")
source("analysis/model.R")

results <- purrr::map_df(1:5, function(i) {
  dataset <- prep_datasets(simulated_cashflows, 10000)
  train_data_keras <- prep_keras_data(filter(dataset$training_data, year > 0))
  
  cvae <- keras_model_cvae(num_categories = 10L, num_latent_distributions = 10L, temperature = 0.5, beta = 1,
                           mean_paid = dataset$mean_paid, sd_paid = dataset$sd_paid)
  
  recon_loss <- custom_metric("recon_loss", function(y_true, y_pred) {
    masked_mse(-99)(y_true, y_pred)
  })
  
  recon_loss2 <- custom_metric("recon_loss2", function(y_true, y_pred) {
    masked_future_paid_error(-99)(y_true, y_pred)
  })
  
  vae_loss <- function(y_true, y_pred) {
    recon_loss(y_true, y_pred) + recon_loss2(y_true, y_pred)
  }
  
  cvae$cvae %>%
    compile(
      loss = vae_loss,
      optimizer = tf$keras$optimizers$Adam(1e-3),
      metrics = list(recon_loss, recon_loss2)
    )
  
  history <- cvae$cvae %>%
    fit(
      x = train_data_keras$x,
      y = train_data_keras$y,
      batch_size = 2048,
      epochs = 1,
      view_metrics = FALSE,
      verbose = 1
    )
  
  tidied <- dataset$training_data %>%
    group_by(ClNr) %>%
    # get the latest valuation of claim
    slice(n()) %>%
    # don't forecast claims already at maturity 12 (which we assume to be ultimate)
    filter(year < 11) %>%
    compute_tidy_forecasts(cvae$predictor, 100, mean_paid = dataset$mean_paid, sd_paid = dataset$sd_paid,
                           num_categories = 10L, num_latent_distributions = 10)
  
  # claims not already at maturity
  claim_ids_for_comparison <- tidied %>%
    distinct(ClNr)
  
  actual_ultimate <- dataset$cashflow_history %>%
    inner_join(claim_ids_for_comparison, by = "ClNr") %>%
    summarize(ultimate = sum(Pay)) %>%
    pull(ultimate)
    
  mack_ultimate <- compute_mack_ultimate(dataset$training_data, claim_ids_for_comparison)
  
  predicted_future_paid <- tidied %>%
    group_by(ClNr, sample) %>%
    summarize(predicted_future_paid = sum(paid_loss)) %>%
    group_by(ClNr) %>%
    summarize(mean_predicted_future_paid = mean(predicted_future_paid))
  
  predicted_df <- dataset$training_data %>%
    group_by(ClNr) %>%
    summarize(paid = sum(Pay)) %>%
    right_join(predicted_future_paid, by = "ClNr") %>%
    mutate(ultimate = paid + ifelse(is.na(mean_predicted_future_paid), 0, mean_predicted_future_paid))
  
  nn_ultimate <- sum(predicted_df$ultimate)
  
  list(
    forecasts = list(tidied),
    actual_ultimate = actual_ultimate,
    mack_ultimate = mack_ultimate,
    nn_ultimate = nn_ultimate,
    resample = i
  )
})

