source("analysis/data-prep.R")
train_data_keras <- prep_keras_data(filter(training_data, year > 0))

cvae <- keras_model_cvae(num_categories = 10L, num_latent_distributions = 10L, temperature = 0.5, beta = 100,
                         mean_paid = mean_pay, sd_paid = sd_pay)

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
    epochs =1000,
    view_metrics = FALSE,
    verbose = 1
  )

tidied <- training_data %>%
  group_by(ClNr) %>%
  # get the latest valuation of claim
  slice(n()) %>%
  # don't forecast claims already at maturity 11 (which we assume to be ultimate)
  filter(year < 11) %>%
  mutate_scoring_tensors() %>%
  compute_forecasts(100L, cvae$predictor, num_categories = 10L, num_latent_distributions = 10L) %>% 
  tidy_forecasts(mean_pay, sd_pay)
