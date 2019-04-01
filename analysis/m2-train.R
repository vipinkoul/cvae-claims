source("analysis/initialize-environment.R")
source("analysis/data-prep.R")
source("analysis/model2.R")
dataset <- prep_datasets(simulated_cashflows, 20000)
train_data_keras <- prep_keras_data(filter(dataset$training_data, year > 0))

model2 <- make_model2(regularizer = NULL, ln_scale_bound = 1)

model2 %>%
  compile(
    loss = list(cust_loss, cust_loss),
    loss_weights = list(1, 1),
    optimizer = tf$keras$optimizers$Adam(1e-3)
  )

history <- model2 %>%
  fit(
    x = train_data_keras$x,
    y = unname(train_data_keras$y),
    batch_size = 50000,
    epochs = 1000,
    view_metrics = FALSE,
    verbose = 1,
    callbacks = list(callback_early_stopping(monitor = "loss", patience = 200, restore_best_weights = TRUE))
  )
plot(history)


scoring_data <- bind_rows(dataset$training_data, dataset$dev_year_zero_records) %>%
  arrange(ClNr, year) %>% 
  group_by(ClNr) %>% 
  filter(max(year) < 11)



records_to_score <- scoring_data %>% 
  group_by(ClNr) %>%
  # get the latest valuation of claim
  slice(n()) %>%
  # don't forecast claims already at maturity 12 (which we assume to be ultimate)
  # filter(year < 11) %>% 
  mutate(
    paid_loss_lags = map2(paid_loss_lags, paid_loss, ~ c(.x[-1], .y)),
    claim_open_indicator_lags = map2(claim_open_indicator_lags, claim_open_indicator, ~c(.x[-1], .y)),
    scaled_dev_year = scaled_dev_year + 1/11
  )

scoring_data_keras <- records_to_score %>%
  prep_keras_data()

batch_size <- 50
n_samples <- 1000

ds <- tensor_slices_dataset(scoring_data_keras$x) %>% 
  dataset_map(function(record) {
    record$claim_open_indicator_lags_ <- tf$cast(record$claim_open_indicator_lags_, k_floatx())
    record$paid_loss_lags_ <- tf$cast(record$paid_loss_lags_, k_floatx())
    record$recovery_lags_ <- tf$cast(record$recovery_lags_, k_floatx())
    record$lob_ <- tf$cast(record$lob_, tf$int32)
    record$claim_code_ <- tf$cast(record$lob, tf$int32)
    record$age_ <- tf$cast(record$age_, k_floatx())
    record$injured_part_ <- tf$cast(record$injured_part_, tf$int32)
    
    # Bogus data
    # record$claim_open_indicator_lags_ <- tf$zeros_like(record$claim_open_indicator_lags_)
    # record$paid_loss_lags_ <-tf$zeros_like(record$paid_loss_lags_)
    # record$recovery_lags_ <- tf$zeros_like(record$recovery_lags_)
    # record$lob_ <- tf$zeros_like(record$lob_)
    # record$claim_code_ <- tf$zeros_like(record$claim_code_)
    # record$age_ <- tf$zeros_like(record$age_)
    # record$injured_part_ <- tf$zeros_like(record$injured_part_)
    
    record
  }) %>% 
  dataset_batch(batch_size)

iter <- make_iterator_one_shot(ds)

preds_list <- list()
i <- 1
until_out_of_range({
  cat("scoring batch number: ", i, "\n")
  batch <- iterator_get_next(iter)
  
  samples <-  with(tf$device("/cpu:0"), {
    
    dists <- model2(unname(batch))
    
    dists %>%
      lapply(tfd_sample, n_samples) %>%
      reduce(`-`) %>%
      as.array()
  })

  preds_list <- c(preds_list, list(samples))
  i <- i + 1
})

preds <- array(0, dim = c(n_samples, batch_size * (length(preds_list) - 1), 11))
for(i in 1:length(preds_list[1:(length(preds_list) - 1)])) {
  preds[,((i-1)*batch_size+1):(i*batch_size),] <- preds_list[[i]]
}

preds <- rray::rray_bind(preds, preds_list[[length(preds_list)]], axis = 2)

dev_years <- records_to_score %>%
  distinct(ClNr, year) %>%
  transmute(
    development_year = list(year + 1:11)
  ) %>%
  ungroup() %>% 
  unnest() %>% 
  mutate(type = "predicted")

tidied <- preds %>% 
  apply(1, as_tibble, .name_repair = ~ paste0("V", 1:11)) %>% 
  map_dfr(~ {
    .x %>% 
      gather() %>% 
      bind_cols(dev_years) %>% 
      mutate(paid_loss = value) %>% 
      filter(development_year <= 11) %>%
      select(-key)
  }, .id = "sample")


claim_ids_for_comparison <- tidied %>%
  distinct(ClNr)

actual_ultimate <- dataset$cashflow_history %>%
  inner_join(claim_ids_for_comparison, by = "ClNr") %>%
  summarize(ultimate = sum(Pay)) %>%
  pull(ultimate)

mack_ultimate <- compute_mack_ultimate(bind_rows(dataset$training_data, dataset$dev_year_zero_records), claim_ids_for_comparison)

predicted_future_paid <- tidied %>%
  group_by(ClNr, sample) %>%
  summarize(predicted_future_paid = sum(paid_loss)) %>%
  group_by(ClNr) %>%
  summarize(mean_predicted_future_paid = mean(predicted_future_paid))

predicted_df <- scoring_data %>%
  group_by(ClNr) %>%
  summarize(paid = sum(Pay)) %>%
  right_join(predicted_future_paid, by = "ClNr") %>%
  mutate(ultimate = paid + mean_predicted_future_paid)

nn_ultimate <- sum(predicted_df$ultimate)


tidied$paid_loss %>% max()
tidied$paid_loss %>% min()


already_paid <- scoring_data %>% 
  group_by(ClNr) %>% 
  summarize(paid = sum(Pay)) %>% 
  pull(paid) %>% 
  sum()

actual_future_paid <- actual_ultimate - already_paid
nn_future_paid <- predicted_future_paid$mean_predicted_future_paid %>% sum()
nn_future_paid / actual_future_paid - 1
mack_future_paid <- mack_ultimate - already_paid
mack_future_paid / actual_future_paid - 1
(actual_future_paid + already_paid) / actual_ultimate - 1


already_paid_by_claim <- scoring_data %>% 
  group_by(ClNr) %>% 
  summarize(paid = sum(Pay))
future_paid_by_claim <- dataset$cashflow_history %>%
  inner_join(claim_ids_for_comparison, by = "ClNr") %>%
  group_by(ClNr) %>% 
  summarize(ultimate = sum(Pay)) %>% 
  inner_join(already_paid_by_claim) %>% 
  mutate(future_paid = ultimate - paid)

predicted_future_paid$mean_predicted_future_paid %>% max()
predicted_future_paid$mean_predicted_future_paid %>% min()
future_paid_by_claim$future_paid %>% max()
future_paid_by_claim$future_paid %>% min()


predicted_future_paid %>% 
  filter(mean_predicted_future_paid > 0) %>% 
  ggplot(aes(x = mean_predicted_future_paid)) +
  geom_histogram(binwidth = 100)


future_paid_by_claim %>% 
  filter(future_paid > 0) %>% 
  ggplot(aes(x = future_paid)) +
  geom_histogram(binwidth = 100)

dists[[1]]$distribution$distribution$stddev()
dists[[1]]$distribution$components[[2]]$distribution$loc
dists[[1]]$distribution$components[[2]]$distribution$mean()
dists[[1]]$distribution$components[[2]]$distribution$scale
dists[[1]]$distribution$components[[2]]$distribution$variance()


dists[[1]]$distribution$cat$probs
dists[[1]]$distribution$components[[1]]$distribution$mean()
dists[[1]]$distribution$components[[1]]$distribution$scale
dists[[1]]$distribution$components[[1]]$distribution$variance()

