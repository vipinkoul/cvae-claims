claim_ids_for_comparison <- tidied %>%
  distinct(ClNr)
mack_ult <- compute_mack_ultimate(training_data, claim_ids_for_comparison)

predicted_future_paid <- tidied %>%
  group_by(ClNr, sample) %>%
  summarize(predicted_future_paid = sum(paid_loss)) %>%
  group_by(ClNr) %>%
  summarize(mean_predicted_future_paid =mean(predicted_future_paid))

predicted_df <- training_data %>%
  group_by(ClNr) %>%
  summarize(paid = sum(Pay)) %>%
  right_join(predicted_future_paid, by = "ClNr") %>%
  mutate(ultimate = paid + ifelse(is.na(mean_predicted_future_paid), 0, mean_predicted_future_paid))

nn_ult <- sum(predicted_df$ultimate)

true_ultimates <- simulated_cashflows %>%
  inner_join(claim_ids_for_comparison, by = "ClNr") %>%
  group_by(ClNr) %>%
  summarize(true_ultimate = sum(Pay))

true_ult <- true_ultimates$true_ultimate %>% sum()

cat("Mack error: ", mack_ult / true_ult - 1)
cat("NN error: ", nn_ult / true_ult - 1)


