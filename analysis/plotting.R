claim_ids_for_comparison <- tidied %>%
  distinct(ClNr)

claim_numbers <- tidied %>% distinct(ClNr)

claim_number <- claim_numbers %>%
  sample_n(1) %>%
  pull(ClNr)

forecast_df <- tidied %>%
  filter(ClNr == !! claim_number)

actual_series <- simulated_cashflows %>%
  filter(ClNr == !!claim_number) %>%
  transmute(
    ClNr = ClNr,
    development_year = year,
    paid_loss = paid_loss
  )

plot_forecasts(forecast_df, actual_series)

