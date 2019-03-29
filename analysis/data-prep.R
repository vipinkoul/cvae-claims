record_year_cutoff <- 2005
timesteps <- 11

simulated_cashflows <- readr::read_delim("Simulated.Cashflow.txt", delim = ";", col_types = "dccdddcddddddddddddddddddddddddd")

#' Take a subset
set.seed(99)
simulated_cashflows <- simulated_cashflows %>%
  sample_n(10000)

#' Initial feature engineering
simulated_cashflows <- simulated_cashflows %>%
  gather(key, value, Pay00:Open11) %>%
  arrange(ClNr) %>%
  separate(key, into = c("variable", "year"), sep = "(?<=[a-z])(?=[0-9])") %>%
  mutate(year = as.integer(year)) %>%
  mutate(
    # Report year denotes when the claim was reported
    report_year = AY + RepDel,
    # Calendar year is the accounting year of the transaction
    calendar_year = AY + year,
    # Record year is the year in which the data becomes available,
    #  it must be after the claim was reported
    record_year = pmax(report_year, calendar_year)
  ) %>%
  spread(variable, value) %>%
  mutate(
    paid_loss = Pay,
    lob = LoB,
    claim_code = cc,
    injured_part = inj_part,
    claim_open_indicator = Open
  ) %>%
  group_by(ClNr) %>%
  mutate(cumulative_paid_loss = cumsum(paid_loss)) %>%
  ungroup() %>%
  mutate(
    paid_loss_original = paid_loss,
    cumulative_paid_loss_original = cumulative_paid_loss
  )

training_data <- simulated_cashflows %>%
  filter(record_year <= record_year_cutoff)
claim_ids <- training_data %>%
  distinct(ClNr)

rec <- recipe(training_data, ~ .) %>%
  step_integer(lob, claim_code, injured_part, zero_based = TRUE) %>%
  step_center(age) %>%
  step_scale(age, paid_loss) %>%
  prep(training_data)

#' Capture mean/sd for paids so we can recover after prediction
mean_pay <- 0
sd_pay <- rec$steps[[3]]$sds[["paid_loss"]]

training_data <- training_data %>% 
  mutate_sequences(rec, timesteps)