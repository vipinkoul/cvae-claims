keras_model_cvae <- function(name = NULL, num_categories = 10L, num_latent_distributions = 20L, temperature = 0.5, beta = 1,
                             mean_paid = 0, sd_paid = 1) {
  
  keras_model_custom(name = name, function(self) {
    
    # Initialize parameters
    self$num_categories <- num_categories
    self$num_latent_distributions <- num_latent_distributions
    self$temperature <- temperature
    self$beta <- beta
    
    # Input layers
    self$paid_loss_target <- layer_input(shape = c(11, 1), name = "paid_loss_target_")
    self$paid_loss_lags <- layer_input(shape = c(11, 1), name = "paid_loss_lags_")
    self$claim_open_indicator_lags <- layer_input(shape = c(11, 2), name = "claim_open_indicator_lags_")
    self$scaled_dev_year <- layer_input(shape = c(1), name = "scaled_dev_year_")
    self$lob <- layer_input(shape = c(1), name = "lob_")
    self$claim_code <- layer_input(shape = c(1), name = "claim_code_")
    self$age <- layer_input(shape = 1, name = "age_")
    self$injured_part <- layer_input(shape = 1, name = "injured_part_")
    
    # Condition encoder
    self$lob_embedding <- self$lob %>%
      layer_embedding(4, 2) %>%
      layer_flatten()
    self$claim_code_embedding <- self$claim_code %>%
      layer_embedding(53, 16) %>%
      layer_flatten()
    self$injured_part_embedding <- self$injured_part %>%
      layer_embedding(90, 16) %>%
      layer_flatten()
    self$paid_loss_lags_gru <- layer_gru(units = 128)
    self$paid_loss_lags_gru_out <- self$paid_loss_lags %>%
      layer_masking(mask_value = -99) %>%
      self$paid_loss_lags_gru()
    self$claim_open_indicator_lags_gru_out <- self$claim_open_indicator_lags %>% 
      layer_masking(-99) %>%
      layer_gru(units = 128)
    
    # Concatenate encoded conditions
    self$cond <- layer_concatenate(list(
      self$scaled_dev_year,
      self$lob_embedding, 
      self$claim_code_embedding, 
      self$injured_part_embedding, 
      self$paid_loss_lags_gru_out,
      self$claim_open_indicator_lags_gru_out
    ))
    
    
    # Encode Y
    self$paid_loss_target_gru <- self$paid_loss_target %>%
      layer_masking(mask_value = -99) %>%
      layer_gru(units = 128)
    
    # Concatenate X with Y
    self$target_with_cond <- layer_concatenate(list(self$cond, self$paid_loss_target_gru))
    
    # CVAE encoder
    self$encoded <- self$target_with_cond %>%
      layer_dense(self$num_categories * self$num_latent_distributions, activation = "relu") %>%
      layer_reshape(c(self$num_latent_distributions, self$num_categories))
    
    
    self$prior <- tfd_one_hot_categorical(logits = k_ones_like(self$encoded))
    
    # Sampling
    self$z <- self$encoded %>%
      layer_distribution_lambda(
        make_distribution_fn = function(x) tfd_relaxed_one_hot_categorical(temperature = self$temperature, logits = x),
        activity_regularizer = layer_kl_divergence_regularizer(distribution_b = self$prior, weight = self$beta)
      )
    
    self$draw_input_flattened <- self$z %>%
      layer_reshape(c(self$num_categories * self$num_latent_distributions))
    
    # Concatenate draw from q(z|X, Y) with encoded X
    self$z_cond <- layer_concatenate(list(self$cond, self$draw_input_flattened), axis = 1)
    
    # Initialize layers used for decoding
    self$gru_1 <- layer_gru(units = 128, return_sequences = TRUE)

    self$prob_output_layer <- layer_distribution_lambda(
      make_distribution_fn = function(x) {
        mix <- tfd_mixture(
          tfd_categorical(logits = x[,,1:3]),
          components = list(
            tfd_deterministic(k_zeros_like(x[,,1])),
            tfd_normal(loc = x[,,4], scale = k_softplus(x[,,5])),
            tfd_normal(loc = -x[,,6], scale = k_softplus(x[,,7]))
          )
        )
        tfd_independent(mix, reinterpreted_batch_ndims = NULL)
      }
    )
    
    self$dense_7 <- layer_dense(units = 7, activation = "relu")
    
    # Decoder
    self$sequence <- self$z_cond %>%
      layer_repeat_vector(n = 11) %>%
      self$gru_1() %>%
      self$dense_7()
    
    self$predicted_incremental <- self$sequence %>%
      self$prob_output_layer()
    
    self$cvae <- keras_model(
      inputs = c(self$paid_loss_lags, self$claim_open_indicator_lags, self$scaled_dev_year, self$lob, self$claim_code, self$age, self$injured_part, self$paid_loss_target),
      outputs = self$predicted_incremental
    )
    
    # Test time
    self$test_draw_input <- layer_input(shape = c(self$num_latent_distributions, self$num_categories), name = "draw_input_")
    self$test_draw_input_flattened <- self$test_draw_input %>%
      layer_reshape(c(self$num_categories * self$num_latent_distributions))
    self$test_z_cond <- layer_concatenate(list(self$cond, self$test_draw_input_flattened), axis = 1)

    # Test decoder
    self$test_sequence <- self$test_z_cond %>%
        layer_repeat_vector(n = 11) %>%
        self$gru_1() %>%
        self$dense_7()

    self$test_predicted_incremental <- self$test_sequence %>%
      self$prob_output_layer() %>% 
      layer_lambda(function(x) tfd_sample(x)) %>% 
      layer_reshape(c(11, 1))

    self$predictor <- keras_model(
      c(self$test_draw_input, self$paid_loss_lags, self$claim_open_indicator_lags, self$scaled_dev_year, self$lob, self$claim_code, self$age, self$injured_part),
      self$test_predicted_incremental
    )
  })
}
