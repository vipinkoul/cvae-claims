keras_model_cvae <- function(name = NULL, num_categories = 10L, num_latent_distributions = 20L, temperature = 0.5, beta = 1,
                             mean_paid = 0, sd_paid = 1) {
  keras_model_custom(name = name, function(self) {

    #' Initialize parameters
    self$num_categories <- num_categories
    self$num_latent_distributions <- num_latent_distributions
    #' Temperature for conrete distribution, approches categorical as temperature -> 0
    self$temperature <- temperature
    #' $\beta$-VAE parameter
    self$beta <- beta

    #' Input layers
    
    #' Future values we're trying to predict/generate
    self$paid_loss_target <- layer_input(shape = c(11, 1), name = "paid_loss_target_")
    
    #' Past paid values
    self$paid_loss_lags <- layer_input(shape = c(11, 1), name = "paid_loss_lags_")
    
    #' Past claim status, 1 for open 0 for closed, one-hot encoded
    self$claim_open_indicator_lags <- layer_input(shape = c(11, 2), name = "claim_open_indicator_lags_")
    
    #' How long has the claim developed, in years divided by 11, so this input is in [1/11, 1]
    self$scaled_dev_year <- layer_input(shape = c(1), name = "scaled_dev_year_")
    
    #' Line of business, integer indexed
    self$lob <- layer_input(shape = c(1), name = "lob_")
    
    #' Occupation of claimant, integer indexed
    self$claim_code <- layer_input(shape = c(1), name = "claim_code_")
    
    #' Age, normalized
    self$age <- layer_input(shape = 1, name = "age_")
    
    #' Injured body part, integer indexed
    self$injured_part <- layer_input(shape = 1, name = "injured_part_")

    #' Condition encoder
    
    #' Categorical variables go into embedding layers
    self$lob_embedding <- self$lob %>%
      layer_embedding(4, 2) %>%
      layer_flatten()
    self$claim_code_embedding <- self$claim_code %>%
      layer_embedding(53, 16) %>%
      layer_flatten()
    self$injured_part_embedding <- self$injured_part %>%
      layer_embedding(90, 16) %>%
      layer_flatten()
    
    #' Historical sequences of paid losses and claim statuses processed by GRU
    self$paid_loss_lags_gru <- layer_gru(units = 128)
    self$paid_loss_lags_gru_out <- self$paid_loss_lags %>%
      layer_masking(mask_value = -99) %>%
      self$paid_loss_lags_gru()
    self$claim_open_indicator_lags_gru_out <- self$claim_open_indicator_lags %>%
      layer_masking(-99) %>%
      layer_gru(units = 128)

    #' Concatenate encoded conditions
    self$cond <- layer_concatenate(list(
      self$scaled_dev_year,
      self$lob_embedding,
      self$claim_code_embedding,
      self$injured_part_embedding,
      self$paid_loss_lags_gru_out,
      self$claim_open_indicator_lags_gru_out
    ))


    #' Encode Y
    #' We take the values we want to generate and encode them via GRU
    self$paid_loss_target_gru <- self$paid_loss_target %>%
      layer_masking(mask_value = -99) %>%
      layer_gru(units = 128)

    #' Concatenate X with Y
    self$target_with_cond <- layer_concatenate(list(self$cond, self$paid_loss_target_gru))

    #' CVAE encoder
    #' The encoding happens here, we're going to use the encoded output to parameterize num_latent_distributions concrete distributions
    #'   each with num_categories categories
    self$encoded <- self$target_with_cond %>%
      layer_dense(self$num_categories * self$num_latent_distributions, activation = "relu") %>%
      layer_reshape(c(self$num_latent_distributions, self$num_categories))

    #' Prior is just a one hot categorical distribution
    self$prior <- tfd_one_hot_categorical(logits = k_ones_like(self$encoded))

    #' Sampling
    #' Draw from the distribution parameterized by the encoder. Recall we can't draw from tfd_one_hot_categorical because it's not
    #'   differentiable.
    self$z <- self$encoded %>%
      layer_distribution_lambda(
        make_distribution_fn = function(x) tfd_relaxed_one_hot_categorical(temperature = self$temperature, logits = x),
        #' KL term
        activity_regularizer = layer_kl_divergence_regularizer(distribution_b = self$prior, weight = self$beta)
      )

    #' Flatten the draw
    self$draw_input_flattened <- self$z %>%
      layer_reshape(c(self$num_categories * self$num_latent_distributions))

    #' Concatenate draw from q(z|X, Y) with encoded X
    self$z_cond <- layer_concatenate(list(self$cond, self$draw_input_flattened), axis = 1)

    #' Initialize layers used for decoding
    self$gru_1 <- layer_gru(units = 128, return_sequences = TRUE)

    #' This is our "output" layer (which is the main thing in the architecture we want to get right). 
    #' The input to this layer will be (batch, 11, 7) with relu activation.
    #' We output an 11-dimensional distribution.
    #'
    #' We use a mixture. Justification is (loosely) as follows:
    #'   At each time step, we either have a payment (positive), a recovery (e.g. we get paid by another insurer, 
    #'   which would mean a cash inflow (negative)), or nothing (0). The first three values of the 7-element tensor
    #'   slice parameterize the probabilities of getting each of these. The first one is a degenerate distribution
    #'   with all of the density at 0. The second one is a normal with a positive loc (for cash outflow). The third is another normal
    #'   with a negative loc (for cash inflow). The params of the normals are parameterized by the remaining elements of the
    #'   tensor slice.
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

    # self$prob_output_layer <- layer_distribution_lambda(
    #   make_distribution_fn = function(x) {
    #     td <- tfd_transformed(
    #       distribution = tfd_normal(
    #         loc = x[, , 4],
    #         scale = k_softplus(x[, , 5])
    #       ),
    #       bijector = tfb_masked_autoregressive_flow(
    #         masked_autoregressive_default_template(
    #           hidden_layers = c(32, 32),
    #           shift_only = TRUE
    #         ),
    #         is_constant_jacobian = TRUE
    #       ),
    #       event_shape = tf$TensorShape(1)
    #     )
    #     tfd_independent(td, reinterpreted_batch_ndims = NULL)
    #   }
    # )

    self$dense_7 <- layer_dense(units = 7, activation = "relu")

    #' Decoder
    #' Take the concatenation of the draw and the encoded static conditions (age, injury type, etc.), repeat it 11 times
    #'   then feed it through a GRU, then each timestep is fed to a dense layer, ending up with (batch, 11, 7). 
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
