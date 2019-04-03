benchmark_tbl <- results %>% mutate(
  nn_error = nn_ultimate / actual_ultimate - 1,
  mack_error = mack_ultimate / actual_ultimate - 1
)

benchmark_tbl %>% 
  gather("model", "ultimate_relative_error", nn_error, mack_error) %>% 
  ggplot(aes(model, ultimate_relative_error)) +
  geom_boxplot() +
  coord_flip() +
  theme_bw() +
  stat_summary(fun.y = "mean", geom = "point", colour = "red", shape = 15, size = 2)

benchmark_tbl %>%
  summarize(mean_nn_error = mean(nn_error), mean_mack_error = mean(mack_error))
