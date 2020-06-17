###################################
# REGULARIZED MODEL FOR REAL DATA #
###################################

library(tidyverse)
library(mvtnorm)
library(stringr)
library(glmnet)
library(glmnetUtils)

loan("anon_real_data.rda")

source("Functions.R")


final_rd <- rbind(cv_rd(data_df, alpha_param = 1, regularize = FALSE) %>% mutate(method = "mle"),
                  cv_rd(data_df, alpha_param = 1, regularize = TRUE) %>% mutate(method = "lasso"),
                  cv_rd(data_df, alpha_param = 0, regularize = TRUE) %>% mutate(method = "ridge regression"),
                  cv_rd(data_df, alpha_param = seq(0, 1, len = 11)^3, regularize = FALSE) %>% mutate(method = "elastic net"))

summary <- final_rd %>% 
  group_by(method) %>% 
  summarize(median_auc = median(auc),
            med_auc_l = get_bootsrapped_measures(data = auc)[[2]][1],
            med_auc_u = get_bootsrapped_measures(data = auc)[[2]][2],
            nnzero_l = get_bootsrapped_measures(data = non_zero)[[2]][1],
            nnzero_u = get_bootsrapped_measures(data = non_zero)[[2]][2])

final <- left_join(final_rd, summary, by = "method")

final %>% 
  ggplot(aes(x = method, y = auc)) +
  geom_boxplot()

# save(final_rd, file = "log_example_rd.rda")
# save(final, file = "log_example_rd_2.rda")


final %>%
  select(method, param_est) %>%
  mutate(x = map(param_est, ~as_tibble(split(., paste0("b", 1:length(b)))))) %>%
  unnest(x, .drop = FALSE) %>%
  select(-param_est) %>%
  gather(key = "parameter", value = "value", -method) %>%
  ggplot(aes(x = parameter, y = value)) +
  geom_boxplot() +
  facet_wrap(~method) +
  theme_bw()

#### INTERNAL -----

#### REDACTED ####

pairwise.wilcox.test(x = final$auc, g = final_df$method)
