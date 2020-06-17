###################################################
# FILE FOR SIMULATIONS STUDIES FOR REGULARIZATION #
###################################################

# Model of form y = X * b + s * e,
# e ~ N(0; 1).
# Notation of training nr./validation nr./test nr. is used.

#### LIBRARIES ----
library(tidyverse)
library(mvtnorm)
library(stringr)
library(glmnet)
library(glmnetUtils)
library(ggplot2)

library(doParallel)

source("Functions.R")

#### EXAMPLE 1 ----

# 50 datasets 160/320 observations and 8 predictors.
# b = (3; 1.5; 0; 0; 2; 0; 0; 0).
# s = 3.
# corr(xi; xj) = 0.5^|i - j|

r <- 0.5
b <- rep(c(3, 1.5, 0, 0, 2, 0, 0, 0), times = 1)
p <- length(b)
s <- 3
n <- list(train = 160, test = 320)
repl <- 50

corr <- create_corr(r, p)

final_df <- rbind(repl_simulation(repl, "regular", 1, FALSE)
                  ,repl_simulation(repl, "lasso", 1, TRUE)
                  ,repl_simulation(repl, "ridge", 0, TRUE)
                  ,repl_simulation(repl, "elastic net", seq(0, 1, len = 11)^3, TRUE)
)

# save(final_df, file = "example_1.rda")

pairwise.wilcox.test(x = final_df$mse_real_vec, g = final_df$method, paired = TRUE)

final_df %>%
  select(-MSE_vec, -nnzero_vec, -mse_real_vec, -cvsd_vec, -param_est_vec) %>%
  distinct()

create_sim_graphs(final_df, b)


#### EXAMPLE 2 ----

# same as EXAMPLE 1 with
# b = rep(0.85, 8).

r <- 0.5
b <- rep(0.85, 8)
p <- length(b)
s <- 3
n <- list(train = 160, test = 320)

corr <- create_corr(r, p)

final_df <- rbind(repl_simulation(repl, "regular", 1, FALSE)
                  ,repl_simulation(repl, "lasso", 1, TRUE)
                  ,repl_simulation(repl, "ridge", 0, TRUE)
                  ,repl_simulation(repl, "elastic net", seq(0, 1, len = 11)^3, TRUE)
)

save(final_df, file = "example_2.rda")
# save(final_df_100, file = "example_2_100.rda")
# save(final_df_100_40var, file = "example_2_100_40var.rda")

pairwise.wilcox.test(x = final_df$mse_real_vec, g = final_df$method, paired = TRUE)

final_df %>%
  select(-MSE_vec, -nnzero_vec, -mse_real_vec, -cvsd_vec, -param_est_vec) %>%
  distinct()

create_sim_graphs(final_df, b)

#### EXAMPLE 3 (elastic net paper) ----

corr <- matrix(rep(0.5, 40*40), nrow = 40, ncol = 40) + 0.5 * diag(40)
b <- c(rep(0, 10), rep(2, 10),rep(0, 10), rep(2, 10))
p <- length(b)
s <- 15
n <- list(train = 800, test = 1600)


final_df <- rbind(repl_simulation(repl, "regular", 1, FALSE)
                  ,repl_simulation(repl, "lasso", 1, TRUE)
                  ,repl_simulation(repl, "ridge", 0, TRUE)
                  ,repl_simulation(repl, "elastic net", seq(0, 1, len = 11)^3, TRUE)
)

# save(final_df, file = "example_3.rda")

pairwise.wilcox.test(x = final_df$mse_real_vec, g = final_df$method, paired = TRUE)

final_df %>%
  select(-MSE_vec, -nnzero_vec, -mse_real_vec, -cvsd_vec, -param_est_vec) %>%
  distinct()

create_sim_graphs(final_df, b)

#### EXAMPLE 4 (elastic net paper) ----

# 4th examples corr matrix ir variance-covariance matrix. It isn't used for data generation
# but instead for real_MSE (FMSE) calculation.
corr <- diag(40)
for (i in 1:15) {
  corr[i, i] <- 1.01
  for (j in 1:15) {
    if ((i - 1) %/% 5 == (j - 1) %/% 5 & i != j & i <= 15) {
      corr[i, j] <- 1
    }
  }
}
b <- c(rep(3, 15), rep(0, 25))
p <- length(b)
s <- 15
n <- list(train = 800, test = 1600)
repl <- 50


final_df <- rbind(repl_simulation(repl, "regular", 1, FALSE, example_4 = TRUE)
                  ,repl_simulation(repl, "lasso", 1, TRUE, example_4 = TRUE)
                  ,repl_simulation(repl, "ridge", 0, TRUE, example_4 = TRUE)
                  ,repl_simulation(repl, "elastic net", seq(0, 1, len = 11)^3, TRUE, example_4 = TRUE)
)

# save(final_df, file = "example_4.rda")

pairwise.wilcox.test(x = final_df$mse_real_vec, g = final_df$method, paired = TRUE)


final_df %>% 
  select(-MSE_vec, -nnzero_vec, -mse_real_vec, -cvsd_vec, -param_est_vec) %>% 
  distinct()

create_sim_graphs(final_df, b)

true_param <- as.data.frame(b) %>%
  mutate(name = paste0("b", 1:length(b))) %>%
  rename(true_beta = b) %>%
  spread(key = name, value = true_beta)

true_param_df <- data.frame(method = unique(final_df$method)) %>%
  cbind(true_param) %>%
  gather(key = "parameter", value = "value", -method)

final_df %>%
  select(method, param_est_vec) %>%
  mutate(x = map(param_est_vec, ~as_tibble(split(., paste0("b", 1:length(b)))))) %>%
  unnest(x, .drop = FALSE) %>%
  select(-param_est_vec) %>%
  gather(key = "parameter", value = "value", -method) %>%
  ggplot(aes(x = reorder(parameter, str_extract(parameter, pattern = "\\d+") %>% as.numeric()), y = value)) +
  geom_boxplot() +
  geom_boxplot(data = true_param_df, col = "red") +
  facet_wrap(~method) +
  theme_bw()

#### LOGISTIC REGRESSION ##### ------

#### EXAMPLE 1 ----

r <- 0.5
b <- c(3, 1.5, 0, 0, 2, 0, 0, 0)
p <- length(b)
n <- list(train = 160, test = 320)
repl <- 50

corr <- create_corr(r, p)

final_df <- rbind(repl_simulation(repl, "regular", 1, FALSE, logistic_regression = TRUE)
                  ,repl_simulation(repl, "lasso", 1, TRUE, logistic_regression = TRUE)
                  ,repl_simulation(repl, "ridge", 0, TRUE, logistic_regression = TRUE)
                  ,repl_simulation(repl, "elastic net", seq(0, 1, len = 11)^3, TRUE, logistic_regression = TRUE)
)

# save(final_df, file = "log_example_1.rda")

pairwise.wilcox.test(x = final_df$auc_vec, g = final_df$method, paired = TRUE)

final_df %>% 
  select(-auc_vec, -nnzero_vec, -cvsd_vec, -param_est_vec) %>% 
  distinct()

create_sim_graphs(final_df, b, type = "logit")

create_class_hist(corr, b, s, n)

#### EXAMPLE 2 ----

# same as EXAMPLE 1 with
# b = rep(0.85, 8).

r <- 0.5
b <- rep(0.85, 8)
p <- length(b)
n <- list(train = 160, test = 320)
repl <- 50
corr <- create_corr(r, p)

final_df <- rbind(repl_simulation(repl, "regular", 1, FALSE, logistic_regression = TRUE)
                  ,repl_simulation(repl, "lasso", 1, TRUE, logistic_regression = TRUE)
                  ,repl_simulation(repl, "ridge", 0, TRUE, logistic_regression = TRUE)
                  ,repl_simulation(repl, "elastic net", seq(0, 1, len = 11)^3, TRUE, logistic_regression = TRUE)
)

# save(final_df, file = "log_example_2.rda")

pairwise.wilcox.test(x = final_df$auc_vec, g = final_df$method, paired = TRUE)

final_df %>% 
  select(-auc_vec, -nnzero_vec, -cvsd_vec, -param_est_vec) %>% 
  distinct()

create_sim_graphs(final_df, b, type = "logit")

create_class_hist(corr, b, s, n)

#### EXAMPLE 3 (elastic net paper) ----

corr <- matrix(rep(0.5, 40*40), nrow = 40, ncol = 40) + 0.5 * diag(40)
b <- c(rep(0, 10), rep(2, 10),rep(0, 10), rep(2, 10))
p <- length(b)
n <- list(train = 800, test = 1600)
repl <- 50

final_df <- rbind(repl_simulation(repl, "regular", 1, FALSE, logistic_regression = TRUE)
                  ,repl_simulation(repl, "lasso", 1, TRUE, logistic_regression = TRUE)
                  ,repl_simulation(repl, "ridge", 0, TRUE, logistic_regression = TRUE)
                  ,repl_simulation(repl, "elastic net", seq(0, 1, len = 11)^3, TRUE, logistic_regression = TRUE)
)

# save(final_df, file = "log_example_3.rda")

pairwise.wilcox.test(x = final_df$auc_vec, g = final_df$method, paired = TRUE)

final_df %>% 
  select(-auc_vec, -nnzero_vec, -cvsd_vec, -param_est_vec) %>% 
  distinct()

create_sim_graphs(final_df, b, type = "logit")

create_class_hist(corr, b, s, n)

#### EXAMPLE 4 (elastic net paper) ----

corr <- diag(40)
for (i in 1:15) {
  corr[i, i] <- 1.01
  for (j in 1:15) {
    if ((i - 1) %/% 5 == (j - 1) %/% 5 & i != j & i <= 15) {
      corr[i, j] <- 1
    }
  }
}
b <- c(rep(3, 15), rep(0, 25))
n <- list(train = 800, test = 1600)
repl <- 50

final_df <- rbind(repl_simulation(repl, "regular", 1, FALSE, example_4 = TRUE, logistic_regression = TRUE)
                  ,repl_simulation(repl, "lasso", 1, TRUE, example_4 = TRUE, logistic_regression = TRUE)
                  ,repl_simulation(repl, "ridge", 0, TRUE, example_4 = TRUE, logistic_regression = TRUE)
                  ,repl_simulation(repl, "elastic net", seq(0, 1, len = 11)^3, TRUE, example_4 = TRUE, logistic_regression = TRUE)
)

# save(final_df, file = "log_example_4.rda")

pairwise.wilcox.test(x = final_df$auc_vec, g = final_df$method, paired = TRUE)

final_df %>% 
  select(-auc_vec, -nnzero_vec, -cvsd_vec, -param_est_vec) %>% 
  distinct()

create_sim_graphs(final_df, b, type = "logit")

create_class_hist(corr, b, s, n, example_4 = TRUE)
