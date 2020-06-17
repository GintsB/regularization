################################
# FUNCTIONS FOR REGULARIZATION #
################################
create_corr <- function(r, p) {
  R <- matrix(data = rep(0, p^2), nrow = p, ncol = p)
  for (i in 1:p) {
    for (j in 1:p) {
      R[i,j] <- r ^ abs(i - j)
    }
  }
  return(R)
}

calc_MSE <- function(pred, actual) {
  x <- ((actual - pred) ^ 2) 
  MSE <- mean(x)
  
  return(MSE)
}

# x <- matrix(c(1, 2, 3, 4), nrow = 2)
# x
# x_s <- standardize_data(x)
# cor(x_s)

standardize_data <- function(data) {
  convert <- "matrix" %in% class(data)
  if(convert) {
    data <- as_data_frame(data)
    names(data) <- paste0("x", 1:ncol(data))
  }
  
  data_stand <- data %>% 
    mutate_if(.predicate = str_detect(names(.),"x"), .funs = function(x) (x - mean(x)) / sd(x))
  
  if(convert) {
    data_stand <- as.matrix(data_stand)
  }
  return(data_stand)
}

gen_data <- function(corr, b, s = NULL, n, 
                     standardize = FALSE, 
                     example_4 = FALSE, 
                     logistic_regression = FALSE) {
  
  N <- sum(unlist(n))
  res <- list()
  
  if (example_4) {
    
    Z <- matrix(rnorm(3 * N), nrow = N, ncol = 3)
    X_1 <- bind_cols(
      lapply(1:3, function(n) {
        
      x <- matrix(Z[,n] + rnorm(5 * N, sd = 0.01), ncol = 5, nrow = N) %>% as.data.frame()
      names(x) <- paste0("x", n, "_", 1:5)
      return(x)
      
      }))
    
    X_r <- matrix(rnorm(25 * N), ncol = 25, nrow = N)
    X_r <-   as.data.frame(X_r)
    names(X_r) <- paste0("xr_", 1:25)
    
    X <- bind_cols(X_1, X_r)
    
    if (logistic_regression) {
      pi <- logistic(as.matrix(X) %*% b)
      y <- rbinom(n = length(pi), size = 1, prob = pi)
    } else {
      y <- as.matrix(X) %*% b + rnorm(nrow(X), mean = 0, sd = s)
    }
    
    if (standardize) {X <- standardize_data(X)}
    
    data <- cbind(X, y)
  } else {
    R <- corr
    
    X <- rmvnorm(N, mean = rep(0, p), sigma = R)
    
    if (standardize) {X <- standardize_data(X)}
  
    if (logistic_regression) {
      pi <- logistic(X %*% b)
      y <- rbinom(n = length(pi), size = 1, prob = pi)
    } else {
      y <- X %*% b + rnorm(nrow(X), mean = 0, sd = s)
    }
    
    data <- as.data.frame(X)
    names(data) <- paste0("x", 1:p)
    
    data <- cbind(data, y)
  }
  
  res$data_train <- data[1:n$train,]
  res$data_test <- data[(N - n$test + 1):N,]
  
  return(res)
}

validate_model <- function(training_x, training_y, 
                           regularize = TRUE, 
                           alpha_vec,
                           logistic_regression = FALSE) {
  
  if (!regularize) {
    lambda_param <- 0 # regular OLS
    alpha_param <- 1 # doesn't matter
    cv_dev_sd <- 0 # no CV
  } else if (length(alpha_vec) > 1) {
    
    x <- training_x
    y <- training_y

    cva_fit <- cva.glmnet(x, y, 
                          alpha = alpha_vec, 
                          family = if_else(logistic_regression, "binomial", "gaussian"),
                          type.measure = "deviance", # deviance for binomial and mse for gaussian 
                          nfolds = 5)

    n_alpha <- length(cva_fit$alpha)
     
    cva_dt <- lapply(X = 1:n_alpha, FUN = 
                       function(i) {
                         
                         model <- cva_fit$modlist[[i]]
                         min_mse_i <- which(model$cvm == min(model$cvm))
                         min_mse <-  model$cvm[min_mse_i]
                         min_lambda <- model$lambda[min_mse_i]
                         cv_dev_sd <- model$cvsd[min_mse_i] # deviance is mse for linear regression
                         alpha <- cva_fit$alpha[i]
                         
                         new_df <- data.frame(id = i, 
                                              alpha = alpha, 
                                              min_mse = min_mse, 
                                              min_lambda = min_lambda, 
                                              cv_dev_sd = cv_dev_sd)
                         
                         return(new_df)
                       }
    ) %>% bind_rows()

    best_params <- cva_dt %>%
      filter(min_mse == min(min_mse)) %>%
      select(-min_mse, lambda = min_lambda)

    alpha_param <- best_params$alpha
    lambda_param <- best_params$lambda
    cv_dev_sd <- best_params$cv_dev_sd
  } else {
    
    x <- training_x
    y <- training_y

    cv_fit <- cv.glmnet(x, y, 
                        alpha = alpha_vec, 
                        family = if_else(logistic_regression, "binomial", "gaussian"),
                        type.measure = "deviance", # deviance for binomial and mse for gaussian 
                        nfolds = 5)

    alpha_param <- alpha_vec
    min_i <- which(cv_fit$cvm == min(cv_fit$cvm))
    lambda_param <- cv_fit$lambda[min_i]
    cv_dev_sd <- cv_fit$cvsd[min_i]
  }
  
  return(list(alpha = alpha_param,
              lambda = lambda_param,
              cv_dev_sd = cv_dev_sd))
}

get_model_measures <- function(alpha_param = seq(0, 1, len = 11)^3, 
                               regularize = TRUE, 
                               ex_4 = FALSE,
                               logistic_regression = FALSE) {
  
  data <- gen_data(corr, b, s, n, 
                  standardize = FALSE, 
                  example_4 = ex_4, 
                  logistic_regression = logistic_regression)
  
  # Get data
  x_train <- data$data_train %>% select(-y) 
  x_train <- as.matrix(x_train)
  y_train <- data$data_train$y
  
  x_test <- select(data$data_test, -y) 
  x_test <- as.matrix(x_test)
  y_test <- data$data_test$y
  
  # Validate
  validated_params <- validate_model(x_train, y_train,
                                     regularize = regularize, 
                                     alpha_vec = alpha_param, 
                                     logistic_regression = logistic_regression)
  
  fit_valid <- glmnet(x = x_train, 
                      y = y_train,
                      lambda = validated_params$lambda,  
                      alpha = validated_params$alpha,
                      family = if_else(logistic_regression, "binomial", "gaussian"))
  
  non_zero <- fit_valid$df
  
  param_est <- fit_valid$beta %>% as.numeric()
  names(param_est) <- rownames(fit_valid$beta)
  
  if (logistic_regression) {
    
    # Test
    pred <- predict(object = fit_valid, 
                    newx = x_test, 
                    type = 'response',
                    s = validated_params$lambda, 
                    alpha = validated_params$alpha)
    
    auc <- auc(y_test, pred)
    
    res <- data.frame(non_zero = non_zero,
                      auc = auc,
                      cv_dev_sd = validated_params$cv_dev_sd,
                      param_est = I(list(param_est)))
    
  } else { # linear regression
    d_b <- as.numeric(fit_valid$beta - b)
    mse_real <- t(d_b) %*% corr %*% d_b
    
    # Test
    pred <- predict.glmnet(object = fit_valid, 
                           newx = x_test, 
                           s = validated_params$lambda, 
                           alpha = validated_params$alpha)
    MSE <- calc_MSE(pred, y_test)
    
    res <- data.frame(non_zero = non_zero,
                      mse_real = mse_real,
                      MSE = MSE,
                      cv_dev_sd = validated_params$cv_dev_sd,
                      param_est = I(list(param_est)))
  }
  
  return(res)
}

get_bootsrapped_measures <- function(data, resamplings = 500, p_l = 0.025, p_u = 0.975) {
  
  repl_vec <- replicate(n = resamplings, expr = sampled_median(data))
  
  SE <- calc_MSE(repl_vec, median(data))
  
  Q <- quantile(repl_vec, probs = c(p_l, p_u))
  
  res <- list()
  res$SE <- SE
  res$Q  <- Q
  
  return(res)
}

sampled_median <- function(data){
  
  s <- sample(data, size = length(data), replace = TRUE) 
  
  sampled_median <- median(s)
  
  return(sampled_median)
  
}

repl_simulation <- function(n_repl = 50, 
                            name = "lasso", 
                            alpha = 1, 
                            regularize = TRUE, 
                            example_4 = FALSE, 
                            parallel = FALSE,
                            logistic_regression = FALSE) {
  
  if (parallel) {
    cl <- makeCluster(future::availableCores())  
    
    registerDoParallel(cl)
    clusterCall(cl, function(x) .libPaths(x), .libPaths())
    clusterCall(cl, function() 
    {
      library(dplyr)
      library(mvtnorm)
      library(stringr)
      library(glmnet)
      library(glmnetUtils)
      source("functions.R")
    })
    
    clusterSetRNGStream(cl)
    
    clusterExport(cl,c("n", "b", "p", "corr", "n_repl", "alpha", "regularize", "example_4"), envir = environment())
    
    
    list <- parLapply(cl, 
                     1:n_repl, 
                     function(i,...) {
                       get_model_measures(alpha_param = alpha, 
                                          regularize = regularize, 
                                          ex_4 = example_4, 
                                          logistic_regression = logistic_regression)
                     } )
    
    df <- bind_rows(list)
    
    stopCluster(cl)
    
  } else {
    
    list <- lapply(X = 1:n_repl, 
                   FUN = function(x) {
                     get_model_measures(alpha_param = alpha, 
                                        regularize = regularize, 
                                        ex_4 = example_4, 
                                        logistic_regression = logistic_regression)
                     })
    df <- bind_rows(list)
    
  }
  
  if (logistic_regression) {
    
    bootsrap_res_auc <- get_bootsrapped_measures(df$auc)
    bootsrap_res_nnzero <- get_bootsrapped_measures(df$non_zero)
    bootsrap_res_cvsd <- get_bootsrapped_measures(df$cv_dev_sd)
    
    res <- data.frame(method = name,
                      median_auc = median(df$auc),
                      med_auc_SE = bootsrap_res_auc$SE,
                      med_auc_l = bootsrap_res_auc$Q[1],
                      med_auc_u = bootsrap_res_auc$Q[2],
                      auc_vec = df$auc,
                      median_nnzero = median(df$non_zero),
                      nnzero_SE = bootsrap_res_nnzero$SE,
                      nnzero_l = bootsrap_res_nnzero$Q[1],
                      nnzero_u = bootsrap_res_nnzero$Q[2],
                      nnzero_vec = df$non_zero,
                      median_cvsd = median(df$cv_dev_sd),
                      mcvsd_SE = bootsrap_res_cvsd$SE,
                      mcvsd_l = bootsrap_res_cvsd$Q[1],
                      mcvsd_u = bootsrap_res_cvsd$Q[2],
                      cvsd_vec = df$cv_dev_sd,
                      param_est_vec = I(df$param_est))
    
  } else { # linear regression
   
    bootsrap_res_MSE <- get_bootsrapped_measures(df$MSE)
    bootsrap_res_nnzero <- get_bootsrapped_measures(df$non_zero)
    bootsrap_res_mse_real <- get_bootsrapped_measures(df$mse_real)
    bootsrap_res_cvsd <- get_bootsrapped_measures(df$cv_dev_sd)
    
    res <- data.frame(method = name,
                      median_MSE = median(df$MSE),
                      mMSE_SE = bootsrap_res_MSE$SE,
                      mMSE_l = bootsrap_res_MSE$Q[1],
                      mMSE_u = bootsrap_res_MSE$Q[2],
                      MSE_vec = df$MSE,
                      median_nnzero = median(df$non_zero),
                      nnzero_SE = bootsrap_res_nnzero$SE,
                      nnzero_l = bootsrap_res_nnzero$Q[1],
                      nnzero_u = bootsrap_res_nnzero$Q[2],
                      nnzero_vec = df$non_zero,
                      median_mse_real = median(df$mse_real),
                      mse_real_SE = bootsrap_res_mse_real$SE,
                      mse_real_l = bootsrap_res_mse_real$Q[1],
                      mse_real_u = bootsrap_res_mse_real$Q[2],
                      mse_real_vec = df$mse_real,
                      median_cvsd = median(df$cv_dev_sd),
                      mcvsd_SE = bootsrap_res_cvsd$SE,
                      mcvsd_l = bootsrap_res_cvsd$Q[1],
                      mcvsd_u = bootsrap_res_cvsd$Q[2],
                      cvsd_vec = df$cv_dev_sd,
                      param_est_vec = I(df$param_est))
     
  }
  
  return(res)
}

logistic <- function(x) {
  logistic <- 1 / (1 + exp(-x))
  return(logistic)
}
logistic <- Vectorize(logistic)

# GRAPHICS -----

create_sim_graphs <- function(final_df, b, type = "linear") {
  
  if (type == "linear") {
    
    p <- final_df %>%
      select(method, MSE_vec) %>%
      ggplot(aes(x = method, y = MSE_vec, fill = method)) +
      geom_boxplot() +
      theme_bw()
    
    print(p)
    
    p <- final_df %>%
      select(method, mse_real_vec) %>%
      ggplot(aes(x = method, y = mse_real_vec, fill = method)) +
      geom_boxplot() +
      theme_bw()
    
    print(p)
    
  } else if (type == "logit") {
    
    p <- final_df %>% 
      select(method, auc_vec) %>% 
      ggplot(aes(x = method, y = auc_vec, fill = method)) +
      geom_boxplot() +
      theme_bw()
    
    print(p)
    
  }
  
  
  p <- final_df %>%
    filter(method %in% c("lasso", "elastic net")) %>%
    select(method, nnzero_vec) %>%
    ggplot(aes(x = method, y = nnzero_vec, fill = method)) +
    geom_boxplot() +
    theme_bw()
  
  print(p)
  
  p <- final_df %>%
    select(method, cvsd_vec) %>%
    ggplot(aes(x = method, y = cvsd_vec, fill = method)) +
    geom_boxplot() +
    theme_bw()
  
  print(p)
  
  true_param <- as.data.frame(b) %>%
    mutate(name = paste0("b", 1:length(b))) %>%
    rename(true_beta = b) %>%
    spread(key = name, value = true_beta)
  
  true_param_df <- data.frame(method = unique(final_df$method)) %>%
    cbind(true_param) %>%
    gather(key = "parameter", value = "value", -method)
  
  p <- final_df %>%
    select(method, param_est_vec) %>%
    mutate(x = map(param_est_vec, ~as_tibble(split(., paste0("b", 1:length(b)))))) %>%
    unnest(x, .drop = FALSE) %>%
    select(-param_est_vec) %>%
    gather(key = "parameter", value = "value", -method) %>%
    ggplot(aes(x = parameter, y = value)) +
    geom_boxplot() +
    geom_boxplot(data = true_param_df, col = "red") +
    facet_wrap(~method) +
    theme_bw()
  
  print(p)
  
}

create_class_hist <- function(corr, b, s, n, example_4 = FALSE, N = 1000) {
  
  m <- replicate(n = N, expr = {
    data <- gen_data(corr, b, s, n, 
                     standardize = FALSE, 
                     example_4 = example_4, 
                     logistic_regression = TRUE)
    c(data$data_train$y, data$data_test$y) %>% mean()
  })
  
  hist(m)
  
  res <- list()
  
  res$q_l <- quantile(m, probs = 0.025)
  res$q_u <- quantile(m, probs = 0.975)
  
  return(res)
  
}

# Real data main functions -----

get_model_measures_rd <- function(data_train, data_test,
                                  alpha_param = seq(0, 1, len = 11)^3, 
                                  regularize = TRUE,
                                  internal_method = FALSE) {
  
  # Get data
  x_train <- select(data_train, -y)
  x_train <- as.matrix(x_train)
  y_train <- data_train$y
  
  x_test <- select(data_test, -y)
  x_test <- as.matrix(x_test)
  y_test <- data_test$y
  
  if (internal_method) {
    
    #### REDACTED ####
  
  } else {
   
    # Validate
    validated_params <- validate_model(x_train, y_train,
                                       regularize = regularize, 
                                       alpha_vec = alpha_param, 
                                       logistic_regression = TRUE)
    
    fit_valid <- glmnet(x = x_train, 
                        y = y_train,
                        lambda = validated_params$lambda,  
                        alpha = validated_params$alpha,
                        family = "binomial")
    
    non_zero <- fit_valid$df
    
    param_est <- fit_valid$beta %>% as.numeric()
    names(param_est) <- rownames(fit_valid$beta)
    
    # Test
    pred <- predict(object = fit_valid, 
                    newx = x_test, 
                    type = 'response',
                    s = validated_params$lambda, 
                    alpha = validated_params$alpha)
    
    auc <- auc(y_test, pred)
     
  }
  
  res <- data.frame(non_zero = non_zero,
                    auc = auc,
                    cv_dev_sd = validated_params$cv_dev_sd,
                    param_est = I(list(param_est)))
  
  
  return(res)
}

cv_rd <- function(data, k_folds = 10, alpha_param, regularize, internal_method = FALSE) {
  
  folds <- cut(seq(1, nrow(data)), breaks = 10, labels = FALSE)
  
  res <- list()
  
  for (i in 1:k_folds){
    row_i <- which(folds == i, arr.ind = TRUE)
    data_test <- data[row_i, ]
    data_train <- data[-row_i, ]
    res[[i]] <- get_model_measures_rd(data_train = data_train, 
                                      data_test = data_test, 
                                      alpha_param = alpha_param, 
                                      regularize = regularize,
                                      internal_method = internal_method)
  }
  
  res_df <- bind_rows(res)

  return(res_df)
    
}

mat <- data.frame(a = c(1,2,3), b = c(2,2,3), c = c(1,-3,5)) %>% as.matrix()
corr_filter <- function(mat, max_corr) {
  
  c <- cor(mat)
  
  i_col <- 1
  while (i_col <= ncol(mat)) {
    high_cor_i <- which(c[,i_col] > max_corr)
    high_cor_i <- high_cor_i[-which(names(high_cor_i) == colnames(mat)[i_col])]
    high_cor_names <- names(high_cor_i)
    if (length(high_cor_names) == 0) {
      i_col <- i_col + 1
    } else {
      c <- c[-high_cor_i, -high_cor_i]
      mat <- mat[, -high_cor_i]
    }
  }
  
  return(mat)  
}
