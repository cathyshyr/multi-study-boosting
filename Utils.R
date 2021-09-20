library(caret)
library(splines)
library(data.table)
library(nlme)
library(randomForest)
library(mboost)
library(expm)
library(matrixStats)
library(doMC)
library(foreach)
library(gridExtra)
library(ggplot2)
library(reshape2)
library(curatedBreastData)
library(FSelector)
library(lme4)
library(plyr)
library(tidyverse)
library(latex2exp)

### Create a vector with all 0s except 1 in the k-th position
make_basis <- function(k, p = 10) replace(numeric(p), k, 1)

### Compute the trace of a matrix
tr <- function(M){
  sum(diag(M))
}

### Sample datasets from curatedOvarianData
# 
# Input:
# edat_orig: list of datasets
# ndat: number of datasets to sample
# nvar: number of predictors to sample
#
# Output:
# edat: list of data sets
#
init_data <- function(edat_orig, ndat, nvar){
  
  edat <- edat_orig
  edat <- edat[sample(1:length(edat), ndat)] # Randomize dataset order
  
  idx <- sample(1:ncol(edat[[1]]), nvar)
  for(i in 1:ndat){
    edat[[i]] <- edat[[i]][,idx]
    edat[[i]] <- as.data.frame(edat[[i]])
    colnames(edat[[i]]) <- paste0("V", 1:nvar)
  }
  return(edat)
}


### Calculate the boosting fit with linear learners (Algorithm 1)
#
# Input:
# edat_train: list of training studies
# edat_test: list of test studies
# lambda_opt: tuning parameter lambda
# lambdak_opt: study-specific tuning parameters
# M_merge_linear: number of boosting iterations for the merged learner
# M_ens_linear: vector of study-specific boosting iterations for the ensemble learner
# learning_rate: learning rate eta
# sigma_eps: residual error variance
#
# Output:
# R: \tilde{R} 
# R_k: \tilde{R}_k
# 
boost_fit <- function(edat_train, edat_test, lambda_opt, lambdak_opt, M_merge_linear, M_ens_linear, learning_rate, sigma_eps){
  
  ############################################################################
  #                                Merged                                    #
  ############################################################################
  # train and test are the (merged) training and test data
  train <- rbindlist(edat_train)
  if(class(edat_test) == "data.frame"){
    test = edat_test
  }else test <- rbindlist(edat_test)
  
  # train_X is the design matrix for the merged data set (\tilde{X})
  train_X <- data.frame(train)
  train_X <- train_X[, names(train_X) != "y", drop = FALSE]
  train_X <- as.matrix(train_X)
  
  # test_X is the design matrix for the test data set (\tilde{X}_0)
  test_X <- data.frame(test)
  test_X <- test_X[, names(test_X) != "y", drop = FALSE]
  test_X <- as.matrix(test_X)
  
  P <- ncol(train_X)
  N <- nrow(train_X)
  ndat <- length(edat_train)
  
  # Calculate R (\tilde{R})
  R <- vector("list", length = M_merge_linear)
  for(m in 1:M_merge_linear){
    R[[m]] <- ((diag(N) - learning_rate * train_X %*% solve(t(train_X) %*% train_X + lambda_opt * diag(P)) %*% t(train_X)) %^% (m - 1))
  }
  B <- solve(t(train_X) %*% train_X + lambda_opt * diag(P)) %*% t(train_X)
  R <- learning_rate * as.matrix(B) %*% Reduce("+", R[1:length(R)])
  
  ############################################################################
  #                             Ensemble                                     #
  ############################################################################
  
  n_k <- vector()
  Z_k <- train_X_k <- R_k <- vector("list", length = length(edat_train))
  
  # Calculate R_k (\tilde{R}_k)
  for(k in 1:length(edat_train)){
    # train_X_k is the design matrix for the kth dataset
    train_X_k[[k]] <- data.frame(edat_train[[k]])
    train_X_k[[k]] <- train_X_k[[k]][, names(train_X_k[[k]]) != "y", drop = FALSE]
    train_X_k[[k]] <- as.matrix(train_X_k[[k]])
    
    p <- ncol(train_X_k[[k]])
    n_k[k] <- nrow(train_X_k[[k]])
    
    R_k[[k]] <- vector("list", length = M_ens_linear[k])
    # Calculate R
    for(m in 1:M_ens_linear[k]){
      R_k[[k]][[m]] <- ((diag(n_k[k]) - learning_rate * train_X_k[[k]] %*% solve(t(train_X_k[[k]]) %*% train_X_k[[k]] + lambdak_opt[k] * diag(p)) %*% t(train_X_k[[k]])) %^% (m - 1))
    }
    B_k <- solve(t(train_X_k[[k]]) %*% train_X_k[[k]] + lambdak_opt[k] * diag(p)) %*% t(train_X_k[[k]])
    R_k[[k]] <- learning_rate * B_k %*% Reduce("+", R_k[[k]][1:length(R_k[[k]])])
  }
  
  return(list(R = R, R_k = R_k))
}


### Function to calculate the component-wise boosting fit with OLS (Algorithm 2)
#
# Input:
# mod_mboost: component-wise boosting model object
# edat_sim: data that the model was trained on
# M_m: number of boosting iterations 
#
#
# Output:
# R: \tilde{R}^{\text{CW}} 
# 
boost_cw_ols_fit <- function(mod_mboost, edat_sim, M_m) {
  
  # Obtain unique indices of selected variables
  sel <- sort(unique(selected(mod_mboost)))
  
  # Obtain response and length of data set
  y <- mod_mboost$response
  n <- length(y)
  
  ############################################
  #      Calculate the test vector v^T       #
  ############################################
  
  # train is the (merged) training data
  if (class(edat_sim) == "list") {
    train <- rbindlist(edat_sim)
  } else if (class(edat_sim) == "data.frame") {
    train <- edat_sim
  }
 
  # train_X is the design matrix for the merged data set
  train_X <- data.frame(train)
  train_X <- train_X[, names(train_X) != "y", drop = FALSE]
  train_X <- as.matrix(train_X)
  
  # Initialize objects for calculating v^T
  selected_variables <- colnames(train_X)[mod_mboost$xselect()]
  p <- ncol(train_X)
  n <- nrow(train_X)
  B <- R_t <- u <- vector("list", length = M_m)
  H <- X <- vector("list", length = M_m + 1)
  t_ind <- vector()
  
  # Calculate H0
  H[[1]] <- matrix(0L, nrow = n, ncol = n)
  u[[1]] <- diag(n)
  
  # Calculate Xt, Ht, and Rt
  for (t in 1:M_m) {
    selected_variables_t <- selected_variables[t]
    t_ind[t] <- which(colnames(train) %in% selected_variables_t)
    X[[t]] <- train_X[, selected_variables_t]
    X[[t]] <- as.matrix(X[[t]])
    B[[t]] <- solve((t(X[[t]]) %*% X[[t]])) %*% t(X[[t]])
    H[[t + 1]] <- X[[t]] %*% solve(t(X[[t]]) %*% X[[t]]) %*% t(X[[t]])
    I_H <- lapply(1:t, function(x)
      diag(n) - learning_rate * H[[x]])
    R_t[[t]] <-
      learning_rate * as.matrix(make_basis(t_ind[t], p)) %*% B[[t]] %*% Reduce("%*%", rev(I_H))
    u[[t]] <- Reduce("%*%", rev(I_H))
  }
  
  # Calculate R
  R <- Reduce("+", R_t)
  
  return(R)
}

### Calculate corrected AIC for Algorithm 1
# 
# Input:
# edat_sim: training data
# M_m: stopping iteration
# lambda: regularization parameter
#
# Output
# AICc: corrected AICc
#
calc_AICc_Alg1 <- function(edat_sim, M_m, lambda, learning_rate){
  
  ############################################################################
  #                                Merged                                    #
  ############################################################################
  # train and test are the (merged) training and test data
  train <- edat_sim
  Y <- train$y

  # train_X is the design matrix for the merged data set (\tilde{X})
  train_X <- data.frame(train)
  train_X <- train_X[, names(train_X) != "y", drop = FALSE]
  train_X <- as.matrix(train_X)
  
  P <- ncol(train_X)
  N <- nrow(train_X)
 
  # Calculate \mathcal{B}_{(m)}
  mathcalBm <- vector("list", length = M_m)
  df <- AICc <- vector()
  for(m in 1:M_m){
    mathcalBm[[m]] <- diag(N) - ((diag(N) - learning_rate * train_X %*% solve(t(train_X) %*% train_X + lambda * diag(P)) %*% t(train_X)) %^% (m + 1))
    df[m] <- tr(mathcalBm[[m]])
    AICc[m] <- (1 + df[m]/N)/(1 - df[m] + 2)/N + log(mean((Y - mathcalBm[[m]] %*% Y)^2))
  }

  return(AICc[M_m])
}

# Function to calculate the transition point for boosting with linear learners (Theorems 1 & 2)
#
# Input:
# edat_train: list of training studies
# edat_test: list of test studies
# f_train: list of mean functions for training data
# f_test: list of mean functions for test data
# Z_train: list of random predictor for training data
# sigma_eps: residual error variance
# wk: study-specific weights
# lambda_opt: lambda for merged data
# lambdak_opt: vector of lambdas for study-specific data
# learning_rate: learning rate eta
# M_merge_linear: stopping iteration for merged model
# M_ens_linear: vector of stopping iterations for study-specific models
# cols_re_list: list of column indices that correspond to predictors with random effects
#
# Output:
# tau_1: lower bound of interval below which merging outperforms ensembling
# tau_2: upper bound of interval above which ensembling outperforms merging
#
tau_range <- function(edat_train, edat_test, f_train, f_test, Z_train, sigma_eps, wk, lambda_opt, lambdak_opt, learning_rate, M_merge_linear, M_ens_linear, cols_re_list){
  
  # K = ndat
  ndat = length(edat_train)
  # P = nvar
  nvar = P = ncol(edat_train[[1]])
  
  # tilde(X_0), design matrix for test data
  X_0 = as.matrix(rbindlist(edat_test))
  
  f_0 = Reduce(c, f_test)
  f = Reduce(c, f_train)
  
  # tilde(X)
  X = as.matrix(rbindlist(edat_train))
  # N
  N <- nrow(X)
 
  # tilde(R)
  R <- vector("list", length = M_merge_linear)
  for(m in 1:M_merge_linear){
    R[[m]] <- ((diag(N) - learning_rate * X %*% solve(t(X) %*% X + lambda_opt * diag(P)) %*% t(X)) %^% (m - 1))
  }
  B <- solve(t(X) %*% X + lambda_opt * diag(P)) %*% t(X)
  R <- learning_rate * as.matrix(B) %*% Reduce("+", R[1:length(R)])
  
  
  # tilde(R)_k
  n_k <- vector()
  X_k <- Z_k <- R_k <- bias_k.list <- f_k <- vector("list", length = ndat)
  
  for(k in 1:ndat){
    # X_k is the design matrix for the kth data set
    X_k[[k]] = as.matrix(edat_train[[k]])
    
    # f_k is the mean function for the kth data set
    f_k[[k]] = as.matrix(f_train[[k]])
    
    # Z_k is the subset of covariates that corresponds to the random effects
    Z_k[[k]] = as.matrix(Z_train[[k]])
    
    p = ncol(X_k[[k]])
    
    n_k[k] = nrow(X_k[[k]])
    
    R_k[[k]] = vector("list", length = M_ens_linear[k])
    # Calculate R
    for(m in 1:M_ens_linear[k]){
      R_k[[k]][[m]] = ((diag(n_k[k]) - learning_rate * X_k[[k]] %*% solve(t(X_k[[k]]) %*% X_k[[k]] + lambdak_opt[k] * diag(p)) %*% t(X_k[[k]])) %^% (m - 1))
    }
    B_k = solve(t(X_k[[k]]) %*% X_k[[k]] + lambdak_opt[k] * diag(p)) %*% t(X_k[[k]])
    R_k[[k]] = learning_rate * B_k %*% Reduce("+", R_k[[k]][1:length(R_k[[k]])])
    
    bias_k.list[[k]] = wk[k] * (X_0 %*% R_k[[k]] %*% f_k[[k]] - f_0)
  }
  
  # Bias terms
  b_ens = Reduce("+", bias_k.list)
  b_merge = X_0 %*% R %*% f - f_0
  
  # Z' 
  Z_prime <- bdiag(Z_k)
  
  # Calculate the transition interval
  denomk_tr <- vector()
  num1k <- denom2k <- denomk <- vector("list", ndat)
  num2 = tr(t(R) %*% t(X_0) %*% X_0 %*% R)
  denom1 = t(Z_prime) %*% t(R) %*% t(X_0) %*% X_0 %*% R %*% Z_prime
  denom1_tr <- tr(denom1)
  for (k in 1:ndat) {
    # wk^2 * tr(R_k^T X_0^T X_0 R_k)
    num1k[[k]] = wk[k]^2 * tr(t(R_k[[k]]) %*% t(X_0) %*% X_0 %*% R_k[[k]])
    
    # wk^2 * Z_k^T R_k^T X_0^T X_0 R_k Z_k
    denom2k[[k]] = wk[k]^2 * t(Z_k[[k]]) %*% t(R_k[[k]]) %*% t(X_0) %*% X_0 %*% R_k[[k]] %*% Z_k[[k]]
    
    # wk^2 * tr(Z_k^T R_k^T X_0^T X_0 R_k Z_k) 
    denomk_tr[k] = wk[k]^2 * tr(t(Z_k[[k]]) %*% t(R_k[[k]]) %*% t(X_0) %*% X_0 %*% R_k[[k]] %*% Z_k[[k]])
  }

  num1 = Reduce("+", num1k)
  squared_bias_ens = t(b_ens) %*% b_ens
  squared_bias_m = t(b_merge) %*% b_merge
  
  
  if(length(cols_re_list) == 1){
    # tr(wk^2 * Z_k^T R_k^T X_0^T X_0 R_k Z_k) 
    denom2_tr = sum(denomk_tr)
    denom = denom1_tr - denom2_tr
    
    return(c((sigma_eps^2 * (num1 - num2) + squared_bias_ens - squared_bias_m)/(nvar/ncol(Z_train[[1]]) * max(denom)),
             (sigma_eps^2 * (num1 - num2) + squared_bias_ens - squared_bias_m)/(nvar/ncol(Z_train[[1]]) * min(denom))))
  }else if(length(cols_re_list) > 1){
    denom <- vector()
    denom2 <- vector("list", length = length(cols_re_list))
    for(i in 1:length(cols_re_list)){
      # wk^2 (Z_k^T R_k^T X_0^T X_0 R_k Z_k)_{ii} = (wk^2 * Z_k^T R_k^T X_0^T X_0 R_k Z_k)_{ii}
      denom2[[i]] = sapply(1:length(denom2k), function(x){diag(denom2k[[x]])[cols_re_list[[i]]]})
      # {sum_{k=1}^K (Z'^T R^T X_0^T X_0 R_k Z')_{k x i, k x i} - wk^2 (Z_k^T R_k^T X_0^T X_0 R_k Z_k)_{ii}}/Jd
      denomj <- vector()
      for(j in 1:length(cols_re_list[[i]])){
        if(ncol(as.matrix(denom2[[i]])) > 1){
          denom2ij <- denom2[[i]][j, ]
        }else denom2ij = denom2[[i]]
        denomj[j] = sum(diag(denom1)[cols_re_list[[i]][j] + ncol(Z_train[[1]]) * (0:(ndat - 1))] - unlist(denom2ij))
      }
      denom[i] = sum(denomj)/length(cols_re_list[[i]])
    }
    
    return(c((sigma_eps^2 * (num1 - num2) + squared_bias_ens - squared_bias_m)/(nvar * max(denom)),
             (sigma_eps^2 * (num1 - num2) + squared_bias_ens - squared_bias_m)/(nvar * min(denom))))
  }
}



# Compute the conditional MSE for merging and ensembling in Proposition 2
#
# Input: 
# mod_mboost: mboost model object
# edat_sim: training data
# M_m: number of boosting iterations
# ind: index j
# f_train: f(\tilde{X})
# cols_re: indices of random effects
# study_k: index of study k
#
# Output
# mse: conditional MSE
# bias_squared: conditional bias_squared
# var: conditional variance
#
proposition2 <- function(mod_mboost, edat_sim, ind, sigma_re, f_train, cols_re, study_k) {
  # Obtain response and length of data set
  y <- mod_mboost$response
  n <- length(y)
  M_m <- length(mod_mboost$xselect())
  
  # Initialize mod_mboostects for Z_k, covYk, train_X_k, and n_k
  if (class(edat_sim) == "list") {
    Z_k <- covYk <- train_X_k <- G_k <- vector("list", length = length(edat_sim))
  } else if (class(edat_sim) == "data.frame") {
    Z_k <- covYk <- train_X_k <- G_k <- vector("list", length = 1)
  }
  n_k <- vector()
  
  # Populate Z_k, covYk, train_X_k, and n_k
  if (class(edat_sim) == "list") {
    num_study = length(edat_sim)
  } else if (class(edat_sim) == "data.frame") {
    num_study = 1
  }
  for (k in 1:num_study) {
    if (class(edat_sim) == "list") {
      train_X_k[[k]] <- data.frame(edat_sim[[k]])
    } else if (class(edat_sim) == "data.frame") {
      train_X_k[[k]] <- data.frame(edat_sim)
    }
    train_X_k[[k]] <-
      train_X_k[[k]][, names(train_X_k[[k]]) != "y", drop = FALSE]
    
    # Z_k is the subset of covariates that corresponds to the random effects
    Z_k[[k]] <- train_X_k[[k]][, sort(cols_re)]
    Z_k[[k]] <- as.matrix(Z_k[[k]])
    
    # G_k is the covariance matrix for the k-th study
    if (all(sigma_re[cols_re] == 0)) {
      G_k[[k]] <-
        matrix(0L, nrow = length(cols_re), ncol = length(cols_re))
    } else
      G_k[[k]] <- diag(sigma_re[cols_re])
    
    n_k[k] <- nrow(train_X_k[[k]])
    covYk[[k]] <-
      Z_k[[k]] %*% G_k[[k]] %*% t(Z_k[[k]]) + sigma_eps ^ 2 * diag(n_k[k])
  }
  
  # Calculate covY
  covY <- bdiag(covYk)
  
  ############################################
  #      Calculate the test vector v^T       #
  ############################################
  
  # train is the (merged) training data
  if (class(edat_sim) == "list") {
    train <- rbindlist(edat_sim)
  } else if (class(edat_sim) == "data.frame") {
    train <- edat_sim
  }
  
  # train_X is the design matrix for the merged data set
  train_X <- data.frame(train)
  train_X <- train_X[, names(train_X) != "y", drop = FALSE]
  train_X <- as.matrix(train_X)
  
  expected_val <- var <- vector()
  selCourse_all <- selected(mod_mboost)
  for(iter in 1:M_m){
    selCourse <- selCourse_all[1:iter]
    selected_variables <- colnames(train_X)[selCourse]
    sel <- sort(unique(selCourse))
    
    # Initialize objects for calculating v^T
    p <- ncol(train_X)
    n <- nrow(train_X)
    B <- R_t <- u <- vector("list", length = iter)
    H <- X <- vector("list", length = iter + 1)
    t_ind <- vector()
    
    # Calculate H0
    H[[1]] <- matrix(0L, nrow = n, ncol = n)
    u[[1]] <- diag(n)
    
    # Calculate Xt, Ht, and Rt
    for (t in 1:iter) {
      selected_variables_t <- selected_variables[t]
      t_ind[t] <- which(colnames(train) %in% selected_variables_t)
      X[[t]] <- train_X[, selected_variables_t]
      X[[t]] <- as.matrix(X[[t]])
      B[[t]] <- solve((t(X[[t]]) %*% X[[t]])) %*% t(X[[t]])
      H[[t + 1]] <- X[[t]] %*% solve(t(X[[t]]) %*% X[[t]]) %*% t(X[[t]])
      I_H <- lapply(1:t, function(x)
        diag(n) - learning_rate * H[[x]])
      R_t[[t]] <-
        learning_rate * as.matrix(make_basis(t_ind[t], p)) %*% B[[t]] %*% Reduce("%*%", rev(I_H))
      u[[t]] <- Reduce("%*%", rev(I_H))
    }
    
    # Calculate R
    R <- Reduce("+", R_t)
    
    # Calculate vT (note that the coefficient estimates beta_hat = vT %*% y)
    vT <- lapply(seq_len(nrow(R)), function(i) R[i,])
    
    signCourse <-
      sapply(1:iter, function(m)
        sapply(mod_mboost[m]$coef(), "[[", 1))
    nams <- attr(signCourse[[length(signCourse)]], "names")
    
    # Calculate signs
    signCourseS <- do.call("rbind", lapply(signCourse, function(sc) {
      lenSc <- length(sc)
      if (lenSc < length(nams)) {
        namSc <- names(sc)
        namsN <- nams[!nams %in% namSc]
        sc <- c(rep(0, length(namsN)), sc)
        names(sc) <- c(namsN, namSc)
        sc <- sc[nams]
      }
      unlist(sc)
    }))
    signCoursePM <-
      apply(rbind(rep(0, ncol(signCourseS)), signCourseS), 2, diff)
    signs <- if (length(signCoursePM) == 1){
      sign(signCoursePM)
    }else{
      rowSums(sign(signCoursePM))
    }
    
    
    Gamma <- unlist(lapply(1:length(selCourse), function(i) {
      k = selCourse[i]
      lapply(c(1:p)[-k], function(j) {
        x <- rbind((signs[i] * olsFun(train_X[, k]) + olsFun(train_X[, j])) %*% u[[i]],
                   (signs[i] * olsFun(train_X[, k]) - olsFun(train_X[, j])) %*% u[[i]])
        rownames(x) <-
          c(paste(i, k, j, "+", sep = "_"), paste(i, k, j, "-", sep = "_"))
        return(x)
      })
    }), recursive = F)
    
    # Gamma has 2 * (p - 1) * m_stop rows and N columns
    # Each row contains i_k_j_sign, where i = iteration, k = index of selected variable, and j = index of another variable neq k
    # and sign is the sign of the coefficient of the selected variable
    Gamma <- do.call("rbind", Gamma)
    
    #############################################
    #  Calculate conditional mean and variance  #
    #############################################
    
    # The linear contraints are rho %*% b <= rhs
    # rho - Gamma %*% Sigma %*% V(V^T %*% Sigma V)^{-1}
    # b - beta hat coefficient estimates V^TY
    # rhs - - Gamma %*% Z, where Z = (I - Sigma %*% V(V^T %*% Sigma V)^{-1}V^T) %*% Y
    v_mat <- data.frame(do.call(rbind, vT))
    v_mat <- v_mat[apply(v_mat, 1, function(x) !all(x == 0)), ]
    upper <- lower <- vector()
    # loop over coefficients
    for (coef in 1:nrow(v_mat)) {
      C <-  covY %*% t(v_mat[coef,]) %*% solve(as.matrix(v_mat[coef,]) %*% covY %*% t(as.matrix(v_mat[coef,])))
      Z_star <- (diag(n) - as.matrix(C) %*% as.matrix(v_mat[coef, ])) %*% as.matrix(y)
      Gamma_C <- as.numeric(Gamma %*% C)
      neg_Gamma_Z_star <- -Gamma %*% Z_star
      plus_index <- which(Gamma_C > 0)
      neg_index <- which(Gamma_C < 0)
      if (length(plus_index) == 0) {
        lower[coef] = -Inf
      } else{
        lower[coef] <- max(neg_Gamma_Z_star[plus_index] / Gamma_C[plus_index])
      }
      if (length(neg_index) == 0) {
        upper[coef] = Inf
      } else{
        upper[coef] <- min(neg_Gamma_Z_star[neg_index] / Gamma_C[neg_index])
      }
    }
    
    if(n == length(f_train[[1]])){
      mu <- f_train[[study_k]]
    }else if(n == length(Reduce("c", f_train))){
      mu <- Reduce("c", f_train)
    }
    v_jT <- vT[[ind]]
    mu_bar_j <- as.numeric(v_jT %*% mu)
    vartheta_j <- as.numeric(t(v_jT) %*% covY %*% v_jT)
    sqrt_vartheta_j <- sqrt(vartheta_j)
    lower <- "[<-"(numeric(p), sel, lower)
    upper <- "[<-"(numeric(p), sel, upper)
    alpha_j <- as.numeric((lower[ind] - mu_bar_j)/sqrt_vartheta_j)
    xi_j <- as.numeric((upper[ind] - mu_bar_j)/sqrt_vartheta_j)
    expected_val[iter] <- mu_bar_j - sqrt_vartheta_j * (dnorm(xi_j)- dnorm(alpha_j))/(pnorm(xi_j) - pnorm(alpha_j))
    var[iter] <- vartheta_j * (1 - (xi_j * dnorm(xi_j) - alpha_j * dnorm(alpha_j))/(pnorm(xi_j) - pnorm(alpha_j)) - ((dnorm(xi_j)- dnorm(alpha_j))/(pnorm(xi_j) - pnorm(alpha_j)))^2)
  }
  return(list(expected_val = expected_val, var = var))
}


sim_prop2 <- function(edat_train, edat_test, f_train, f_test, Z_train, Z_test, sigma_re, sigma_eps, wk, learning_rate, M_upp, ind, cols_re, true_coefs){
  
  nvar = ncol(edat_train[[1]])
  
  # Generate the random effects and outcomes for the training data
  edat_sim = edat_train
  for (k in 1:length(edat_train)) {
    dataset = edat_train[[k]]
    f_k = as.matrix(f_train[[k]])
    Z_k = as.matrix(Z_train[[k]])
    gamma = rnorm(ncol(Z_k), 0, sigma_re)
    eps = rnorm(nrow(dataset), 0, sigma_eps)
    dataset$y = f_k + Z_k %*% gamma + eps
    dataset$y = scale(dataset$y, center = TRUE, scale = FALSE)
    edat_sim[[k]] = dataset
  }
  
  # Generate random effects and outcomes for test data
  edat_sim_test = edat_test
  for (k in 1:length(edat_test)) {
    dataset = edat_test[[k]]
    f_0 = as.matrix(f_test[[k]])
    Z_0 = as.matrix(Z_test[[k]])
    gamma2 = rnorm(ncol(Z_k), 0, sigma_re)
    eps2 = rnorm(nrow(dataset), 0, sigma_eps)
    dataset$y = f_0 + Z_0 %*% gamma2 + eps2
    edat_sim_test[[k]] = dataset
  }
  
  train = as.data.frame(rbindlist(edat_sim))
  
  if(class(edat_sim_test) == "data.frame"){
    edat_sim_test <- list(edat_sim_test)
  }
  
  if(class(edat_sim_test) == "list"){
    test <- as.data.frame(rbindlist(edat_sim_test))
  }else test <- edat_sim_test
  
  test_X <- data.frame(test)
  test_X <- test_X[, names(test_X) != "y", drop = FALSE]
  test_X <- as.matrix(test_X)
  
  all_data <- c(edat_sim, edat_sim_test)
  ndat_total <- length(all_data)
  
  ######################################################
  #                  Algorithm 2 (OLS)                 #
  ######################################################
  fmla <- as.formula(paste0('y ~ ', paste0('bols(', setdiff(names(train), 
                                                            'y'), ', intercept = FALSE)', collapse= " + ")))
  M_merge <- mstop(aic <- AIC(glmboost(y ~., data = train, offset = 0, control = boost_control(mstop = M_upp, nu = learning_rate)), "corrected"))
  prop2_merge <- proposition2(mod_mboost = mboost(fmla, data = train, offset = 0, control= boost_control(mstop = M_upp, nu = learning_rate)), edat_sim = train, ind = ind, sigma_re = sigma_re, f_train = f_train, cols_re = cols_re)
  
  #############################################
  #                                           #
  #                     Ens                   #
  #                                           #
  #############################################
  
  #################################################
  #                 Algorithm 2                   #
  #################################################
  
  obj_ens <- coefficients_ens <- prop2_ens <- vector("list", length(edat_sim))
  M_ens <- vector()
  for(k in 1:length(edat_sim)){
    # Boosting with OLS (Algorithm 2)
    fmla <- as.formula(paste0('y ~ ', paste0('bols(', setdiff(names(edat_sim[[k]]),
                                                              'y'), ', intercept = FALSE)', collapse= " + ")))
    edat_sim[[k]] <- do.call(data.frame, edat_sim[[k]])
    M_ens[k] <- mstop(aic <- AIC(glmboost(y ~., data = edat_sim[[k]], offset = 0, control = boost_control(mstop = M_upp, nu = learning_rate)), "corrected"))
    obj_ens[[k]] <- mboost(fmla, data = edat_sim[[k]], offset = 0, control= boost_control(mstop = M_upp, nu = learning_rate))
    prop2_ens[[k]] <- proposition2(mod_mboost = obj_ens[[k]], edat_sim = edat_sim[[k]], ind = ind, f_train = f_train, sigma_re = sigma_re, cols_re = cols_re, study_k = k)
  }

  # Calculate the MSE for merged and ensemble estimators
  merge_bias_sq <- (prop2_merge$expected_val - true_coefs[ind])^2
  merge_variance <- prop2_merge$var
  merge_mse <- merge_bias_sq + merge_variance
  
  prop2_ens_bias <- lapply(1:length(f_train), function(x){
    prop2_ens[[x]]$expected_val
  })
  prop2_ens_var <- lapply(1:length(f_train), function(x){
    prop2_ens[[x]]$var
  })
  ens_bias_sq <- (colWeightedMeans(do.call("rbind", prop2_ens_bias), wk) - true_coefs[ind])^2
  ens_variance <- colWeightedMeans(do.call("rbind", prop2_ens_var), wk^2)
  ens_mse <- ens_bias_sq + ens_variance
  return(list(merge_mse = merge_mse, merge_bias_sq = merge_bias_sq, merge_variance = merge_variance, 
              ens_mse = ens_mse, ens_bias_sq = ens_bias_sq, ens_variance = ens_variance, M_merge = M_merge, M_ens = mean(M_ens)))
}


sim_prop2_multiple <- function(nreps, edat_train, edat_test, f_train, f_test, Z_train, Z_test, sigma_re, sigma_eps, wk, learning_rate, M_upp, ind, cols_re, true_coefs){
  registerDoMC(cores = 48)
  results = foreach (j = 1:nreps, .combine = rbind) %dopar% {
    print(paste("Iteration =", j))
    sim_prop2(edat_train = edat_train, edat_test = edat_test, f_train = f_train, f_test = f_test, Z_train = Z_train, Z_test = Z_test, 
             sigma_re = sigma_re, sigma_eps = sigma_eps, wk = wk, learning_rate = learning_rate, M_upp = M_upp, ind = ind, cols_re = cols_re, true_coefs = true_coefs)
  }
}


# Calculate performance ratio asymptote
#
# Input:
# edat_train: list of training studies
# edat_test: list of test studies
# f_train: list of mean functions for training data
# f_test: list of mean functions for test data
# Z_train: list of random predictor for training data
# sigma_eps: residual error variance
# wk: study-specific weights
# lambda_opt: lambda for merged data
# lambdak_opt: vector of lambdas for study-specific data
# learning_rate: learning rate eta
# M_merge_linear: stopping iteration for merged model
# M_ens_linear: vector of stopping iterations for study-specific models
# cols_re_list: list of column indices that correspond to predictors with random effects
#
# Output:
# asymptote: asymptote from corollary 1
#
cor1 <- function(edat_train, edat_test, f_train, f_test, Z_train, sigma_eps, wk, lambda_opt, lambdak_opt, learning_rate, M_merge_linear, M_ens_linear, cols_re_list){
  
  # K = ndat
  ndat = length(edat_train)
  # P = nvar
  nvar = P = ncol(edat_train[[1]])
  
  # tilde(X_0), design matrix for test data
  X_0 = as.matrix(rbindlist(edat_test))
  
  f_0 = Reduce(c, f_test)
  f = Reduce(c, f_train)
  
  # tilde(X)
  X = as.matrix(rbindlist(edat_train))
  # N
  N <- nrow(X)
  
  # tilde(R)
  R <- vector("list", length = M_merge_linear)
  for(m in 1:M_merge_linear){
    R[[m]] <- ((diag(N) - learning_rate * X %*% solve(t(X) %*% X + lambda_opt * diag(P)) %*% t(X)) %^% (m - 1))
  }
  B <- solve(t(X) %*% X + lambda_opt * diag(P)) %*% t(X)
  R <- learning_rate * as.matrix(B) %*% Reduce("+", R[1:length(R)])
  
  
  # tilde(R)_k
  n_k <- vector()
  X_k <- Z_k <- R_k <- bias_k.list <- f_k <- vector("list", length = ndat)
  
  for(k in 1:ndat){
    # X_k is the design matrix for the kth data set
    X_k[[k]] = as.matrix(edat_train[[k]])
    
    # f_k is the mean function for the kth data set
    f_k[[k]] = as.matrix(f_train[[k]])
    
    # Z_k is the subset of covariates that corresponds to the random effects
    Z_k[[k]] = as.matrix(Z_train[[k]])
    
    p = ncol(X_k[[k]])
    
    n_k[k] = nrow(X_k[[k]])
    
    R_k[[k]] = vector("list", length = M_ens_linear[k])
    # Calculate R
    for(m in 1:M_ens_linear[k]){
      R_k[[k]][[m]] = ((diag(n_k[k]) - learning_rate * X_k[[k]] %*% solve(t(X_k[[k]]) %*% X_k[[k]] + lambdak_opt[k] * diag(p)) %*% t(X_k[[k]])) %^% (m - 1))
    }
    B_k = solve(t(X_k[[k]]) %*% X_k[[k]] + lambdak_opt[k] * diag(p)) %*% t(X_k[[k]])
    R_k[[k]] = learning_rate * B_k %*% Reduce("+", R_k[[k]][1:length(R_k[[k]])])
    
    bias_k.list[[k]] = wk[k] * (X_0 %*% R_k[[k]] %*% f_k[[k]] - f_0)
  }
  
  # Bias terms
  b_ens = Reduce("+", bias_k.list)
  b_merge = X_0 %*% R %*% f - f_0
  
  # Z' 
  Z_prime <- bdiag(Z_k)
  
  # Calculate the transition interval
  denomk_tr <- vector()
  num1k <- denom2k <- denomk <- vector("list", ndat)
  num2 = tr(t(R) %*% t(X_0) %*% X_0 %*% R)
  denom1 = t(Z_prime) %*% t(R) %*% t(X_0) %*% X_0 %*% R %*% Z_prime
  denom1_tr <- tr(denom1)
  for (k in 1:ndat) {
    # wk^2 * tr(R_k^T X_0^T X_0 R_k)
    num1k[[k]] = wk[k]^2 * tr(t(R_k[[k]]) %*% t(X_0) %*% X_0 %*% R_k[[k]])
    
    # wk^2 * Z_k^T R_k^T X_0^T X_0 R_k Z_k
    denom2k[[k]] = wk[k]^2 * t(Z_k[[k]]) %*% t(R_k[[k]]) %*% t(X_0) %*% X_0 %*% R_k[[k]] %*% Z_k[[k]]
    
    # wk^2 * tr(Z_k^T R_k^T X_0^T X_0 R_k Z_k) 
    denomk_tr[k] = wk[k]^2 * tr(t(Z_k[[k]]) %*% t(R_k[[k]]) %*% t(X_0) %*% X_0 %*% R_k[[k]] %*% Z_k[[k]])
  }
  
  num1 = Reduce("+", num1k)
  squared_bias_ens = t(b_ens) %*% b_ens
  squared_bias_m = t(b_merge) %*% b_merge
  
  

  # tr(wk^2 * Z_k^T R_k^T X_0^T X_0 R_k Z_k) 
  denom2_tr = sum(denomk_tr)
  denom = denom1_tr - denom2_tr
    
  return(denom2_tr/denom1_tr)
}

# Simulation
# 
# Input:
# edat_train: list of training data
# edat_test: list of test data
# f_train: list of mean function for fixed effects (training data)
# f_test: list of mean function for fixed effects (test data)
# Z_train: list of covariates with random effects (training data)
# Z_test: list of covariates with random effects (test data)
# sigma_re: variance for random effects
# wk: study-specific weights
# lambda_opt: optimal lambda for merged model
# lambdak_opt: optimal lambads for study-specific models
# learning_rate: learning rate
# M_merge_linear: stopping iteration for Alg 1 (Merged)
# M_merge_cw: stopping iteration for Alg 2 (Merged)
#
sim_each <- function(edat_train, edat_test, f_train, f_test, Z_train, Z_test, sigma_re, sigma_eps, wk, lambda_opt, lambdak_opt, learning_rate, 
                     M_merge_linear, M_merge_cw, M_ens_linear, M_ens_cw, M_merge_cw_bsplines, M_ens_cw_bsplines, M_merge_tree, M_ens_tree){
  
  nvar = ncol(edat_train[[1]])
  
  # Generate the random effects and outcomes for the training data
  edat_sim = edat_train
  for (k in 1:length(edat_train)) {
    dataset = edat_train[[k]]
    f_k = as.matrix(f_train[[k]])
    Z_k = as.matrix(Z_train[[k]])
    gamma = rnorm(ncol(Z_k), 0, sigma_re)
    eps = rnorm(nrow(dataset), 0, sigma_eps)
    dataset$y = f_k + Z_k %*% gamma + eps
    dataset$y = scale(dataset$y, center = TRUE, scale = FALSE)
    edat_sim[[k]] = dataset
  }
  
  # Generate random effects and outcomes for test data
  edat_sim_test = edat_test
  for (k in 1:length(edat_test)) {
    dataset = edat_test[[k]]
    f_0 = as.matrix(f_test[[k]])
    Z_0 = as.matrix(Z_test[[k]])
    gamma2 = rnorm(ncol(Z_k), 0, sigma_re)
    eps2 = rnorm(nrow(dataset), 0, sigma_eps)
    dataset$y = f_0 + Z_0 %*% gamma2 + eps2
    edat_sim_test[[k]] = dataset
  }
  
  train = as.data.frame(rbindlist(edat_sim))
  
  if(class(edat_sim_test) == "data.frame"){
    edat_sim_test <- list(edat_sim_test)
  }
  
  if(class(edat_sim_test) == "list"){
    test <- as.data.frame(rbindlist(edat_sim_test))
  }else test <- edat_sim_test
  
  test_X <- data.frame(test)
  test_X <- test_X[, names(test_X) != "y", drop = FALSE]
  test_X <- as.matrix(test_X)
  
  all_data <- c(edat_sim, edat_sim_test)
  ndat_total <- length(all_data)
  
  # Formulas for RE models
  # ind.features = 1:nvar
  # lm.formula <- as.formula(paste("y ~ ", paste(c(names(train)[ind.features], "0"), collapse = "+")))
  # reStruct.formula = as.formula(paste(paste("~", paste(c(names(Z_train[[1]]), "0"), collapse = "+")), "|study"))
  
  #############################################
  #                                           #
  #                  Merging                  #
  #                                           #
  #############################################
  
  #################################################
  #                 Algorithm 1                   #
  #################################################
  out <- boost_fit(edat_train = edat_sim, edat_test = test, lambda_opt = lambda_opt, lambdak_opt = lambdak_opt,
                   M_merge_linear = M_merge_linear, M_ens_linear = M_ens_linear, learning_rate = learning_rate)
  R <- out$R
  R_k <- out$R_k
  train_y <- as.matrix(train$y)
  R <- as.matrix(R)
  Rboost_merge <- mean((test$y - test_X %*% R %*% train_y)^2)
  
  
  ######################################################
  #                  Algorithm 2 (OLS)                 #
  ######################################################
  fmla <- as.formula(paste0('y ~ ', paste0('bols(', setdiff(names(train), 
                                                            'y'), ', intercept = FALSE)', collapse= " + ")))
  mod_mboost <- mboost(fmla, data = train, offset = 0, control= boost_control(mstop = M_merge_cw, nu = learning_rate))
  Mboost_merge <- mean((test$y - predict(mod_mboost, newdata = as.data.frame(test_X)))^2)
  
  ###########################################################
  #                  Algorithm 2 (Bsplines)                 #
  ###########################################################
  
  fmla_bbs <- as.formula(paste0('y ~ ', paste0('bbs(', setdiff(names(train), 
                                                               'y'), ', knots = 3, df = 3)', collapse= " + ")))
  mod_gamboost <- gamboost(fmla_bbs, data = train, offset = 0, control= boost_control(mstop = M_merge_cw_bsplines, nu = learning_rate))
  
  Gamboost_merge <- mean((test$y - predict(mod_gamboost, newdata = as.data.frame(test_X)))^2)
  
  # #################################################
  # #                      LME                      #
  # #################################################
  # # equal variances
  # edat_sim_lme <- edat_sim
  # for(i in 1:length(edat_sim_lme)){
  #   edat_sim_lme[[i]]$study <- i
  # }
  # train_lme <- do.call(rbind, edat_sim_lme[1:ndat])
  # fit.lme2 = tryCatch(do.call(lme, list(lm.formula, data = train_lme,
  #                                       random = reStruct(reStruct.formula, pdClass = "pdIdent"))),
  #                     error = function(e) NA)
  # pred.lme2 = tryCatch(predict(fit.lme2, newdata = data.frame(test_X), level = 0), error = function(e) rep(NA, nrow(test)))
  # LME_merge = mean((test$y - pred.lme2)^2)
  # 
  
  ###########################################################
  #                  Boosting with trees                    #
  ###########################################################
  Tboost_merge_mod <- grad_boost(data = train, learning_rate = learning_rate, M = M_merge_tree, grad.fun = grad.fun, loss.fun = loss.fun, max_depth = max_depth)
  Tboost_merge_pred <- grad_boost_pred(mod = Tboost_merge_mod[[1]], initial = mean(train$y), newdata = data.frame(test_X), learning_rate = learning_rate)
  Tboost_merge <- mean((test$y - Tboost_merge_pred)^2)
  
  
  #############################################
  #                                           #
  #                     Ens                   #
  #                                           #
  #############################################
  
  #################################################
  #                 Algorithm 2                   #
  #################################################
  
  obj_ens <- coefficients_ens <- Rboost_pred <- Tboost_pred <- Tboost_mod <- Gamboost_mod <- Gamboost_pred <- vector("list", length(edat_sim))
  
  for(k in 1:length(edat_sim)){
    # Boosting with OLS (Algorithm 2)
    fmla <- as.formula(paste0('y ~ ', paste0('bols(', setdiff(names(edat_sim[[k]]),
                                                              'y'), ', intercept = FALSE)', collapse= " + ")))
    edat_sim[[k]] <- do.call(data.frame, edat_sim[[k]])
    obj_ens[[k]] <- mboost(fmla, data = edat_sim[[k]], offset = 0, control= boost_control(mstop = M_ens_cw[k], nu = learning_rate))
    sel <- sort(unique(selected(obj_ens[[k]])))
    coefficients_ens[[k]] <- as.numeric(unlist(coef(obj_ens[[k]])))
    coefficients_ens[[k]] <- "[<-"(numeric(nvar), sel, coefficients_ens[[k]])
    
    # Boosting with linear learners (Algorithm 1)
    Rboost_pred[[k]] <- t(test_X %*% R_k[[k]] %*% edat_sim[[k]]$y)
    
    # Boosting with Bsplines (Algorithm 2)
    Gamboost_mod[[k]] <- gamboost(fmla_bbs, data = edat_sim[[k]], offset = 0, control= boost_control(mstop = M_ens_cw_bsplines[k], nu = learning_rate))
    Gamboost_pred[[k]] <- predict(Gamboost_mod[[k]], newdata = as.data.frame(test_X))[, 1]
    
    # Boosting with trees
    Tboost_mod[[k]] <- grad_boost(data = edat_sim[[k]], learning_rate = learning_rate, M = M_ens_tree[k], grad.fun = grad.fun, loss.fun = loss.fun, max_depth = max_depth)
    Tboost_pred[[k]] <- grad_boost_pred(mod = Tboost_mod[[k]][[1]], initial = mean(edat_sim[[k]]$y), newdata = data.frame(test_X), learning_rate = learning_rate)
  }
  
  #################################################
  #                    Algorithm 1                #
  #################################################
  Rboost_ens <- mean((test$y - colWeightedMeans(do.call(rbind, Rboost_pred), w = wk))^2)
  
  ######################################################
  #                    Algorithm 2  (OLS)              #
  ######################################################
  coefficients_ens <- colWeightedMeans(do.call(rbind, coefficients_ens), w = wk)
  Mboost_ens <- mean((test$y - test_X %*% coefficients_ens)^2)
  
  ######################################################
  #                    Algorithm 2  (OLS)              #
  ######################################################
  Gamboost_ens <- mean((test$y - colWeightedMeans(do.call("rbind", Gamboost_pred), w = wk))^2)
  
  # ######################################################
  # #                Boosting with Trees                 #
  # ######################################################
  Tboost_ens <- mean((test$y - colWeightedMeans(do.call(rbind, Tboost_pred), w = wk))^2)
  
  return(c(Mboost_merge = Mboost_merge, Mboost_ens = Mboost_ens,
           Rboost_merge = Rboost_merge, Rboost_ens = Rboost_ens,
           Tboost_merge = Tboost_merge, Tboost_ens = Tboost_ens,
           Gamboost_merge = Gamboost_merge, Gamboost_ens = Gamboost_ens)) 
}


sim_multiple <- function(nreps, edat_train, edat_test, f_train, f_test, Z_train, Z_test, sigma_re, sigma_eps, wk, lambda_opt, lambdak_opt, learning_rate, 
                         M_merge_linear, M_merge_cw, M_ens_linear, M_ens_cw, M_merge_cw_bsplines, M_ens_cw_bsplines, M_merge_tree, M_ens_tree){
  registerDoMC(cores = 48)
  results = foreach (j = 1:nreps, .combine = rbind) %dopar% {
    print(paste("Iteration =", j))
    sim_each(edat_train = edat_train, edat_test = edat_test, f_train = f_train, f_test = f_test, Z_train = Z_train, Z_test = Z_test, 
             sigma_re = sigma_re, sigma_eps = sigma_eps, wk = wk, lambda_opt = lambda_opt, lambdak_opt = lambdak_opt, learning_rate = learning_rate, 
             M_merge_linear = M_merge_linear, M_merge_cw = M_merge_cw, M_ens_linear = M_ens_linear, M_ens_cw = M_ens_cw, 
             M_merge_cw_bsplines = M_merge_cw_bsplines, M_ens_cw_bsplines = M_ens_cw_bsplines, M_merge_tree = M_merge_tree, M_ens_tree = M_ens_tree)
  }
}

sim_each_highdim <- function(edat_train, edat_test, f_train, f_test, Z_train, Z_test, sigma_re, sigma_eps, wk, learning_rate, 
                             M_merge_cw, M_ens_linear, M_ens_cw, M_merge_cw_bsplines, M_ens_cw_bsplines, M_merge_tree, M_ens_tree){
  
  nvar = ncol(edat_train[[1]])
  
  # Generate the random effects and outcomes for the training data
  edat_sim = edat_train
  for (k in 1:length(edat_train)) {
    dataset = edat_train[[k]]
    f_k = as.matrix(f_train[[k]])
    Z_k = as.matrix(Z_train[[k]])
    gamma = rnorm(ncol(Z_k), 0, sigma_re)
    eps = rnorm(nrow(dataset), 0, sigma_eps)
    dataset$y = f_k + Z_k %*% gamma + eps
    dataset$y = scale(dataset$y, center = TRUE, scale = FALSE)
    edat_sim[[k]] = dataset
  }
  
  # Generate random effects and outcomes for test data
  edat_sim_test = edat_test
  for (k in 1:length(edat_test)) {
    dataset = edat_test[[k]]
    f_0 = as.matrix(f_test[[k]])
    Z_0 = as.matrix(Z_test[[k]])
    gamma2 = rnorm(ncol(Z_k), 0, sigma_re)
    eps2 = rnorm(nrow(dataset), 0, sigma_eps)
    dataset$y = f_0 + Z_0 %*% gamma2 + eps2
    edat_sim_test[[k]] = dataset
  }
  
  train = as.data.frame(rbindlist(edat_sim))
  
  if(class(edat_sim_test) == "data.frame"){
    edat_sim_test <- list(edat_sim_test)
  }
  
  if(class(edat_sim_test) == "list"){
    test <- as.data.frame(rbindlist(edat_sim_test))
  }else test <- edat_sim_test
  
  test_X <- data.frame(test)
  test_X <- test_X[, names(test_X) != "y", drop = FALSE]
  test_X <- as.matrix(test_X)
  
  all_data <- c(edat_sim, edat_sim_test)
  ndat_total <- length(all_data)
  
  # Formulas for RE models
  # ind.features = 1:nvar
  # lm.formula <- as.formula(paste("y ~ ", paste(c(names(train)[ind.features], "0"), collapse = "+")))
  # reStruct.formula = as.formula(paste(paste("~", paste(c(names(Z_train[[1]]), "0"), collapse = "+")), "|study"))
  
  #############################################
  #                                           #
  #                  Merging                  #
  #                                           #
  #############################################
  
  ######################################################
  #                  Algorithm 2 (OLS)                 #
  ######################################################
  fmla <- as.formula(paste0('y ~ ', paste0('bols(', setdiff(names(train), 
                                                            'y'), ', intercept = FALSE)', collapse= " + ")))
  mod_mboost <- mboost(fmla, data = train, offset = 0, control= boost_control(mstop = M_merge_cw, nu = learning_rate))
  sel <- sort(unique(selected(mod_mboost)))
  coefficients <- as.numeric(unlist(coef(mod_mboost)))
  coefficients <- "[<-"(numeric(nvar), sel, coefficients)
  
  Mboost_merge <- mean((test$y - test_X %*% coefficients)^2)
  
  ###########################################################
  #                  Algorithm 2 (Bsplines)                 #
  ###########################################################
  
  fmla_bbs <- as.formula(paste0('y ~ ', paste0('bbs(', setdiff(names(train), 
                                                               'y'), ', knots = 3, df = 3)', collapse= " + ")))
  mod_gamboost <- gamboost(fmla_bbs, data = train, offset = 0, control= boost_control(mstop = M_merge_cw_bsplines, nu = learning_rate))
  
  Gamboost_merge <- mean((test$y - predict(mod_gamboost, newdata = as.data.frame(test_X)))^2)
  
  # #################################################
  # #                      LME                      #
  # #################################################
  # # equal variances
  # edat_sim_lme <- edat_sim
  # for(i in 1:length(edat_sim_lme)){
  #   edat_sim_lme[[i]]$study <- i
  # }
  # train_lme <- do.call(rbind, edat_sim_lme[1:ndat])
  # fit.lme2 = tryCatch(do.call(lme, list(lm.formula, data = train_lme,
  #                                       random = reStruct(reStruct.formula, pdClass = "pdIdent"))),
  #                     error = function(e) NA)
  # pred.lme2 = tryCatch(predict(fit.lme2, newdata = data.frame(test_X), level = 0), error = function(e) rep(NA, nrow(test)))
  # LME_merge = mean((test$y - pred.lme2)^2)
  # 
  
  ###########################################################
  #                  Boosting with trees                    #
  ###########################################################
  Tboost_merge_mod <- grad_boost(data = train, learning_rate = learning_rate, M = M_merge_tree, grad.fun = grad.fun, loss.fun = loss.fun, max_depth = max_depth)
  Tboost_merge_pred <- grad_boost_pred(mod = Tboost_merge_mod[[1]], initial = mean(train$y), newdata = data.frame(test_X), learning_rate = learning_rate)
  Tboost_merge <- mean((test$y - Tboost_merge_pred)^2)
  
  
  #############################################
  #                                           #
  #                     Ens                   #
  #                                           #
  #############################################
  
  #################################################
  #                 Algorithm 2                   #
  #################################################
  
  obj_ens <- coefficients_ens  <- Tboost_pred <- Tboost_mod <- Gamboost_mod <- Gamboost_pred <- vector("list", length(edat_sim))
  
  for(k in 1:length(edat_sim)){
    # Boosting with OLS (Algorithm 2)
    fmla <- as.formula(paste0('y ~ ', paste0('bols(', setdiff(names(edat_sim[[k]]),
                                                              'y'), ', intercept = FALSE)', collapse= " + ")))
    edat_sim[[k]] <- do.call(data.frame, edat_sim[[k]])
    obj_ens[[k]] <- mboost(fmla, data = edat_sim[[k]], offset = 0, control= boost_control(mstop = M_ens_cw[k], nu = learning_rate))
    sel <- sort(unique(selected(obj_ens[[k]])))
    coefficients_ens[[k]] <- as.numeric(unlist(coef(obj_ens[[k]])))
    coefficients_ens[[k]] <- "[<-"(numeric(nvar), sel, coefficients_ens[[k]])
    
    # Boosting with Bsplines (Algorithm 2)
    Gamboost_mod[[k]] <- gamboost(fmla_bbs, data = edat_sim[[k]], offset = 0, control= boost_control(mstop = M_ens_cw_bsplines[k], nu = learning_rate))
    Gamboost_pred[[k]] <- predict(Gamboost_mod[[k]], newdata = as.data.frame(test_X))[, 1]
    
    # Boosting with trees
    Tboost_mod[[k]] <- grad_boost(data = edat_sim[[k]], learning_rate = learning_rate, M = M_ens_tree[k], grad.fun = grad.fun, loss.fun = loss.fun, max_depth = max_depth)
    Tboost_pred[[k]] <- grad_boost_pred(mod = Tboost_mod[[k]][[1]], initial = mean(edat_sim[[k]]$y), newdata = data.frame(test_X), learning_rate = learning_rate)
  }
  
  ######################################################
  #                    Algorithm 2  (OLS)              #
  ######################################################
  coefficients_ens <- colWeightedMeans(do.call(rbind, coefficients_ens), w = wk)
  Mboost_ens <- mean((test$y - test_X %*% coefficients_ens)^2)
  
  ######################################################
  #                    Algorithm 2  (OLS)              #
  ######################################################
  Gamboost_ens <- mean((test$y - colWeightedMeans(do.call("rbind", Gamboost_pred), w = wk))^2)
  
  # ######################################################
  # #                Boosting with Trees                 #
  # ######################################################
  Tboost_ens <- mean((test$y - colWeightedMeans(do.call(rbind, Tboost_pred), w = wk))^2)
  
  return(c(Mboost_merge = Mboost_merge, Mboost_ens = Mboost_ens,
           Tboost_merge = Tboost_merge, Tboost_ens = Tboost_ens,
           Gamboost_merge = Gamboost_merge, Gamboost_ens = Gamboost_ens,
           Mboost_merge_sel = unique(selected(mod_mboost)), Mboost_ens_sel = lapply(1:ndat, function(x) unique(selected(obj_ens[[x]]))),
           Gamboost_merge_sel = unique(selected(mod_gamboost)), Gamboost_ens_sel = lapply(1:ndat, function(x) unique(selected(Gamboost_mod[[x]])))))
}




sim_each_stop_iter <- function(edat_train, edat_test, f_train, f_test, Z_train, Z_test, sigma_re, sigma_eps, wk, learning_rate, M){
  nvar = ncol(edat_train[[1]])
  # Generate the random effects and outcomes for the training data
  edat_sim = edat_train
  for (k in 1:length(edat_train)) {
    dataset = edat_train[[k]]
    f_k = as.matrix(f_train[[k]])
    Z_k = as.matrix(Z_train[[k]])
    gamma = rnorm(ncol(Z_k), 0, sigma_re)
    eps = rnorm(nrow(dataset), 0, sigma_eps)
    dataset$y = f_k + Z_k %*% gamma + eps
    dataset$y = scale(dataset$y, center = TRUE, scale = FALSE)
    edat_sim[[k]] = dataset
  }
  
  # Generate random effects and outcomes for test data
  edat_sim_test = edat_test
  for (k in 1:length(edat_test)) {
    dataset = edat_test[[k]]
    f_0 = as.matrix(f_test[[k]])
    Z_0 = as.matrix(Z_test[[k]])
    gamma2 = rnorm(ncol(Z_k), 0, sigma_re)
    eps2 = rnorm(nrow(dataset), 0, sigma_eps)
    dataset$y = f_0 + Z_0 %*% gamma2 + eps2
    edat_sim_test[[k]] = dataset
  }
  
  train = as.data.frame(rbindlist(edat_sim))
  
  if(class(edat_sim_test) == "data.frame"){
    edat_sim_test <- list(edat_sim_test)
  }
  
  if(class(edat_sim_test) == "list"){
    test <- as.data.frame(rbindlist(edat_sim_test))
  }else test <- edat_sim_test
  
  test_X <- data.frame(test)
  test_X <- test_X[, names(test_X) != "y", drop = FALSE]
  test_X <- as.matrix(test_X)
  
  all_data <- c(edat_sim, edat_sim_test)
  ndat_total <- length(all_data)
  
  #############################################
  #                                           #
  #                  Merging                  #
  #                                           #
  #############################################
  
  ######################################################
  #                  Algorithm 2 (OLS)                 #
  ######################################################
  fmla <- as.formula(paste0('y ~ ', paste0('bols(', setdiff(names(train), 
                                                            'y'), ', intercept = FALSE)', collapse= " + ")))
  aic <- AIC(glmboost(y ~ ., data = train, offset = 0, control = boost_control(mstop = M, nu = learning_rate)), "corrected")
  cv_M_merge <- mstop(aic)
  df_merge <- attr(aic, "df")
  
  #############################################
  #                                           #
  #                     Ens                   #
  #                                           #
  #############################################
  
  #################################################
  #                 Algorithm 2                   #
  #################################################
  
  obj_ens <- coefficients_ens <- df_ens <- vector("list", length(edat_sim))
  cv_M_ens <- vector()
  
  for(k in 1:length(edat_sim)){
    # Boosting with OLS (Algorithm 2)
    fmla <- as.formula(paste0('y ~ ', paste0('bols(', setdiff(names(edat_sim[[k]]),
                                                              'y'), ', intercept = FALSE)', collapse= " + ")))
    edat_sim[[k]] <- do.call(data.frame, edat_sim[[k]])
    aic <- AIC(glmboost(y ~ ., data = edat_sim[[k]], offset = 0, control = boost_control(mstop = M, nu = learning_rate)), "corrected")
    cv_M_ens[k] <- mstop(aic)
    df_ens[[k]] <- attr(aic, "df")
  }
  return(list(cv_M_merge = cv_M_merge, cv_M_ens = mean(cv_M_ens), df_merge = df_merge, df_ens = df_ens)) 
}



sim_multiple_stop_iter <- function(nreps, edat_train, edat_test, f_train, f_test, Z_train, Z_test, sigma_re, sigma_eps, wk, learning_rate, M){
  registerDoMC(cores = 48)
  results = foreach (j = 1:nreps, .combine = rbind) %dopar% {
    print(paste("Iteration =", j))
    sim_each_stop_iter(edat_train = edat_train, edat_test = edat_test, f_train = f_train, f_test = f_test, Z_train = Z_train, Z_test = Z_test, 
             sigma_re = sigma_re, sigma_eps = sigma_eps, wk = wk, learning_rate = learning_rate, M = M)
  }
}


calc_AICc_Alg2 <- function(edat_sim, M_m, mod_mboost, learning_rate){
  # Obtain unique indices of selected variables 
  sel <- sort(unique(selected(mod_mboost)))
  # Obtain response and length of data set
  y <- mod_mboost$response
  n <- length(y) 
  
 
  n_k <- vector()
  # Populate Z_k, covYk, train_X_k, and n_k
  if(class(edat_sim) == "list"){
    num_study = length(edat_sim)
  }else if(class(edat_sim) == "data.frame"){
    num_study = 1
  }
  train_X_k <- vector("list", length = length(edat_sim))
  for(k in 1:num_study){
    if(class(edat_sim) == "list"){
      train_X_k[[k]] <- data.frame(edat_sim[[k]])
    }else if(class(edat_sim) == "data.frame"){
      train_X_k[[k]] <- data.frame(edat_sim)
    }
    train_X_k[[k]] <- train_X_k[[k]][, names(train_X_k[[k]]) != "y", drop = FALSE]
    n_k[k] <- nrow(train_X_k[[k]])
  }
  
  ############################################################
  #      Calculate the boosting operator \mathcal{B}_m       # 
  ############################################################
  
  # train is the (merged) training data
  if(class(edat_sim) == "list"){
    train <- rbindlist(edat_sim)
  }else if(class(edat_sim) == "data.frame"){
    train <- edat_sim
  }
 
  # train_X is the design matrix for the merged data set
  train_X <- data.frame(train)
  train_X <- train_X[, names(train_X) != "y", drop = FALSE]
  train_X <- as.matrix(train_X)
  nvar <- ncol(train_X)
 
  # Initialize mod_mboostects for calculating v^T
  selected_variables <- colnames(train_X)[mod_mboost$xselect()]
  p <- ncol(train_X)
  n <- nrow(train_X)
  B <- R_t <- u <- mathcalBm <- vector("list", length = M_m)
  H <- X <- vector("list", length = M_m + 1)
  t_ind <- vector()
  
  # Calculate H0
  H[[1]] <- matrix(0L, nrow = n, ncol = n)
  u[[1]] <- diag(n)
  
  df <- AICc <- vector()
  mathcalBm <- vector("list", length = M_m)
  # Calculate Xt, Ht, and Rt
  for(t in 1:M_m){
    selected_variables_t <- selected_variables[t]
    t_ind[t] <- which(colnames(train) %in% selected_variables_t)
    X[[t]] <- train_X[, selected_variables_t]
    X[[t]] <- as.matrix(X[[t]])
    B[[t]] <- solve((t(X[[t]]) %*% X[[t]])) %*% t(X[[t]])
    H[[t + 1]] <- X[[t]] %*% solve(t(X[[t]]) %*% X[[t]]) %*% t(X[[t]])
    I_H <- lapply(1:t, function(x) diag(n) - learning_rate * H[[x]])
    mathcalBm[[t]] <- diag(n) - Reduce("%*%", rev(I_H))
    df[t] <- tr(mathcalBm[[t]])
    AICc[t] <- (1 + df[t]/n)/(1 - df[t] + 2)/n + log(mean((y - mathcalBm[[t]] %*% y)^2))
  }
  return(list(df = df, AICc = AICc))
}

### Bootstrap MSPE ratio for two models
#
# Input:
# nboot: number of bootstrap iterations
# results.df: dataframe formatted like the output of sim_multi()
# col1: column of results.df corresponding to first model
# col2: column of results.df corresponding to second model
#
# Output:
# avg.mspe.merged: mean mspe for merged model
# avg.mspe.ens: mean mspe for ensemble model
# avg.mspe.ens/avg.mspe.merged: MSPE ratio comparing ensemble to merged
#
boot_ci <- function(nboot, results.df, col1, col2, seed = 1) {
  set.seed(seed)
  results.df = results.df[which(results.df[,col1] > 0 & results.df[, col2]>0),]
  avg.mspe.merged = rep(NA, nboot)
  avg.mspe.ens = rep(NA, nboot)
  for (i in 1:nboot) {
    boot.sample.ind = sample(1:nrow(results.df), nrow(results.df), replace = T)
    avg.mspe.merged[i] = mean(results.df[boot.sample.ind, col1])
    avg.mspe.ens[i] = mean(results.df[boot.sample.ind, col2])
  }
  return(list(avg.mspe.merged, avg.mspe.ens, avg.mspe.ens/avg.mspe.merged))
}

# Gradient function for regression
#
# Input:
# y - true response
# yhat - fitted response
#
# Output:
# gradient assuming a squared error loss function, i.e. residuals
#
grad.fun <- function(y, yhat){
  return((y - yhat))
}

# Loss function for regression
#
# Input:
# y - true response
# yhat - fitted response
#
# Output:
# scaled MSE
#
loss.fun <- function(y, yhat) return(1/2 * (y - yhat)^2)


# Gradient Boosting with Trees
#
# Input:
# data - data.frame that contains labels in the column "y" and covariates in the other columns
# learning_rate - step size
# M - number of iterations
# grad.fun - gradient function
# loss.fun - loss function
#
# Output:
# mod - list of weak models over M iterations
# store_loss - training error
# store_grad - training gradient
#
grad_boost <- function(data, learning_rate, M, grad.fun, loss.fun, max_depth) {
  
  # Initialize fit with mean of y
  fit <- mean(data$y)
  
  # Calculate negative gradient and loss
  grad <- grad.fun(y = data$y, yhat = fit)
  loss <- loss.fun(y = data$y, yhat = fit)
  mod <- list()
  
  store_grad <- c()
  store_loss <- c()
  
  # Loop over a total of M iterations
  for(i in 1:M){
    store_grad[i] <- sum(grad)
    # Fit base learner (tree) to the gradient
    tmp <- data$y
    data$y <- grad
    base_learner <- rpart::rpart(y ~ ., data = data, control = list(maxdepth = max_depth))
    data$y <- tmp
    
    # Update fitted values
    fit <- fit + learning_rate * as.vector(predict(base_learner, newdata = data[, names(data) != "y", drop = FALSE]))
    
    # Update gradient
    grad <- grad.fun(y = data$y, yhat = fit)
    
    # Update loss
    loss <- loss.fun(y = data$y, yhat = fit)
    
    # Store current model (index is i + 1 because i = 1 contain the initialized estiamtes)
    mod[[i + 1]] <- base_learner
    store_grad[i] = sum(grad)
    store_loss[i] = sum(loss)
  }
  return(list(mod, store_loss[length(store_loss)], store_grad[length(store_grad)]))
}


# Gradient Boosting Prediction
#
# Input:
# mod - list containing gradient boosting models
# newdata - data.frame that contains labels in the column "y" and covariates in the other columns
# learning_rate - vector of step sizes, one per iteration
#
# Output:
# predictions on newdata
#
grad_boost_pred <- function(mod, initial, newdata, learning_rate){
  rep(initial, nrow(newdata)) + apply(t(sapply(2:length(mod), function(x) predict(mod[[x]], newdata = newdata))) * learning_rate, 2, sum)
}


# Cross validation for gradient boost
#
# Input:
# data - data.frame containing the data
# nfolds - number of folds to perform CV on
# learning_rate - step size
# M - total number of max_nround to run
# max_depth - largest possible depth of tree
# grad.fun - gradient function
# loss.fun - loss function
# early_stopping_rounds - number of consecutive values that does not decrease
# initial - baseline initializations
#
# Ouput:
# number of optimal iterations
#
cv_grad_boost <- function(data, nfolds, learning_rate, M, max_depth, grad.fun = grad.fun, loss.fun = loss.fun, early_stopping_rounds){
  
  # Split data into nfolds
  if(class(data) == "data.frame"){
    nr <- nrow(data)
    data <- data[sample(nrow(data)), ]
    data_split <- split(data, rep(1:ceiling(nr/nfolds), each = ceiling(nr/nfolds), length.out = nr))
  }else if(class(data) == "list"){
    data_split <- list()
    for(k in 1:ndat){
      nr <- nrow(data[[k]])
      # Shuffle the rows
      data[[k]] <- data[[k]][sample(nrow(data[[k]])), ]
      data_split[[k]] <- split(data[[k]], rep(1:ceiling(nr/nfolds), each = ceiling(nr/nfolds), length.out = nr))
    }
  }
  
  # Initialize mse and pred_gb 
  # pred_gb[[1]][[2]] stores the predictions on fold 1 at iteration 2
  mse <- vector("list", nfolds)
  pred_gb <- replicate(nfolds, vector("list", M + 1), simplify = FALSE)
  
  # Loop over each fold
  for(i in 1:nfolds){
    # Use the i-th fold as the testing data
    if(class(data) == "data.frame"){
      training <- do.call(rbind, data_split[-i])
      testing <- data.frame(data_split[i])
      names(testing) <- sub('.*\\.', '', names(testing))
    }else if(class(data) == "list"){
      training <- lapply(data_split, function(x) do.call(rbind, x[-i]))
      testing <- do.call(rbind, (lapply(data_split, function(x) do.call(rbind, x[i]))))
      names(testing) <- sub('.*\\.', '', names(testing))
      training <- do.call(rbind, training)
    }
    
    # Fit gradient boosting model
    out_grad_boost <- grad_boost(data = training, learning_rate = learning_rate, M = M, grad.fun = grad.fun, loss.fun = loss.fun,
                                 max_depth = max_depth)
    
    # First iteration's prediction is the mean of the training data
    pred_gb[[i]][[1]] <- rep(mean(training$y), nrow(testing))
    mse[[i]][[1]] <- mean((testing$y - pred_gb[[i]][[1]])^2)
    
    # Loop over each iteration
    for(j in 2:(M + 1)){
      if(ncol(testing) == 2){
        newdata_test <- data.frame(V1 = testing[, -which(names(testing) %in% c("y"))])
      }else newdata_test <- testing[, -which(names(testing) %in% c("y"))]
      pred_gb[[i]][[j]] <- pred_gb[[i]][[j - 1]] + learning_rate * predict(out_grad_boost[[1]][[j]], newdata = newdata_test)
      mse[[i]][[j]] <- mean((testing$y - pred_gb[[i]][[j]])^2)
    }
    # Obtain the 2:(M + 1) entries because the first entry is 0
    # mse[[i]] <- mse[[i]][2:(M + 1)]
  }
  mse <- lapply(1:length(mse), function(x) unlist(mse[[x]]))
  mse.final <- apply(data.frame(do.call("rbind", mse)), 2, mean)
  mse.final <- as.numeric(mse.final)
  
  # cv_nrounds contains the iteration for which the evaluation metric
  # does not improve for early_stopping_rounds 
  cv_nrounds <- best_iter(mse.final, early_stopping_rounds = early_stopping_rounds)
  if(is.null(cv_nrounds)){
    cv_nrounds = M
  }else cv_nrounds = cv_nrounds
  return(cv_nrounds)
}

olsFun <- function(x){
  t(x)/sqrt(as.numeric(crossprod(x)))
}

best_iter <- function(metric, early_stopping_rounds) {
  if (length(metric) == 1) {
    return(1)
  } else if (length(metric) > 1) {
    for (i in 1:(length(metric)  - early_stopping_rounds)) {
      if (all(metric[i] <= metric[(i + 1):(i + early_stopping_rounds)])) {
        return(i)
      }
    }
  }
}
