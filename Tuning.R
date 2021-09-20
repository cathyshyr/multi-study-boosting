# Load packages, utility functions, and data
source("Utils.R")
load("edat_orig.RData")

# Number of studies
ndat <- 4
# Size of each study
n <- 100
# Number of covariates
nvar <- 10
# Residual error sigma_eps
sigma_eps <- 1
# Beta coefficients
set.seed(123)
beta <- rnorm(nvar, 0, 1)
set.seed(123)
ind.small <- sample(2:nvar, nvar/2)
set.seed(123)
beta[ind.small] <- rnorm(length(ind.small), 0, 0.01)
# Spline coefficients
set.seed(123)
alpha <- rnorm(4, 0, 0.5)
# Study-specific weights
wk <- rep(1/ndat, ndat)
# Maximum depth of tree
max_depth <- 2
# Learning rate
learning_rate <- 0.5
# Number of boosting iterations
M <- 50
true_coefs <- c(beta[2:nvar], alpha)

# Remove small studies with < 100 observations
edat_orig <- edat_orig[which(sapply(edat_orig, function(x) nrow(x)) >= 100)]

# Training studies (covariates centered and scaled)
set.seed(1)
edat_train <- init_data(edat_orig[1:5], ndat = 4, nvar)
f_train <- list(mode = "vector", length = length(edat_train))
Z_train <- list(mode = "vector", length = length(edat_train))

for (i in 1:length(edat_train)) {
  dataset = edat_train[[i]]
  dataset = dataset[1:n,]
  spline.design = bs(dataset[, 1], knots = 0)
  colnames(spline.design) = paste0("bs1_", 1:4)
  edat_train[[i]] = cbind(dataset[, 2:nvar], spline.design)
  edat_train[[i]] = data.frame(scale(edat_train[[i]], center = TRUE, scale = TRUE))
  f_train[[i]] = as.matrix(edat_train[[i]][, 1:(nvar - 1)]) %*% beta[2:nvar] + as.matrix(spline.design) %*% alpha
  # Random effects on V3-V7
  Z_train[[i]] = edat_train[[i]][, 2:6]
}

# Test studies (covariates centered and scaled)
set.seed(100)
edat_test <- init_data(edat_orig[6:9], ndat = 4, nvar)
f_test <- list(mode = "vector", length = length(edat_test))
Z_test <- list(mode = "vector", length = length(edat_test))
for (i in 1:length(edat_test)) {
  dataset = edat_test[[i]]
  dataset = dataset[1:n,]
  spline.design = bs(dataset[, 1], knots = 0)
  colnames(spline.design) = paste0("bs1_", 1:4)
  edat_test[[i]] = cbind(dataset[, 2:nvar], spline.design)
  edat_test[[i]] <- data.frame(scale(edat_test[[i]], center = TRUE, scale = TRUE))
  f_test[[i]] = as.matrix(edat_test[[i]][, 1:(nvar - 1)]) %*% beta[2:nvar] + as.matrix(spline.design) %*% alpha
  # Random effects on V3-V7
  Z_test[[i]] = edat_test[[i]][, 2:6]
}


##################################################################
#                   Hyper-parameter tuning                       #
##################################################################
set.seed(1000)
train_ens <- vector("list", length = ndat)
for(i in 1:length(train_ens)){
  train_ens[[i]] <- edat_train[[i]]
  eps <- rnorm(nrow(train_ens[[i]]), 0, sigma_eps)
  train_ens[[i]]$y <- as.matrix(train_ens[[i]][, 1:(nvar - 1)]) %*% beta[2:nvar] + as.matrix(train_ens[[i]][, 10:13])  %*% alpha + eps
  train_ens[[i]]$y <- scale(train_ens[[i]]$y, center = TRUE, scale = FALSE)
}

train_merge <- rbindlist(edat_train)
eps <- rnorm(nrow(train_merge), 0, sigma_eps)
train_merge$y <- as.matrix(train_merge[, 1:(nvar - 1)]) %*% beta[2:nvar] + as.matrix(train_merge[, 10:13])  %*% alpha + eps
train_merge$y <- scale(train_merge$y, center = TRUE, scale = FALSE)
nfolds <- 3
folds_ens <- createFolds(train_ens[[1]]$y, nfolds, list = TRUE, returnTrain = FALSE)
folds_merge <- createFolds(train_merge$y, nfolds, list = TRUE, returnTrain = FALSE)

# Initialize vectors to store cv-error
lambdas <- 2^seq(-2, 2)
error_merge_linear <- rep(0, length(lambdas))
error_ens_linear <- vector("list", ndat)
for(i in 1:length(error_ens_linear)){
  error_ens_linear[[i]] <- rep(0, length(lambdas))
}  

# Tune lambda for Algorithm 1: Boosting with linear learners
for (i in 1:length(lambdas)) {
  print(paste("i =", i))
  lambda = lambdas[i]
  error_merge_linear_j <- rep(0, nfolds)
  error_ens_linear_j <- vector("list", ndat)
  for(l in 1:length(error_ens_linear_j)){
    error_ens_linear_j[[l]] <- rep(0, nfolds)
  }  
  for (j in 1:length(folds_ens)) {
    print(paste("j =", j))
    # Tune lambda on each study
    for(k in 1:ndat){
      dataset = train_ens[[k]][unlist(folds_ens[-j]), ]
      test = train_ens[[k]][folds_ens[[j]], ]
      test_X = test[, names(test) != "y", drop = FALSE]
      ens_fit = boost_fit(edat_train = list(dataset), edat_test = list(test), lambda_opt = lambda, lambdak_opt = lambda,
                          M_merge_linear = M, M_ens_linear = M, learning_rate = learning_rate)
      R_k <- ens_fit$R_k[[1]]
      pred_ens <- as.numeric(as.matrix(test_X) %*% R_k %*% as.matrix(dataset$y))
      error_ens_linear_j[[k]][j] = error_ens_linear_j[[k]][j] + mean((test$y - pred_ens)^2)
    }
    # Tune lambda on merged data
    dataset_m = train_merge[unlist(folds_merge[-j]), ]
    test_m = as.data.frame(train_merge[folds_merge[[j]], ])
    test_m_X = test_m[, names(test_m) != "y", drop = FALSE]
    merge_fit = boost_fit(edat_train = list(dataset_m), edat_test = list(test_m), lambda_opt = lambda, lambdak_opt = lambda,
                          M_merge_linear = M, M_ens_linear = M, learning_rate = learning_rate)
    R <- merge_fit$R
    pred_merge <- as.numeric(as.matrix(test_m_X) %*% R %*% as.matrix(dataset_m$y))
    error_merge_linear_j[j] = error_merge_linear_j[j] + mean((test_m$y - pred_merge)^2)
  }
  # Average across folds
  error_merge_linear[i] = mean(error_merge_linear_j)
  error_ens_linear[[i]] = sapply(1:ndat, function(x) mean(error_ens_linear_j[[x]]))
}

# Optimal lambdas
error_ens_linear_combined <- do.call("rbind", error_ens_linear)
lambda_opt <- lambdas[which(error_merge_linear == min(error_merge_linear))[1]]
lambdak_opt <- sapply(1:ndat, function(x) lambdas[which(error_ens_linear_combined[, x] == min(error_ens_linear_combined[, x]))[1]])


# Tune M using the corrected AIC criterion for Algorithms 1 and 2 (and other base learners)
# Initialize vectors to store cv-error
M_upp <- M

AICc_merge_linear <- MSPE_merge_tboost <- rep(0, M_upp)
AICc_ens_linear  <- MSPE_ens_tboost <- vector("list", ndat)
for(i in 1:length(AICc_ens_linear)){
  AICc_ens_linear[[i]] <- rep(0, M_upp)
  MSPE_ens_tboost[[i]] <- rep(0, M_upp)
}
fmla <- as.formula(paste0('y ~ ', paste0('bols(', setdiff(names(train_merge), 
                                                          'y'), ', intercept = FALSE)', collapse= " + ")))
fmla_bbs <- as.formula(paste0('y ~ ', paste0('bbs(', setdiff(names(train_merge), 
                                                              'y'), ', knots = 3, df = 3)', collapse= " + ")))

for(m in 1:M_upp){
  print(paste("m =", m))
  MSPE_merge_tboost_j <-rep(0, nfolds)
  MSPE_ens_tboost_j <- vector("list", ndat)
  for(i in 1:length(MSPE_ens_tboost_j)){
    MSPE_ens_tboost_j[[i]] <- rep(0, nfolds)
  }
  
  # Algorithm 1
  # Tune M on each study
  for(k in 1:ndat){
    dataset = as.data.frame(scale(train_ens[[k]][sample(1:nrow(train_ens[[k]]), replace = TRUE), ], center = TRUE, scale = TRUE))
    # Algorithm 1
    AICc_ens_linear[[k]][m] = AICc_ens_linear[[k]][m] + calc_AICc_Alg1(edat_sim = dataset, M_m = m, lambda = lambdak_opt[k], learning_rate = learning_rate)
  }
  
  # Tune M on merged data
  dataset_m = as.data.frame(scale(train_merge[sample(1:nrow(train_merge), replace = TRUE)], center = TRUE, scale = TRUE))
  AICc_merge_linear[m] =  AICc_merge_linear[m] + calc_AICc_Alg1(edat_sim = dataset_m, M_m = m, lambda = lambda_opt, learning_rate = learning_rate)
  
  for (j in 1:length(folds_ens)) {
    print(paste("j =", j))
    for(k in 1:ndat){
      dataset = train_ens[[k]][unlist(folds_ens[-j]), ]
      dataset = as.data.frame(scale(dataset, center = TRUE, scale = TRUE))
      test = train_ens[[k]][folds_ens[[j]], ]
      test_X = test[, names(test) != "y", drop = FALSE]
      
      # Trees
      ens_tboost <- grad_boost(data = dataset, learning_rate = learning_rate, M = m, grad.fun = grad.fun, loss.fun = loss.fun, max_depth = max_depth)
      ens_tboost_pred <- grad_boost_pred(mod = ens_tboost[[1]], initial = mean(dataset$y), newdata = data.frame(test_X), learning_rate = learning_rate)
      MSPE_ens_tboost_j[[k]][j] <- mean((test$y - ens_tboost_pred)^2)
    }
    
    # Tune M on merged data
    dataset_m = train_merge[unlist(folds_merge[-j]), ]
    dataset_m = as.data.frame(scale(dataset_m, center = TRUE, scale = TRUE))
    test_m = as.data.frame(train_merge[folds_merge[[j]], ])
    test_m_X = test_m[, names(test_m) != "y", drop = FALSE]
    
    # Trees
    merge_tboost <- grad_boost(data = dataset_m, learning_rate = learning_rate, M = m, grad.fun = grad.fun, loss.fun = loss.fun, max_depth = max_depth)
    merge_tboost_pred <- grad_boost_pred(mod = merge_tboost[[1]], initial = mean(dataset_m$y), newdata = data.frame(test_m_X), learning_rate = learning_rate)
    MSPE_merge_tboost_j[j] <- mean((test_m$y - merge_tboost_pred)^2)
  }
  
  # Average across folds
  MSPE_merge_tboost[m] = mean(MSPE_merge_tboost_j)
  MSPE_ens_tboost[[m]] = sapply(1:ndat, function(x) mean(MSPE_ens_tboost_j[[x]]))
}

# Optimal M
AICc_ens_linear_combined <- do.call("rbind", AICc_ens_linear)
MSPE_ens_tboost_combined <- do.call("rbind", MSPE_ens_tboost)

# Algorithm 2 (OLS)
M_merge_cw <- mstop(aic <- AIC(glmboost(y ~., data = train_merge, offset = 0, control = boost_control(mstop = M_upp, nu = learning_rate)), "corrected"))
M_ens_cw <- vector()
for(k in 1:ndat){
  M_ens_cw[k] <- mstop(aic <- AIC(glmboost(y ~., data = train_ens[[k]], offset = 0, control = boost_control(mstop = M_upp, nu = learning_rate)), "corrected"))
}

# Algorithm 2 (B-splines)
M_merge_cw_bsplines <- mstop(aic <- AIC(gamboost(fmla_bbs, data = as.data.frame(scale(train_merge, center = TRUE, scale = TRUE)),
                                                 offset = 0, control = boost_control(mstop = M_upp, nu = learning_rate)), "corrected"))
M_ens_cw_bsplines <- vector()
for(k in 1:ndat){
  M_ens_cw_bsplines[k] <- mstop(aic <- AIC(gamboost(fmla_bbs, data = as.data.frame(scale(train_ens[[k]], center = TRUE, scale = TRUE)),
                                                      offset = 0, control = boost_control(mstop = M_upp, nu = learning_rate)), "corrected"))
}

# Algorithm 1
M_merge_linear <- c(1:M_upp)[which(AICc_merge_linear == min(AICc_merge_linear))[1]]
M_ens_linear <- sapply(1:ndat, function(x) c(1:M_upp)[which(AICc_ens_linear_combined[, x] == min(AICc_ens_linear_combined[, x]))[1]])

# Tree
M_merge_tree <- c(1:M_upp)[which(MSPE_merge_tboost == min(MSPE_merge_tboost))[1]]
M_ens_tree <- sapply(1:ndat, function(x) c(1:M_upp)[which(MSPE_ens_tboost_combined[, x] == min(MSPE_ens_tboost_combined[, x]))[1]])


save(ndat, n, nvar, sigma_eps, beta, alpha, true_coefs,
     edat_train, edat_test,
     f_train, f_test,
     Z_train, Z_test,
     lambda_opt, lambdak_opt,
     M_merge_linear, M_merge_cw, M_merge_cw_bsplines, M_merge_tree,
     M_ens_linear, M_ens_cw, M_ens_cw_bsplines, M_ens_tree,
     learning_rate = learning_rate, wk = wk, max_depth = max_depth,
     file = "tuning_parameters.RData")
