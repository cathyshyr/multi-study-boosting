# Load utility functions and tuning parameters
source("Utils.R")
load("tuning_parameters.RData")

# Columns that correspond to random effects (i.e., Z_train[[1]][, cols_re] corresponds to V3-V7, where cols_re = 1:5)
cols_re <- 1:5

# Theorem 2 transition point for equal variances
sigma2_bar <- tau_range(edat_train = edat_train, edat_test = edat_test, f_train = f_train, f_test = f_test, Z_train = Z_train, 
                        sigma_eps = sigma_eps, wk = wk, lambda_opt = lambda_opt, lambdak_opt = lambdak_opt, learning_rate = learning_rate,
                        M_merge_linear = M_merge_linear, M_ens_linear = M_ens_linear, cols_re_list = as.list(cols_re))
sigma2_star_lo <- (ncol(edat_train[[1]])/ncol(Z_train[[1]])) * sigma2_bar[1]
sigma2_star_hi <- (ncol(edat_train[[1]])/ncol(Z_train[[1]])) * sigma2_bar[2]

# Initialize sigma values and error data frame
sigma.vals <-  seq(0, sigma2_star_hi * 3, length = 13)
err <- data.frame(Mboost_merge = NA, Mboost_ens = NA,
                  Rboost_merge = NA, Rboost_ens = NA,
                  Tboost_merge = NA, Tboost_ens = NA,
                  Gamboost_merge = NA, Gamboost_ens = NA)
results <- vector("list", length = length(sigma.vals))

set.seed(2021)
ind <- 1:length(sigma.vals)
nreps <- 500
for (j in ind) {
  print(j)
  sigmas = sqrt(sigma.vals[j] * c(0.5, 0.5, 1, 1, 2))
  results[[j]] = sim_multiple(nreps = nreps, 
                              edat_train = edat_train, edat_test = edat_test, f_train = f_train, f_test = f_test, Z_train = Z_train, Z_test = Z_test, 
                              sigma_re = sigmas, sigma_eps = sigma_eps, wk = wk, lambda_opt = lambda_opt, lambdak_opt = lambdak_opt, learning_rate = learning_rate, 
                              M_merge_linear = M_merge_linear, M_merge_cw = M_merge_cw, M_ens_linear = M_ens_linear, M_ens_cw = M_ens_cw,
                              M_merge_cw_bsplines = M_merge_cw_bsplines, M_ens_cw_bsplines = M_ens_cw_bsplines, M_merge_tree = M_merge_tree, M_ens_tree = M_ens_tree)
  err[j, ] = colMeans(results[[j]])
  print(err[j, ])
}

save(err, results, sigma.vals, sigma2_bar, cols_re,
     file = "Theorem2.RData")
