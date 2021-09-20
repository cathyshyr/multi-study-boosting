# Load utility functions and tuning parameters
source("Utils.R")
load("tuning_parameters.RData")

M_upp <- 30
# Coef_ind = 5 corresponds to beta6 in the simulations
coef_ind <- 5
cols_re <- 1:5

# Initialize sigma values and error data frame
# sigma.vals <-  seq(0, 0.3, length = 5)
sigma.vals <- c(0.01, 0.05) * ncol(edat_train[[1]])/ncol(Z_train[[1]])
results <- err_merge <- err_ens <- vector("list", length = length(sigma.vals))
set.seed(3)
ind <- 1:length(sigma.vals)
nreps <- 200
for (j in ind) {
  print(j)
  sigmas = rep(sqrt(sigma.vals[j]), ncol(Z_train[[1]]))
  results[[j]] = sim_prop2_multiple(nreps = nreps, 
                              edat_train = edat_train, edat_test = edat_test, f_train = f_train, f_test = f_test, Z_train = Z_train, Z_test = Z_test, 
                              sigma_re = sigmas, sigma_eps = sigma_eps, wk = wk, learning_rate = learning_rate, M_upp = M_upp, ind = coef_ind, cols_re = cols_re, true_coefs = true_coefs)
  err_merge[[j]] = colMeans(do.call("rbind", results[[j]][, "merge_mse"]))
  err_ens[[j]] = colMeans(do.call("rbind", results[[j]][, "ens_mse"]))
}

save(err_merge, err_ens, results, sigma.vals,
     file = "Proposition2_New.RData")