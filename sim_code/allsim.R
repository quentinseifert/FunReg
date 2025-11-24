

################## R - Simulation Study ########################################
################################################################################
## File estimates the reported simulation results. Data is generated in       ##
## separate Python script. Needs to be brought into functional shape before   ##  
## plugging it into refund. Done using csv_to_fct                             ##
################################################################################
################################################################################


library(dplyr)
library(refund)
library(mgcv)

csv_to_fct <- function(path, setting) {
  df <- read.csv(path, row.names = 1)
  if (setting == "lin") {
    df <- df %>%
      arrange(id) %>%                    
      group_by(id) %>%
      summarise(
        x = first(x),                    
        y = I(matrix(y, nrow = 1)),
        .groups = "drop"
      )
  } else if (setting=="smoo"){
      df <- df %>%
      arrange(id) %>%                     
      group_by(id) %>%
      summarise(
        x = first(x), 
        x2 = first(x2),
        y = I(matrix(y, nrow = 1)),       
        .groups = "drop"
        )
  }
  else {
    df <- df %>%
      arrange(id) %>%                     
      group_by(id) %>%
      summarise(
        x = first(x), 
        x2 = first(x2),
        y = I(matrix(beta, nrow = 1)),       
        .groups = "drop"
      )
  }
  return(df)
}

t <- seq(0, 1, length.out=100)


results <- list()
results[["lin"]] <- list()
for (n in c("1000")){
  results[["lin"]][[n]] <- list()
    for (i in 0:99){
      df <- csv_to_fct(paste0("sim_data/lin", n, "/", "df", i, ".csv"), setting = "lin")
      start <- Sys.time()
      mod <- pffr(y ~ x , yind = t, data = df)
      end <- Sys.time()
      results[["lin"]][[n]][[i+1]] <- list(rmse=mean((mod$fitted.values - mod$y)**2)**.5, aic=AIC(mod), BIC(mod), logLik(mod), t=end-start)
    }
}

results_lin <- results[["lin"]]
save(results_lin, file="sim_results/R_lin_results.Rda")


lin100 <- round(colMeans(do.call(rbind, lapply(results_lin[["100"]], unlist))), 4)[c(1, 4, 2, 3)]
lin1000 <- round(colMeans(do.call(rbind, lapply(results_lin[["1000"]], unlist))), 4)[c(1, 4, 2, 3)]
lin10000 <- round(colMeans(do.call(rbind, lapply(results_lin[["10000"]], unlist))), 4)[c(1, 4, 2, 3)]




results[["smoo"]] <- list()
for (n in c("100", "1000", "10000")){
  results[["smoo"]][[n]] <- list()
  for (i in 17:99){
    df <- csv_to_fct(paste0("sim_data/smoo", n, "/", "df", i, ".csv"), setting = "smoo")
    start <- Sys.time()
    mod <- pffr(y ~ x + s(x2), yind = t, data = df)
    end <- Sys.time()
    results[["smoo"]][[n]][[i+1]] <- list(rmse=mean((mod$fitted.values - mod$y)**2)**.5, aic=AIC(mod), BIC(mod), logLik(mod), t=end-start)
  }
}




results=list()
results_smoo <- results[["smoo"]]
smoo100 <- round(colMeans(do.call(rbind, lapply(results_smoo[["100"]], unlist))), 4)[c(1, 4, 2, 3)]
smoo1000 <- round(colMeans(do.call(rbind, lapply(results_smoo[["1000"]], unlist))), 4)[c(1, 4, 2, 3)]
smoo10000 <- round(colMeans(do.call(rbind, lapply(results_smoo[["10000"]], unlist))), 4)[c(1, 4, 2, 3)]



save(results_smoo, file="R_smoo_results.Rda")

results[["beta"]] <- list()
for (n in c("1000","10000")){
  results[["beta"]][[n]] <- list()
  for (i in 69:99){
    df <- csv_to_fct(paste0("sim_data/beta", n, "/", "df", i, ".csv"), setting="beta")
    start <- Sys.time()
    df$y <- (df$y * (as.numeric(n)*100 - 1) + 0.5) / (as.numeric(n)*100)
    
    mod <- pffr(y ~ x + s(x2), yind = t, data = df, family="betar")
    end <- Sys.time()
    ll <- sum(dbeta(mod$y, mod$fitted.values * mod$family$getTheta(TRUE), (1-mod$fitted.values) * mod$family$getTheta(TRUE), log=TRUE))
    df <- attributes(logLik(mod))$df
    ll_false <- logLik(mod)
    Aic <- -2 * ll + 2* df
    Bic <- -2 * ll + 2 * log(as.numeric(n) * 100) 
    results[["beta"]][[n]][[i+1]] <- list(rmse=mean((mod$fitted.values - mod$y)**2)**.5, ll=ll, aic=Aic, bic=Bic, t=end-start, ll_false=ll_false)
  }
}
results_beta <- results[["beta"]]




results_beta <- results[["beta"]]

save(results_beta, file="sim_results/R_beta_results.Rda")
load("sim_results/R_beta_results.Rda")

beta100 <- do.call(rbind, lapply(results_beta[["100"]], unlist)) 
beta1000 <- do.call(rbind, lapply(results_beta[["1000"]], unlist)) 
beta10000 <- do.call(rbind, lapply(results_beta[["10000"]], unlist)) 

round(colMeans(beta10000), 4)


data <- read.csv(paste0("sim_data/beta", n, "/", "df", i, ".csv"))
start <- Sys.time()
mod <- gam(y ~ s(t) + s(t, by=x, k=5) + ti(x2, t, k=c(8, 5)), data=data, family="betar")
end <- Sys.time()

re


metrics <- round(colMeans(do.call(rbind, lapply(results[["smoo"]][[n]], unlist))), 4)







alpha <- mod$fitted.values * mod$family$getTheta(TRUE)
beta <- (1-mod$fitted.values) * mod$family$getTheta(TRUE)
y <- mod$y


dfbet <- read.csv("sanity_beta.csv", row.names = 1)
alpha1 <- dfbet$a
beta1 <- dfbet$b
y <- dfbet$y
loglik <- sum(
  (alpha - 1) * log(y) + 
    (beta - 1) * log(1 - y) - 
    (lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta))
)
loglik
logLik(mod)
