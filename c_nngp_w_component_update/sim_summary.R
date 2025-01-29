rm(list=ls())
library(coda)
library(tidyverse)
library(MBA)
library(fields)

load("../sim_data/params.rds")
beta.0 <- params["beta.0"]
beta.1 <- params["beta.1"]
phi <- params["phi"]
sigma.sq <- params["sigma.sq"]
tau.sq <- params["tau.sq"]

w <- read.table("../sim_data/w")[,1]
y <- read.table("../sim_data/y")[,1]
p <- 2

## Check MCMC samples.
beta.s <- mcmc(matrix(scan("chain-1-beta"), ncol = p, byrow = FALSE))
colnames(beta.s) <- paste0("beta.",1:p)

summary(beta.s)

w.s <- read_table("chain-1-w", col_names = FALSE)
w.mu <- apply(w.s, 1, median)
plot(w, w.mu)
lines(-10:10, -10:10)

phi.s <- mcmc(matrix(scan("chain-1-phi"), ncol = 1, byrow = FALSE))
summary(phi.s)
plot(phi.s)
phi

sigma.sq.s <- mcmc(matrix(scan("chain-1-sigmaSq"), ncol = 1, byrow = FALSE))
summary(sigma.sq.s)
plot(sigma.sq.s)
sigma.sq

tau.sq.s <- mcmc(matrix(scan("chain-1-tauSq"), ncol = 1, byrow = FALSE))
summary(tau.sq.s)
tau.sq
