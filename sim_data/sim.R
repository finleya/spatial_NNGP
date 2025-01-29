rm(list = ls())

dyn.load("../mk_nn_index/nn.so")
source("../mk_nn_index/nn.R")
source("../mk_nn_index/util.R")
source("../mk_nn_index/mk_nn_index.R")

dyn.load("../libs/cNSCovOMP.so")
source("../libs/cNSCovOMP.R")

library(rgl)
library(dplyr)
library(MBA)
library(fields)
library(viridis)
library(GpGp)

rmvn <- function(n, mu=0, V = matrix(1)){
  p <- length(mu)
  if(any(is.na(match(dim(V),p))))
    stop("Dimension problem!")
  D <- chol(V)
  t(matrix(rnorm(n*p), ncol=p)%*%D + rep(mu,rep(n,p)))
}

## Make some coordinates and order them for NNGP.
set.seed(1)

n <- 1000
coords <- cbind(runif(n,0,1), runif(n,0,1))

ord <- order_maxmin(coords)
coords <- coords[ord,]

## Make the various neighbor vectors.
neighbor.indx <- mk_nn_index(coords, n.neighbors = 15)

## Check out the ordering and neighbors, just for fun.
n.indx <- neighbor.indx$n.indx #see spNNGP manual for description of n.indx.

i <- nrow(coords)
spheres3d(cbind(coords[-i,], 0), col="gray", radius=0.001)
spheres3d(cbind(coords[i,,drop=FALSE], 0), col="blue", radius=0.01)
spheres3d(cbind(coords[n.indx[[i]],,drop=FALSE], 0), col="red", radius=0.01)

## Get the various indexes ready to read into the c code.
nn.indx <- neighbor.indx$nn.indx
nn.indx.lu <- neighbor.indx$nn.indx.lu
u.indx <- neighbor.indx$u.indx
u.indx.lu <- neighbor.indx$u.indx.lu
ui.indx <- neighbor.indx$ui.indx

## Write the various neighbor vectors out for c code.
system("rm nn.indx.* ui.indx.* u.indx.*")

write.table(nn.indx, paste0("nn.indx.",length(nn.indx)), row.names=F, col.names=F)
write.table(nn.indx.lu, paste0("nn.indx.lu.",length(nn.indx.lu)), row.names=F, col.names=F)
write.table(u.indx, paste0("u.indx.",length(u.indx)), row.names=F, col.names=F)
write.table(u.indx.lu, paste0("u.indx.lu.",length(u.indx.lu)), row.names=F, col.names=F)
write.table(ui.indx, paste0("ui.indx.",length(ui.indx)), row.names=F, col.names=F)

## Make some data.
n <- nrow(coords)
n

x <- cbind(1, rnorm(n))

beta <- as.matrix(c(1,5))

sigma.sq <- 5
tau.sq <- 0.1
phi <- 3/0.5

C <- sigma.sq*exp(-phi*as.matrix(dist(coords)))

w <- rmvn(1, rep(0, n), C)

y <- rnorm(n, x%*%beta + w, sqrt(tau.sq))

options(scipen = 100, digits = 4)
write.table(y, "y", row.names=F, col.names=F, sep="\t")
write.table(x, "X", row.names=F, col.names=F, sep="\t")
write.table(w, "w", row.names=F, col.names=F, sep="\t")
write.table(coords, "coords", row.names=F, col.names=F, sep="\t")

params <- c("beta.0" = beta[1], "beta.1" = beta[2],
            "phi" = phi,
            "sigma.sq" = sigma.sq, "tau.sq" = tau.sq)

save(params, file = "params.rds")
