rm(list=ls())
dyn.load("../mk_nn_index/nn.so")
source("../mk_nn_index/nn.R")
source("../mk_nn_index/util.R")
library(fields)
library(viridis)
library(Matrix)
library(coda)
library(GpGp)
library(RhpcBLASctl)

omp_set_num_threads(1)

## Some useful functions.
rmvn <- function(n, mu=0, V = matrix(1)){
  p <- length(mu)
  if(any(is.na(match(dim(V),p))))
    stop("Dimension problem!")
  D <- chol(V)
  t(matrix(rnorm(n*p), ncol=p)%*%D + rep(mu,rep(n,p)))
}

rinvgamma <- function(n, shape, scale){
    1/rgamma(n, shape = shape, rate = scale)
}

logit <- function(theta, a, b){log((theta-a)/(b-theta))}

logit.inv <- function(z, a, b){b-(b-a)/(1+exp(z))}

## Get data.
set.seed(1)

y <- read.table("../sim_data/y")[,1]
x <- as.matrix(read.table("../sim_data/X"))
coords <- as.matrix(read.table("../sim_data/coords"))
w <- read.table("../sim_data/w")[,1]
n <- nrow(coords)
p <- ncol(x)
  
## Sampler.
n.iter <- 2000

beta.samples <- matrix(0, n.iter, p)
w.samples <- matrix(0, n, n.iter)
sigma.sq.samples <- rep(0, n.iter)
phi.samples <- rep(0, n.iter)
tau.sq.samples <- rep(0, n.iter)

## Priors.
sigma.sq.b <- 5
tau.sq.b <- 1
phi.a <- 3/1
phi.b <- 3/0.01

## Tuning.
phi.tuning <- 0.1

## NNGP stuff.
m <- 15

A <- matrix(0, n, n)
D <- rep(0, n)
nn <- mkNNIndx(coords, m)
nn.indx <- mk.n.indx.list(nn$nnIndx, n, m)

## Starting and other stuff.
xx <- t(x)%*%x
beta.s <- coef(lm(y ~ x-1))
w.s <- y - x%*%beta.s
sigma.sq.s <- 1
tau.sq.s <- 1
phi.s <- 3/0.5

batch.iter <- 0
batch.length <- 25
batch.accept <- 0

## Collect samples.
for(s in 1:n.iter){

    ## Update beta.
    V <- chol2inv(chol(xx/tau.sq.s))
    v <- t(x)%*%(y - w.s)/tau.sq.s
    beta.s <- rmvn(1, V%*%v, V)

    ## Update w.
    for(i in 1:(n-1)){
        A[i+1, nn.indx[[i+1]]] <- solve(sigma.sq.s*exp(-phi.s*rdist(coords[nn.indx[[i+1]],,drop=FALSE])),
                                        sigma.sq.s*exp(-phi.s*rdist(coords[nn.indx[[i+1]],,drop=FALSE], coords[i+1,,drop=FALSE])))
        
        D[i+1] <- sigma.sq.s - sigma.sq.s*exp(-phi.s*rdist(coords[i+1,,drop=FALSE], coords[nn.indx[[i+1]],,drop=FALSE])) %*% A[i+1,nn.indx[[i+1]]]
    }

    D[1] <- sigma.sq.s
    
    C.inv <- Matrix(t(diag(1, n) - A)%*%diag(1/D)%*%(diag(1, n) - A), sparse = TRUE) 

    C.log.det <- sum(log(D)) ## Get the log det for updating phi later on.
    
    V <- C.inv + Diagonal(x = 1/tau.sq.s, n = n)
    v <- (y - x%*%beta.s)/tau.sq.s
    chl <- Cholesky(V, perm = TRUE)
    L <- expand1(chl, "L")
    P <- expand1(chl, "P1")
    w.s <- (t(P)%*%solve(t(L), solve(L, P%*%v) + rnorm(n)))[,1]
    
    ## Update phi.
    ## Current.
    current.ltd <- as.numeric(-0.5*C.log.det-0.5*(t(w.s)%*%C.inv%*%w.s) + log(phi.s - phi.a) + log(phi.b - phi.s))

    ## Candidate.
    phi.cand <- logit.inv(rnorm(1, logit(phi.s, phi.a, phi.b), sqrt(phi.tuning)), phi.a, phi.b)

    for(i in 1:(n-1)){
        A[i+1, nn.indx[[i+1]]] <- solve(sigma.sq.s*exp(-phi.cand*rdist(coords[nn.indx[[i+1]],,drop=FALSE])),
                                        sigma.sq.s*exp(-phi.cand*rdist(coords[nn.indx[[i+1]],,drop=FALSE], coords[i+1,,drop=FALSE])))
        
        D[i+1] <- sigma.sq.s - sigma.sq.s*exp(-phi.cand*rdist(coords[i+1,,drop=FALSE], coords[nn.indx[[i+1]],,drop=FALSE])) %*% A[i+1,nn.indx[[i+1]]]
    }

    D[1] <- sigma.sq.s
    
    C.inv.cand <- Matrix(t(diag(1, n) - A)%*%diag(1/D)%*%(diag(1, n) - A), sparse = TRUE) 

    C.log.det.cand <- sum(log(D)) 

    cand.ltd <- as.numeric(-0.5*C.log.det.cand-0.5*(t(w.s)%*%C.inv.cand%*%w.s) + log(phi.cand - phi.a) + log(phi.b - phi.cand))

    if(runif(1,0,1) < exp(cand.ltd-current.ltd)){
        phi.s <- phi.cand
        
        batch.accept <- batch.accept+1
    }

    ## Update sigma.sq.
    sigma.sq.s <- rinvgamma(1, 2 + n/2, sigma.sq.b + as.numeric(t(w.s)%*%(C.inv*sigma.sq.s)%*%w.s)/2)

    ## Update tau.sq.
    tau.sq.s <- rinvgamma(1, 2 + n/2, tau.sq.b + sum((y - x%*%beta.s - w.s)^2)/2)
    
    ## Save samples.
    beta.samples[s,] <- beta.s
    w.samples[,s] <- w.s
    sigma.sq.samples[s] <- sigma.sq.s
    phi.samples[s] <- phi.s
    tau.sq.samples[s] <- tau.sq.s

    ## Progress and reporting.
    batch.iter <- batch.iter + 1
    
    if(batch.iter == batch.length){
        print(paste("Complete:",round(100*s/n.iter)))
        print(paste("Metrop acceptance:", 100*batch.accept/batch.length))
        print("---------")
        batch.iter <- 0
        batch.accept <- 0
    }
    
}


burn.in <- floor(0.5*n.iter)
plot(mcmc(beta.samples), density=FALSE)

w.mu <- apply(w.samples[,burn.in:n.iter], 1, mean)
par(mfrow=c(2,2))
plot(w, w.mu)
plot(mcmc(sigma.sq.samples), density=FALSE)
plot(mcmc(tau.sq.samples), density=FALSE)
plot(mcmc(phi.samples), density=FALSE)


summary(mcmc(beta.samples))$quantiles
summary(mcmc(sigma.sq.samples))$quantiles
summary(mcmc(tau.sq.samples))$quantiles
summary(mcmc(phi.samples))$quantiles
