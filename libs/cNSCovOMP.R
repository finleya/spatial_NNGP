c.ns.cov.omp <- function(coords.1, coords.2, sigma.sq, tau.sq = 0, alpha, gamma, kappa, inv = FALSE, n.omp.threads = 1){
    
    if(!is.matrix(coords.1))
        coords.1 <- as.matrix(coords.1)
    
    if(missing(coords.2))
        coords.2 <- coords.1
    
    if(!is.matrix(coords.2))
        coords.2 <- as.matrix(coords.2)
    
    if(ncol(coords.1) != ncol(coords.2))
        stop("error: ncol(coords.1) != ncol(coords.2)")

    inv <- as.integer(inv)
    
    n1 <- nrow(coords.1)
    n2 <- nrow(coords.2)

    C <- matrix(0, n1, n2)
    
    storage.mode(coords.1) <- "double"
    storage.mode(coords.2) <- "double"
    storage.mode(C) <- "double"
    storage.mode(n1) <- "integer"
    storage.mode(n2) <- "integer"
    storage.mode(sigma.sq) <- "double"
    storage.mode(tau.sq) <- "double"
    storage.mode(alpha) <- "double"
    storage.mode(gamma) <- "double"
    storage.mode(kappa) <- "double"
    storage.mode(inv) <- "integer"
    storage.mode(n.omp.threads) <- "integer"
    
    start.time <- proc.time()
    
    .Call("cNSCovOMP", coords.1, n1, coords.2, n2, C, sigma.sq, tau.sq, alpha, gamma, kappa, inv, n.omp.threads)

    if(inv){
        C[upper.tri(C)] <- t(C)[upper.tri(C)]
    }
    
    list("C" = C, sys.time = proc.time()-start.time)
  }


