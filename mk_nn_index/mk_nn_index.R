mk_nn_index <- function(coords, n.neighbors, n.omp.threads = 1){

    n <- nrow(coords)
    indx <- mkNNIndx(coords, n.neighbors, n.omp.threads)
    
    nn.indx <- indx$nnIndx
    nn.indx.lu <- indx$nnIndxLU
    nn.indx.run.time <- indx$run.time
    
    indx <- mkUIndx(n, n.neighbors, nn.indx, nn.indx.lu, 2)
    
    u.indx <- indx$u.indx
    u.indx.lu <- indx$u.indx.lu
    ui.indx <- indx$ui.indx
       
    list(n.neighbors = n.neighbors, n.indx=mk.n.indx.list(nn.indx, n, n.neighbors),
         nn.indx=nn.indx, nn.indx.lu=nn.indx.lu, u.indx=u.indx, u.indx.lu=u.indx.lu, ui.indx=ui.indx)

}
