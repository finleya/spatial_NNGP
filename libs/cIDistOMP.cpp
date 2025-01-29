#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

  SEXP cIDistOMP(SEXP coords1_r, SEXP n1_r, SEXP coords2_r, SEXP n2_r, SEXP p_r, SEXP D_r, SEXP sigmaSq_r, SEXP phi_r, SEXP nThreads_r){
    
    double *coords1 = REAL(coords1_r);
    int n1 = INTEGER(n1_r)[0];
    
    double *coords2 = REAL(coords2_r);
    int n2 = INTEGER(n2_r)[0];
    
    int p = INTEGER(p_r)[0];

    double *D = REAL(D_r);

    double sigmaSq = REAL(sigmaSq_r)[0];
    double phi = REAL(phi_r)[0];
    
    int nThreads = INTEGER(nThreads_r)[0];

#ifdef _OPENMP
    omp_set_num_threads(nThreads);
#else
    if(nThreads > 1){
      warning("n.omp.threads = %i requested however source code was not compiled with OpenMP support.", nThreads);
      nThreads = 1;
    }
#endif

    int i, j, k;

    double dist = 0.0;

#ifdef _OPENMP
#pragma omp parallel for private(j, dist, k)
#endif
    for(i = 0; i < n1; i++){
      for(j = 0; j < n2; j++){
	dist = 0.0;
	for(k = 0; k < p; k++){
	  dist += pow(coords1[k*n1+i]-coords2[k*n2+j],2);
	}
	D[n1*j+i] = sigmaSq*exp(-phi*sqrt(dist));
      }
    }

    return(R_NilValue);
  }  
}
