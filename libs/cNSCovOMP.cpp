#ifndef R_NO_REMAP
#  define R_NO_REMAP
#endif

#define USE_FC_LEN_T

#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>
#include <R_ext/Linpack.h>
#include <R_ext/Lapack.h>
#include <R_ext/BLAS.h>
#ifndef FCONE
# define FCONE
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

double dist2(double &a1, double &a2, double &b1, double &b2){
  return(sqrt(pow(a1-b1,2)+pow(a2-b2,2)));
}

extern "C" {

  SEXP cNSCovOMP(SEXP coords1_r, SEXP n1_r, SEXP coords2_r, SEXP n2_r, SEXP C_r, SEXP sigmaSq_r, SEXP tauSq_r, SEXP alpha_r, SEXP gamma_r, SEXP kappa_r, SEXP inv_r, SEXP nThreads_r){

    char const *lower = "L";
    int info;
    int i, j;
    double u, h, zero = 0.0;
    
    double *coords1 = REAL(coords1_r);
    int n1 = INTEGER(n1_r)[0];
    
    double *coords2 = REAL(coords2_r);
    int n2 = INTEGER(n2_r)[0];
    
    double *C = REAL(C_r);

    double sigmaSq = REAL(sigmaSq_r)[0];
    double tauSq = REAL(tauSq_r)[0];
    double alpha = REAL(alpha_r)[0];
    double gamma = REAL(gamma_r)[0];
    double kappa = REAL(kappa_r)[0];

    int inv = INTEGER(inv_r)[0];
    
    int nThreads = INTEGER(nThreads_r)[0];

#ifdef _OPENMP
    omp_set_num_threads(nThreads);
#else
    if(nThreads > 1){
      warning("n.omp.threads = %i requested however source code was not compiled with OpenMP support.", nThreads);
      nThreads = 1;
    }
#endif

#ifdef _OPENMP
#pragma omp parallel for private(i, j, h, u, zero)
#endif
    for(i = 0; i < n1; i++){
      for(j = 0; j < n2; j++){
	h = dist2(coords1[i], coords1[n1+i], coords2[j], coords2[n2+j]);
	u = dist2(coords1[n1*2+i], zero, coords2[n2*2+j], zero);
	C[j*n1+i] = sigmaSq/std::pow(alpha*std::pow(u, 2.0) + 1.0, kappa)*exp(-gamma*h/std::pow(alpha*std::pow(u, 2.0) + 1.0, kappa/2.0));
      }
    }

    if(tauSq != 0){
      for(i = 0; i < n1; i++){
	C[i*n1+i] += tauSq;
      }
    }

    if(inv == 1){
      F77_NAME(dpotrf)(lower, &n1, C, &n1, &info FCONE); if(info != 0){Rf_error("c++ Rf_error: dpotrf failed\n");}
      F77_NAME(dpotri)(lower, &n1, C, &n1, &info FCONE); if(info != 0){Rf_error("c++ Rf_error: dpotrf failed\n");}
    }

    return(R_NilValue);
  }  
}
