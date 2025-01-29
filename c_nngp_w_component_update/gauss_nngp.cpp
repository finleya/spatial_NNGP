#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <limits>
#include <omp.h>
using namespace std;

#include "../libs/kvpar.h"

#define MATHLIB_STANDALONE
#include <Rmath.h>
#include <Rinternals.h>

#include <time.h>
#include <sys/time.h>
#define CPUTIME (SuiteSparse_time ( ))

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

double get_cpu_time(){
    return (double)clock() / CLOCKS_PER_SEC;
}

extern "C" {
  extern void dgemm_(const char *transa, const char *transb,
		    const int *m, const int *n, const int *k,
		    const double *alpha, const double *a,
		    const int *lda, const double *b, const int *ldb,
		    const double *beta, double *c, const int *ldc);
   
  extern void  dcopy_(const int *n, const double *dx, const int *incx, double *dy, const int *incy);
  
  extern int dpotrf_(const char *uplo, int *n, double *a, int *lda, int *info);

  extern int dpotri_(const char *uplo, int *n, double *a, int *lda, int *info);

  extern void dsymm_(const char *side, const char *uplo, const int *m,
		     const int *n, const double *alpha,
		     const double *a, const int *lda,
		     const double *b, const int *ldb,
		     const double *beta, double *c, const int *ldc);

  extern double ddot_(int *n, double *x, int *incx, double *y, int *incy);

  extern void dgemv_(const char *trans, const int *m, const int *n, const double *alpha,
		     const double *a, const int *lda, const double *x, const int *incx,
		     const double *beta, double *y, const int *incy);
  
  extern void dsymv_(const char *uplo, const int *n, const double *alpha, const double *a, const int *lda,
		    const double *x, const int *incx, const double *beta, double *y, const int *incy);

  extern void daxpy_(const int *n, const double *alpha, const double *x, const int *incx, double *y, const int *incy);

  extern void dtrmv_(const char *uplo, const char *transa, const char *diag, const int *n,
		     const double *a, const int *lda, double *b, const int *incx);

}

void show(int *a, int n);
void show(double *a, int n);
void show(int *a, int r, int c);
void show(double *a, int r, int c);
void zeros(double *a, int n);
void writeRMatrix(string outfile, double * a, int nrow, int ncol);
void writeRMatrix(string outfile, int * a, int nrow, int ncol);
void mvrnorm(double *des, double *mu, double *cholCov, int dim);
void covTransInv(double *z, double *v, int m);
void covTrans(double *v, double *z, int m);
void covTrans(vector<double> v, double *z, int m);
void covTransInvExpand(double *v, double *z, int m);
void covExpand(double *v, double *z, int m);
double logit(double theta, double a, double b);
double logitInv(double z, double a, double b);
double dist2(double &a1, double &a2, double &b1, double &b2);


double nsCor(double h, double u, double alpha, double gamma, double kappa){
  double r = 1.0/std::pow(alpha*std::pow(u, 2.0) + 1.0, kappa)*exp(-gamma*h/std::pow(alpha*std::pow(u, 2.0) + 1.0, kappa/2.0));
       
  return(r);
}

//Description: update B and F.
void updateBF(double *B, double *F, double *c, double *C, double *coords, int *nnIndx, int *nnIndxLU, int n, int m, double sigmaSq, double phi){
    
  int i, k, l;
  int info = 0;
  int inc = 1;
  double one = 1.0;
  double zero = 0.0;
  char lower = 'L';

  int threadID = 0;
  double h, u;
  int mm = m*m;

#pragma omp parallel for private(k, l, info, threadID, h, u)
    for(i = 0; i < n; i++){
      threadID = omp_get_thread_num();
      if(i > 0){
	for(k = 0; k < nnIndxLU[n+i]; k++){
	  h = dist2(coords[i], coords[n+i], coords[nnIndx[nnIndxLU[i]+k]], coords[n+nnIndx[nnIndxLU[i]+k]]);
	  c[m*threadID+k] = sigmaSq*exp(-phi*h);
	  for(l = 0; l <= k; l++){
	    h = dist2(coords[nnIndx[nnIndxLU[i]+k]], coords[n+nnIndx[nnIndxLU[i]+k]], coords[nnIndx[nnIndxLU[i]+l]], coords[n+nnIndx[nnIndxLU[i]+l]]);
	    C[mm*threadID+l*nnIndxLU[n+i]+k] = sigmaSq*exp(-phi*h);
	  }
	}
	dpotrf_(&lower, &nnIndxLU[n+i], &C[mm*threadID], &nnIndxLU[n+i], &info); if(info != 0){cout << "c++ error: dpotrf failed" << endl;}
	dpotri_(&lower, &nnIndxLU[n+i], &C[mm*threadID], &nnIndxLU[n+i], &info); if(info != 0){cout << "c++ error: dpotri failed" << endl;}
	dsymv_(&lower, &nnIndxLU[n+i], &one, &C[mm*threadID], &nnIndxLU[n+i], &c[m*threadID], &inc, &zero, &B[nnIndxLU[i]], &inc);
	F[i] = sigmaSq - ddot_(&nnIndxLU[n+i], &B[nnIndxLU[i]], &inc, &c[m*threadID], &inc);
      }else{
	B[i] = 0;
	F[i] = sigmaSq;
      }
    }

}

void dimCheck(string test, int i, int j){
  if(i != j)
    cout << test << " " << i << "!=" << j << endl;
}

int main(int argc, char **argv){
  int i, j, k, l, s;
  int info = 0;
  int inc = 1;
  double one = 1.0;
  double negOne = -1.0;
  double zero = 0.0;
  char lower = 'L';
  char upper = 'U';
  char ntran = 'N';
  char ytran = 'T';
  char rside = 'R';
  char lside = 'L';
  
  string parfile;
  if(argc > 1)
    parfile = argv[1];
  else
    parfile = "pfile";
  
  kvpar par(parfile);
  
  bool debug = false;
  
  //Get stuff from the pfile.
  int nThreads; par.getVal("n.threads", nThreads);
  int seed; par.getVal("seed", seed);
  int nSamples; par.getVal("n.samples", nSamples);
  int nSamplesStart; par.getVal("n.samples.start", nSamplesStart);
  int nReport; par.getVal("n.report", nReport);
  string outFile; par.getVal("out.file", outFile);
    
  omp_set_num_threads(nThreads);
  
  //Set seed.
  set_seed(123,seed);
  
  //m number of nearest neighbors.
  //n number of locations.
  //p number of columns of X.
  int m; par.getVal("m", m);
  int n; par.getVal("n", n);
  int p; par.getVal("p", p);
  int pp = p*p;

  //Data and starting values.
  double *X = NULL; par.getFile("X.file", X, i, j);
  double *y = NULL; par.getFile("y.file", y, i, j);
  double *coords = NULL; par.getFile("coords.file", coords, i, j);
  vector<double> betaStarting; par.getVal("beta.starting", betaStarting);
  double *beta = new double[p];
  for(i = 0; i < p; i++){
    beta[i] = betaStarting[i];
  }
  
  double tauSq; par.getVal("tauSq.starting", tauSq);
  double sigmaSq; par.getVal("sigmaSq.starting", sigmaSq);
  double phi; par.getVal("phi.starting", phi);

  //Priors and tuning.
  double tauSq_a = 2.0;
  double tauSq_b; par.getVal("tauSq.b", tauSq_b);
  
  double sigmaSq_a = 2.0;
  double sigmaSq_b; par.getVal("sigmaSq.b", sigmaSq_b);

  double phi_a; par.getVal("phi.a", phi_a);
  double phi_b; par.getVal("phi.b", phi_b);
  double phiTuning; par.getVal("phi.tuning", phiTuning);
  
  int *nnIndx = NULL; par.getFile("nn.indx.file", nnIndx, i, j);
  int *nnIndxLU = NULL; par.getFile("nn.indx.lu.file", nnIndxLU, i, j);
  int *uIndx = NULL; par.getFile("u.indx.file", uIndx, i, j);
  int *uIndxLU = NULL; par.getFile("u.indx.lu.file", uIndxLU, i, j);
  int *uiIndx = NULL; par.getFile("ui.indx.file", uiIndx, i, j);
  int nIndx = static_cast<int>(static_cast<double>(1+m)/2*m+(n-m-1)*m);

  //other stuff
  int mm = m*m;
  double *B = new double[nIndx];
  double *F = new double[n];
  double *BCand = new double[nIndx];
  double *FCand = new double[n];
  double *c = new double[m*nThreads];
  double *C = new double[mm*nThreads];
  
  //return stuff
  int nSamplesKeep = (nSamples-nSamplesStart);
  double *betaSamples = new double[p*nSamplesKeep];
  double *tauSqSamples = new double[nSamplesKeep];
  double *sigmaSqSamples = new double[nSamplesKeep];
  double *phiSamples = new double[nSamplesKeep];
  double *wSamples = new double[n*nSamplesKeep];
  double *fittedSamples = new double[n*nSamplesKeep];
 
  //other stuff
  double logPostCand, logPostCurrent, logDet, status = 0;
  double accept = 0;
  double batchAccept = 0;
  double phiCand;

  //for gibbs update of beta's
  double *XtX = new double[pp];
  double *w = new double[n]; zeros(w, n);

  double *tmp_p = new double[p];
  double *tmp_p2 = new double[p];
  double *tmp_pp = new double[pp];
  double *tmp_n = new double[n];
  double a, v, b, e, mu, var, aij;
  int jj, kk, ll;

  int sKeep = 0;

  dgemm_(&ytran, &ntran, &p, &p, &n, &one, X, &n, X, &n, &zero, XtX, &p);
  
  double wall_start_0 = get_wall_time();

  updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, sigmaSq, phi);
  
  cout << "start sampling" << endl;
  for(s = 0; s < nSamples; s++){
    
    ///////////////
    //update w 
    ///////////////
 
    //for each location
    for(i = 0; i < n; i++){
      a = 0;
      v = 0;    
      if(uIndxLU[n+i] > 0){//is i a neighbor for anybody	
	for(j = 0; j < uIndxLU[n+i]; j++){//how many location have i as a neighbor
	  b = 0;	  
	  //now the neighbors for the jth location who has i as a neighbor
	  jj = uIndx[uIndxLU[i]+j]; //jj is the index of the jth location who has i as a neighbor	  
	  for(k = 0; k < nnIndxLU[n+jj]; k++){// these are the neighbors of the jjth location
	    kk = nnIndx[nnIndxLU[jj]+k];// kk is the index for the jth locations neighbors	    
	    if(kk != i){//if the neighbor of jj is not i
	      b += B[nnIndxLU[jj]+k]*w[kk];//covariance between jj and kk and the random effect of kk
	    }
	  }	  
	  aij = w[jj] - b;	  
	  a += B[nnIndxLU[jj]+uiIndx[uIndxLU[i]+j]]*aij/F[jj];
	  v += pow(B[nnIndxLU[jj]+uiIndx[uIndxLU[i]+j]],2)/F[jj];	  
	}	
      }
      
      e = 0;
      for(j = 0; j < nnIndxLU[n+i]; j++){
	e += B[nnIndxLU[i]+j]*w[nnIndx[nnIndxLU[i]+j]];
      }
      
      mu = (y[i] - ddot_(&p, &X[i], &n, beta, &inc))/tauSq + e/F[i] + a;
      
      var = 1.0/(1.0/tauSq + 1.0/F[i] + v);
      w[i] = rnorm(mu*var, sqrt(var));
    }//end n
    
    ///////////////
    //update beta 
    ///////////////
    for(i = 0; i < n; i++){
      tmp_n[i] = (y[i] - w[i])/tauSq;
    }
    dgemv_(&ytran, &n, &p, &one, X, &n, tmp_n, &inc, &zero, tmp_p, &inc); 	  
    
    for(i = 0; i < pp; i++){
      tmp_pp[i] = XtX[i]/tauSq;
    }
    
    dpotrf_(&lower, &p, tmp_pp, &p, &info); if(info != 0){cout << "c++ error: dpotrf failed" << endl;}
    dpotri_(&lower, &p, tmp_pp, &p, &info); if(info != 0){cout << "c++ error:  dpotrifailed" << endl;}
    dsymv_(&lower, &p, &one, tmp_pp, &p, tmp_p, &inc, &zero, tmp_p2, &inc);
    dpotrf_(&lower, &p, tmp_pp, &p, &info); if(info != 0){cout << "c++ error: dpotrf failed" << endl;}
    mvrnorm(beta, tmp_p2, tmp_pp, p);
    
    /////////////////////
    //update tau^2
    /////////////////////
    for(i = 0; i < n; i++){
      tmp_n[i] = y[i] - w[i] - ddot_(&p, &X[i], &n, beta, &inc);
    }
    
    tauSq = 1.0/rgamma(tauSq_a+n/2.0, 1.0/(tauSq_b+0.5*ddot_(&n, tmp_n, &inc, tmp_n, &inc)));
    
    /////////////////////
    //update sigma^2
    /////////////////////
    a = 0;
    
#pragma omp parallel for private (e, j, b) reduction(+:a, logDet)
    for(i = 0; i < n; i++){
      if(nnIndxLU[n+i] > 0){
	e = 0;
	for(j = 0; j < nnIndxLU[n+i]; j++){
	  e += B[nnIndxLU[i]+j]*w[nnIndx[nnIndxLU[i]+j]];
	}
	b = w[i] - e;
      }else{
	b = w[i];
      }	
      a += b*b/F[i];
    }
    
    sigmaSq = 1.0/rgamma(sigmaSq_a+n/2.0, 1.0/(sigmaSq_b+0.5*a*sigmaSq));

    ///////////////
    //update theta
    ///////////////
    //current
    updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, m, sigmaSq, phi);

    a = 0;
    logDet = 0;
    
#pragma omp parallel for private (e, j, b) reduction(+:a, logDet)
    for(i = 0; i < n; i++){
      if(nnIndxLU[n+i] > 0){
	e = 0;
	for(j = 0; j < nnIndxLU[n+i]; j++){
	  e += B[nnIndxLU[i]+j]*w[nnIndx[nnIndxLU[i]+j]];
	}
	b = w[i] - e;
      }else{
	b = w[i];
      }	
      a += b*b/F[i];
      logDet += log(F[i]);
    }
    
    logPostCurrent = -0.5*logDet - 0.5*a;
    logPostCurrent += log(phi - phi_a) + log(phi_b - phi);
 
    //candidate
    phiCand = logitInv(rnorm(logit(phi, phi_a, phi_b), phiTuning), phi_a, phi_b);
 
    updateBF(BCand, FCand, c, C, coords, nnIndx, nnIndxLU, n, m, sigmaSq, phiCand);
    
    a = 0;
    logDet = 0;
    
#pragma omp parallel for private (e, j, b) reduction(+:a, logDet)
    for(i = 0; i < n; i++){
      if(nnIndxLU[n+i] > 0){
	e = 0;
	for(j = 0; j < nnIndxLU[n+i]; j++){
	  e += BCand[nnIndxLU[i]+j]*w[nnIndx[nnIndxLU[i]+j]];
	}
	b = w[i] - e;
      }else{
	b = w[i];
      }	
      a += b*b/FCand[i];
      logDet += log(FCand[i]);
    }
    
    logPostCand = -0.5*logDet - 0.5*a;
    logPostCand += log(phiCand - phi_a) + log(phi_b - phiCand);
	
    if(runif(0.0,1.0) <= exp(logPostCand - logPostCurrent)){

      phi = phiCand;

      std::swap(BCand, B);
      std::swap(FCand, F);
      
      accept++;
      batchAccept++;
    }
    
    ///////////////
    //fit 
    ///////////////
    for(i = 0; i < n; i++){
      tmp_n[i] = rnorm(ddot_(&p, &X[i], &n, beta, &inc) + w[i], sqrt(tauSq));
    }
    
    if(s >= nSamplesStart){
      dcopy_(&p, beta, &inc, &betaSamples[sKeep*p], &inc);
      tauSqSamples[sKeep] = tauSq;
      sigmaSqSamples[sKeep] = sigmaSq;
      phiSamples[sKeep] = phi;
      dcopy_(&n, w, &inc, &wSamples[sKeep*n], &inc);
      dcopy_(&n, tmp_n, &inc, &fittedSamples[sKeep*n], &inc);
      sKeep++;
    }
    
    /////////////////////////////////////////////
    //report
    /////////////////////////////////////////////
    if(status == nReport){
      cout << "percent complete: " << 100*s/nSamples << endl;     
      cout << "theta: " << 100.0*batchAccept/nReport << endl;
      batchAccept = 0;
      
      cout << "---------------" << endl;
      status = 0;
      
    }
    status++;
    
  }

  cout << "Runtime: " << get_wall_time() - wall_start_0 << endl;
  
  writeRMatrix(outFile+"-beta", betaSamples, p, nSamplesKeep);
  writeRMatrix(outFile+"-phi", phiSamples, 1, nSamplesKeep);
  writeRMatrix(outFile+"-sigmaSq", sigmaSqSamples, 1, nSamplesKeep);
  writeRMatrix(outFile+"-tauSq", tauSqSamples, 1, nSamplesKeep);
  writeRMatrix(outFile+"-w", wSamples, n, nSamplesKeep);
  //writeRMatrix(outFile+"-fitted", fittedSamples, n, nSamplesKeep); 

  return(0);
}

void writeRMatrix(string outfile, double * a, int nrow, int ncol){

    ofstream file(outfile.c_str());
    if ( !file ) {
      cerr << "Data file could not be opened." << endl;
      exit(1);
    }
    
    for(int i = 0; i < nrow; i++){
      for(int j = 0; j < ncol-1; j++){
	file << setprecision(10) << fixed << a[j*nrow+i] << "\t";
      }
      file << setprecision(10) << fixed << a[(ncol-1)*nrow+i] << endl;    
    }
    file.close();
}


void writeRMatrix(string outfile, int* a, int nrow, int ncol){
  
  ofstream file(outfile.c_str());
  if ( !file ) {
    cerr << "Data file could not be opened." << endl;
    exit(1);
  }
  
  
  for(int i = 0; i < nrow; i++){
    for(int j = 0; j < ncol-1; j++){
      file << fixed << a[j*nrow+i] << "\t";
    }
    file << fixed << a[(ncol-1)*nrow+i] << endl;    
  }
  file.close();
}

void mvrnorm(double *des, double *mu, double *cholCov, int dim){
  
  int i;
  int inc = 1;
  double one = 1.0;
  double zero = 0.0;
  
  for(i = 0; i < dim; i++){
    des[i] = rnorm(0, 1);
  }
 
  dtrmv_("L", "N", "N", &dim, cholCov, &dim, des, &inc);
  daxpy_(&dim, &one, mu, &inc, des, &inc);
}

void show(double *a, int n){
  for(int i = 0; i < n; i++)
    cout << setprecision(20) << fixed << a[i] << endl;
}


void show(int *a, int n){
  for(int i = 0; i < n; i++)
    cout << fixed << a[i] << endl;
}


void zeros(double *a, int n){
  for(int i = 0; i < n; i++)
    a[i] = 0.0;
}


void show(double *a, int r, int c){

  for(int i = 0; i < r; i++){
    for(int j = 0; j < c; j++){
      cout << fixed << a[j*r+i] << "\t";
    }
    cout << endl;
  }
}


void show(int *a, int r, int c){

  for(int i = 0; i < r; i++){
    for(int j = 0; j < c; j++){

      cout << fixed << a[j*r+i] << "\t";
    }
    cout << endl;
  }
}

void covTransInv(double *z, double *v, int m){
  int i, j, k;

  for(i = 0, k = 0; i < m; i++){
    for(j = i; j < m; j++, k++){
      v[k] = z[k];
      if(i == j)
	v[k] = exp(z[k]);
    }
  }

}

void covTrans(double *v, double *z, int m){
  int i, j, k;

  for(i = 0, k = 0; i < m; i++){
    for(j = i; j < m; j++, k++){
      z[k] = v[k];
      if(i == j)
	z[k] = log(v[k]);
    }
  }

}

void covTrans(vector<double> v, double *z, int m){
  int i, j, k;

  for(i = 0, k = 0; i < m; i++){
    for(j = i; j < m; j++, k++){
      z[k] = v[k];
      if(i == j)
	z[k] = log(v[k]);
    }
  }

}

void covTransInvExpand(double *v, double *z, int m){
  int i, j, k;
  
  zeros(z, m*m);
  for(i = 0, k = 0; i < m; i++){
    for(j = i; j < m; j++, k++){
      z[i*m+j] = v[k];
      if(i == j)
	z[i*m+j] = exp(z[i*m+j]);
    }
  }
  
}

void covExpand(double *v, double *z, int m){
  int i, j, k;
  
  zeros(z, m*m);
  for(i = 0, k = 0; i < m; i++){
    for(j = i; j < m; j++, k++){
      z[i*m+j] = v[k];
    }
  }
  
}

double logit(double theta, double a, double b){
  return log((theta-a)/(b-theta));
}

double logitInv(double z, double a, double b){
  return b-(b-a)/(1+exp(z));
}

double dist2(double &a1, double &a2, double &b1, double &b2){
  return(sqrt(pow(a1-b1,2)+pow(a2-b2,2)));
}
