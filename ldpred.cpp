// [[Rcpp::plugins(cpp11)]]
#define ARMA_64BIT_WORD

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;


// [[Rcpp::export]]
NumericMatrix ldpred_gibbs(const arma::sp_mat& corr,        /* SNP correlation matrix, shape M x M */
                           const NumericVector& betas_hat,  /* Beta estimation, shape M x 1 */
                           const NumericVector& n_vec,      /*  */
                           const NumericVector& p_vec,      /* Propotion of causal, shape K x 1 */
                           double coeff,                    /* coeff = n * h2 / M */
                           int burn_in = 10, 
                           int num_iter = 60,
                           bool sparse = false) {

  /* 
  betas_hat = gammas_hat / se(gammas_hat) / sqrt(n)
  */
  int m = betas_hat.size();
  int np = p_vec.size();
  NumericMatrix ldpred_shrink(m, np);

  #pragma omp parallel for num_threads(4)
  for (int i = 0; i < np; i++) {

    double p = p_vec[i];
    #pragma omp critical
    Rcout << "p = " << p << std::endl;

    // some useful constants
    double L = coeff / p;                   /* n * h2 / (M * p) */
    double C1 = (1 - p) / p * sqrt(1 + L);  /* (1 - p) / p * sqrt(1 + n * h2 / (M * p)) */
    double C2 = L / (L + 1);               
    NumericVector C3 = (-C2 / 2) * n_vec;
    NumericVector C4 = sqrt(C2 / n_vec);

    NumericVector curr_post_means(m); /* omega */
    /* Algo1 line 2 */
    NumericVector avg_betas(m); /* Big Omega */
    arma::vec curr_betas(m, arma::fill::zeros);
    /* Algo1 LDPred line 3 */
    for (int k = 1; k <= num_iter; k++) {
      /* Algo1 line 4 */
      for (int j = 0; j < m; j++) {
        curr_betas[j] = 0;
        /* Alog1 line 5, calc via page 8 equation 3 */
        double res_beta_hat_j = betas_hat[j] - arma::dot(corr.col(j), curr_betas); 
        /* Algo1 line 6, calc via page 8 equation 4 */
        double postp = 1 / (1 + C1 * ::exp(C3[j] * res_beta_hat_j * res_beta_hat_j)); 
        if (sparse && (postp < p)) {
          curr_betas[j] = curr_post_means[j] = 0;
        } else {
          /* Algo1 line 8, calc via page 9 equation 6 */
          curr_post_means[j] = C2 * postp * res_beta_hat_j;
          /* Algo1 line 7, calc via page 8 equation 5 */
          curr_betas[j] = (postp > ::unif_rand()) ? C4[j] * ::norm_rand() + C2 * res_beta_hat_j : 0;
        }
      }
      /* Alog1 line 11-12 */
      if (k > burn_in) 
        avg_betas += curr_post_means;
    }

    /* Alog1 line 15-16 */
    #pragma omp critical
    ldpred_shrink(_, i) = (avg_betas / (num_iter - burn_in)) / betas_hat;
  }

  return ldpred_shrink;
}
