/******************************************************************************/

#include <bigstatsr/arma-strict-R-headers.h>
#include <bigstatsr/utils.h>
#include <bigsparser/SFBM.h>

/******************************************************************************/

inline double square(double x) {
  return x * x;
}

/******************************************************************************/

/*!
* \param sfbm SNP correlation matrix, shape M x M, M is the marker size
* \param beta_hat Scaled beta_estimation, shape M x 1, beta_hat = beta / scale, where 
* scale = sqrt(n * beta_se^2 + beta ^ 2), n is sample size
* \param beta_init Initial value of beta guess, usually be zeros, shape M x 1
* \param order Sequence from 0 to M - 1, shape M x 1
* \param n_vec Sample size to estimate effective size (namely beta), shape n x 1
* \param h2 Value of heritability
* \param p Value of proportion of causal variants
* \param sparse Force sparse the value of estimation or not
* \param burn_in Number of burn in for Gibbs sampling
* \param num_iter Number of iteration for Gibbs sampling
*/

arma::vec ldpred2_gibbs_one(XPtr<SFBM> sfbm,
                            const arma::vec& beta_hat,
                            const NumericVector& beta_init,
                            const IntegerVector& order,
                            const NumericVector& n_vec,
                            double h2,
                            double p,
                            bool sparse,
                            int burn_in,
                            int num_iter) {

  int m = beta_hat.size();
  arma::vec curr_beta(beta_init.begin(), m);
  arma::vec avg_beta(m, arma::fill::zeros);

  double h2_per_var = h2 / (m * p);
  double inv_odd_p = (1 - p) / p;
  double gap0 = arma::dot(beta_hat, beta_hat);

  for (int k = -burn_in; k < num_iter; k++) {

    double gap = 0;

    for (const int& j : order) {
      /* Page 8 equation 3 */
      double dotprod = sfbm->dot_col(j, curr_beta);
      double resid = beta_hat[j] - dotprod;
      gap += resid * resid;
      double res_beta_hat_j = curr_beta[j] + resid;

      double C1 = h2_per_var * n_vec[j];
      double C2 = 1 / (1 + 1 / C1);
      double C3 = C2 * res_beta_hat_j;
      double C4 = ::sqrt(C2 / n_vec[j]);
      /* line 6, page 8 equation 4 */
      double post_p_j = 1 / (1 + inv_odd_p * ::sqrt(1 + C1) * ::exp(-square(C3 / C4) / 2));

      if (sparse && (post_p_j < p)) {
        curr_beta[j] = 0;
      } else {
        /* line 7, page 8 equation 5 */
        curr_beta[j] = (post_p_j > ::unif_rand()) ? ::Rf_rnorm(C3, C4) : 0;
        /* line 11-12 */
        if (k >= 0) 
          avg_beta[j] += C3 * post_p_j; /* line 8, page 9 equation 6 */
      }
    }
    /* check divergence */
    if (gap > gap0) { 
      avg_beta.fill(NA_REAL); 
      return avg_beta; 
    }
  }

  return avg_beta / num_iter;
}

/******************************************************************************/

/*!
* \param sfbm SNP correlation matrix, shape M x M, M is the marker size
* \param beta_hat Scaled beta_estimation, shape M x 1, beta_hat = beta / scale, where 
* scale = sqrt(n * beta_se^2 + beta ^ 2), n is sample size
* \param beta_init Initial value of beta guess, usually be zeros, shape M x 1
* \param order Sequence from 0 to M - 1, shape M x 1
* \param n_vec Sample size to estimate effective size (namely beta), shape n x 1
* \param h2 Values of heritability to do grid search, shape K x 1
* \param p Value of proportion of causal variants to do grid search, shape K x 1
* \param sparse Force sparse the value of estimation or not, to do grid search, shape K x 1
* \param burn_in Number of burn in for Gibbs sampling
* \param num_iter Number of iteration for Gibbs sampling
* \param ncores Number of CPU cores to use
*/


// [[Rcpp::export]]
arma::mat ldpred2_gibbs(Environment corr,
                        const NumericVector& beta_hat,
                        const NumericVector& beta_init,
                        const IntegerVector& order,
                        const NumericVector& n_vec,
                        const NumericVector& h2,
                        const NumericVector& p,
                        const LogicalVector& sparse,
                        int burn_in,
                        int num_iter,
                        int ncores) {

  XPtr<SFBM> sfbm = corr["address"];

  int m = beta_hat.size();
  myassert_size(sfbm->nrow(), m);
  myassert_size(sfbm->ncol(), m);
  myassert_size(order.size(), m);
  myassert_size(beta_init.size(), m);
  myassert_size(n_vec.size(), m);

  int K = p.size();
  myassert_size(h2.size(),     K);
  myassert_size(sparse.size(), K);

  arma::mat res(m, K);

  #pragma omp parallel for schedule(dynamic, 1) num_threads(ncores)
  for (int k = 0; k < K; k++) {
    arma::vec res_k = ldpred2_gibbs_one(
      sfbm, beta_hat, beta_init, order, n_vec,
      h2[k], p[k], sparse[k], burn_in, num_iter);

    #pragma omp critical
    res.col(k) = res_k;
  }

  return res;
}

/******************************************************************************/
