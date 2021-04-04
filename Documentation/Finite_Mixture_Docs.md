FiniteStudentMixture
===================

```python
from finite_mixture import FiniteStudentMixture

FiniteStudentMixture(n_components=2, tol=1e-3,
reg_covar=1e-6, max_iter=500, n_init=1, df=4, fixed_df=True,
random_state=123, verbose=False)
```

#### Parameters

  * n_components<br>The number of components in the mixture.
  * tol<br>The threshold for change in complete data log likelihood to achieve convergence.
  * reg_covar<br> A small floating point value > 0 added to the diagonal of the 
covariance (aka "scale", since this is a t-distribution) matrix for each component
to ensure it is positive definite.
  * max_iter<br>The maximum number of iterations. If the fit does not converge in < max_iter,
the fitting process is terminated.
  * n_init<br>The EM algorithm converges on a local maximum, so re-running the algorithm several
times with different starting points may improve the chances of obtaining an optimal outcome. n_init
determines the number of re-initializations to run.
  * df<br>The starting value for degrees of freedom for each component. If fixed_df is True,
all components use a fixed degrees of freedom with the specified value; if it is False,
this starting value is optimized during fitting.
  * fixed_df<br>Whether to optimize degrees of freedom or use a fixed user-specified value. 
Optimizing df may provide a better fit and may be informative but also tends to result in slower
convergence.
  * random_state<br>The seed for the random number generator.
  * verbose<br>Whether to print the lower bound and change in lower bound during fitting.

#### Methods

aic(X)
bic(X)
fit(X)
fit_predict(X)
predict(X)
predict_proba(X)
score(X)
score_samples(X)
get_cluster_centers()
get_cluster_scales()
get_weights()

