VariationalStudentMixture
=========================

 - [Parameters](#Parameters)
 - [Attributes](#Attributes)
 - [Methods](#Methods)

```python
import studenttmixture
from studenttmixture import VariationalStudentMixture

VariationalStudentMixture(n_components=2, tol=1e-4, max_iter=2000, n_init=1, df=4.0, 
fixed_df=True, random_state=123, verbose=False, init_type="k++", scale_inv_prior=None,
loc_prior=None, mean_cov_prior = 1e-2, weight_conc_prior=None, wishart_dof_prior = None,
max_df = 100)
```

### Parameters

  * *n_components*<br>The number of components in the mixture.
  * *tol*<br>The threshold for change in complete data log likelihood to achieve convergence.
  * *max_iter*<br>The maximum number of iterations.
  * *n_init*<br>The mean-field algorithm converges on a local maximum, so re-running the algorithm several
times with different starting points may improve the chances of obtaining an optimal outcome. n_init
determines the number of re-initializations to run.
  * *df*<br>The starting value for degrees of freedom for each component. If fixed_df is True,
all components use a fixed degrees of freedom with the specified value; if it is False,
this starting value is optimized during fitting.
  * *fixed_df*<br>Whether to optimize degrees of freedom or use a fixed user-specified value. 
Optimizing df may provide a better fit and may be informative but also tends to result in slower
convergence.
  * *random_state*<br>The seed for the random number generator.
  * *verbose*<br>Whether to print the lower bound and change in lower bound during fitting.
  * *init_type*<br>Must be one of "k++" or "kmeans". If "k++", the component locations are initialized
the KMeans++ procedure described by Arthur and Vassilvitskii (2007), which generally gives reasonably good
starting locations. If "kmeans", these starting locations are further refined using kmeans clustering.
  * *scale_inv_prior*<br>The prior for the inverse of the scale matrix (the precision matrix). If supplied,
must be a square positive definite matrix of size D x D. If None is indicated, the model will use a reasonable
default (the inverse of the diagonal matrix constructed using the diagonal elements of the covariance
of the full dataset).
  * *loc_prior*<br>The prior for the location components. If specified, must be a numpy array of shape D. If
None is indicated (default), the model will use a reasonable default (the mean of the full dataset).
  * *mean_cov_prior*<br>Indicates how strongly you believe in the location prior. Must be greater than 0. A large
value will tend to pull all components towards *loc_prior*; small values will ensure the locations are determined
by the data rather than the prior.
  * *weight_conc_prior*<br>Determines whether the model prefers many or few components. This value is used for
the Dirichlet prior on the mixture weights. A small value will place most of the weight of the Dirichlet prior
on the corners of the probability simplex, so that the model will tend to prefer solutions with few components and
will kill off components that are not significantly improving the likelihood. A small value for weight_conc_prior will
therefore mean that n_components becomes an upper bound. A large value for *weight_conc_prior* will cause the model
to use as many components as possible.
  * *wishart_dof_prior*<br>The degrees of freedom parameter for the Wishart prior distribution on the scale matrices. This
indicates the strength of our belief in *scale_inv_prior*. For numerical stability, this must be > the dimensionality of
the data - 1. If None is indicated, the model uses the dimensionality of the data as a reasonable default.
  * *max_df*<br>If degrees of freedom is not fixed, i.e. if the model is set to optimize degrees of freedom, and the
data is normally distributed, convergence can sometimes be slowed significantly as the degrees of freedom of each
component is increased slightly on each iteration. Student's t-distributions with degrees of freedom > 100 behave 
sufficiently like Gaussians for all practical intents and purposes, so that optimizing the degrees of freedom past a
certain point may slow convergence without achieving any significant benefit. Using a max_df value can therefore speed
convergence. If you would like to fit without any *max_df*, set *max_df* to np.inf .


### Attributes

| Attribute     | Description |
| ---------- | ----------- |
| location     | A K x D array for K components, D dimensions where each row k is the location (analogous to mean of a Gaussian) for component k. |
| scale | A D x D x K array for K components, D dimensions where each D x D array is the scale (analogous to covariance matrix for a Gaussian) matrix for component k. |
| degrees_of_freedom | A size K array for K components where each element is the degrees of freedom for component k. If you specify fixed_df = True, the starting value is used and is not optimized. fixed_df = False, by contrast, will ensure that degrees_of_freedom is optimized. |
| mix_weights | A size K array for K components where each element k is the weight of each mixture component (the probability that a random draw is from component k). |


### Methods

*Note that for all methods, the input X is a numpy array where each row is a datapoint and each column is
a feature.*

| Method     | Description |
| ---------- | ----------- |
| fit(X)     | Fit the model to the input data X. |
| fit_predict(X) | Returns a numpy array of predictions for the input data (after first fitting the model). Each prediction is an integer in [0,n_components-1]. These predictions are "hard" assignments obtained by assigning each datapoint to the cluster for which it has the largest posterior probability (see *predict_proba* below). |
| predict(X) | Returns a numpy array of predictions for the input data. Can only be called for a model that has already been fitted. Each element i is an integer in [0,n_components-1] and is generated by assigning each datapoint to the cluster for which it has the largest posterior probability. |
| predict_proba(X) | Returns a numpy array of dimensions (NxK) for N input datapoints and K components, where each element ij is the posterior probability that sample i belongs to cluster j. Assigning each datapoint to the cluster for which it has the highest posterior probability yields the "hard" assignments generated by *predict* and *fit_predict*. |
| score(X)   | Returns the average (over datapoints) log likelihood of the input data. When n_init is > 1, during fitting, the algorithm chooses to keep the set of parameters that yield the largest score. |
| score_samples(X) | Returns the weighted log likelihood for each datapoint. |
| sample(num_samples = 1, random_seed=123) | Generates samples from a fitted VariationalStudentMixture, using the random number generator seed you provide. |

