VariationalStudentMixture
=========================

 - [Parameters](#Parameters)
 - [Attributes](#Attributes)
 - [Methods](#Methods)
 - [Example](#Example)

```python
import studenttmixture
from studenttmixture import VariationalStudentMixture

VariationalStudentMixture(n_components=2, tol=1e-5, max_iter=2000, n_init=1, df=4.0, 
fixed_df=True, random_state=123, verbose=False, init_type="kmeans", scale_inv_prior=None,
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
  * *init_type*<br>If "kmeans", the component locations are initialized using k-means
  clustering using the scikit-learn library. If "k++", the component locations are initialized
the KMeans++ procedure described by Arthur and Vassilvitskii (2007). KMeans is the default and
generally gives good results.
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
| purge_empty_clusters(X, empty_cluster_threshold = 1) | Removes any clusters that contain fewer than *empty_cluster_threshold* datapoints (using the MAP assignment generated by *predict*). Note that the X you supply *must* be the training set -- do not supply another dataset that is not the training set to this function. This function is useful to remove empty clusters in situations where the mean-field algorithm "kills off" unneeded components during fitting. |
| score(X)   | Returns the average (over datapoints) log likelihood of the input data. When n_init is > 1, during fitting, the algorithm chooses to keep the set of parameters that yield the largest score. |
| score_samples(X) | Returns the weighted log likelihood for each datapoint. |
| sample(num_samples = 1, random_seed=123) | Generates samples from a fitted VariationalStudentMixture, using the random number generator seed you provide. |

### Example

```python
#This is a fairly easy clustering problem with well-separated clusters
#(in other words, unlike any real-world dataset!) However, we show that for
#this toy problem, with well-chosen priors, a mixture fit using 
#variational mean-field can automatically choose an appropriate number
#of clusters for our problem.

import studenttmixture
from studenttmixture import VariationalStudentMixture as VSM
import matplotlib.pyplot as plt, numpy as np

#Note that scikit-learn is not a dependency and is not
#required for studenttmixture; we're only using it here to get
#some convenient toy datasets to demonstrate use of this package, and
#to compare the results of a Gaussian mixture model with a 
#studenttmixture.
import sklearn
from sklearn.datasets import make_blobs

#We generate a dataset with 3 blobs
x_clusters, y, centers = make_blobs(n_samples=100, 
                           return_centers=True, random_state=456)
np.random.seed(456)
x_noise = np.random.uniform(low=2*np.min(x_clusters), 
                            high=2*np.max(x_clusters),
                           size=(5,2))
x = np.vstack([x_clusters, x_noise])
plt.scatter(x[:,0], x[:,1], s=10)
plt.title("Unclustered data")
plt.show()

```
![raw_data](https://github.com/jlparkI/mix_T/blob/main/Documentation/unclustered_data_var.png)

```python
#In this case, unlike for EM, we can choose the number of clusters automatically -- depending on 
#what values we choose for our hyperparameters. We use a small value for the weight conc prior --
#this essentially tells the algorithm we expect there to be relatively few clusters and will
#lead to solutions where some or all of n_components may not be used. A large value
#for weight_conc_prior, by contrast, would cause the algorithm to use all n_components.

#If we can get the algorithm to select the number of clusters with n_components as an upper bound
#by using a small weight_conc_prior, why not set n_components to some really high value? We could, but...
#this will lead to really slow fitting and convergence. One good strategy is to set
#n_components higher than the largest number of clusters that might be present, but not to some
#absurdly large value, and use a small value for weight_conc_prior.

#Finally, as always with Bayesian modeling, priors are important! Changing your priors CAN give you 
#different results. Sometimes this can be very beneficial. If we have a good idea what the covariance
#matrix of each cluster should look like, for example, supplying a prior for that could be useful
#and might enable us to get better results than we could with EM.

student_model = VSM(n_components=6, tol=1e-5, weight_conc_prior = 1e-3, 
        mean_cov_prior = 1, df=4, random_state = 1, fixed_df=True, n_init=3)

student_model.fit(x)




#After fitting, we plot the mixture weights. We notice that -- as expected -- the model has 
#"killed off" two clusters and almost killed off a third (by assigning it to a distant outlier).
#With EM we could have made this same determination by fitting the model multiple times and 
#choosing the model with the best BIC / AIC.

weights = student_model.mix_weights

plt.style.use("ggplot")
plt.bar(np.arange(6), weights)
plt.xlabel("Component")
plt.ylabel("Mixture weight")
plt.title("Mixture weights for each component\nwith a variational mixture")
plt.show()

```
![raw_data](https://github.com/jlparkI/mix_T/blob/main/Documentation/prepurge_mix_weights.png)
![raw_data](https://github.com/jlparkI/mix_T/blob/main/Documentation/prepurge_clustering.png)

```python
#We could refit the model although if we like what we see and only want to remove the empty clusters, 
#we can do so using the purge_empty_clusters command. We must choose a threshold -- a minimum
#number of datapoints assigned to a cluster -- such that it is considered empty. Based on the
#mixture weight plot, we choose 2 as a reasonable threshold, which will leave us with three clusters --
#our expected number.
student_model.purge_empty_clusters(x, empty_cluster_threshold = 2)

#After purging empty clusters, we have three remaining, which makes perfect sense. Fitting using
#mean-field allowed us to reach this result via a different route than EM. Importantly, it also
#enabled us to incorporate any prior knowledge we might have, which for some problems is helpful.

weights = student_model.mix_weights

plt.style.use("ggplot")
plt.bar(np.arange(3), weights)
plt.xlabel("Component")
plt.ylabel("Mixture weight")
plt.title("Mixture weights for each component\nwith a variational mixture")
plt.show()

sns.scatterplot(x=x[:,0], y=x[:,1], hue=[str(z) for z in student_model.predict(x)])
```
![raw_data](https://github.com/jlparkI/mix_T/blob/main/Documentation/postpurge_mix_weights.png)
![raw_data](https://github.com/jlparkI/mix_T/blob/main/Documentation/postpurge_clustering.png)
