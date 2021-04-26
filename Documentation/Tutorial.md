Modeling data using finite and infinite mixtures of t-distributions -- a tutorial
===================

- [Introduction](#Introduction)
- [Finite Mixtures](#FiniteMixtures)
- [Infinite Mixtures](#InfiniteMixtures)


# Introduction

Hierarchical clustering, k-means and mixture models are among the simplest and most
popular techniques for clustering (as of 2021). K-means owes its popularity to excellent scaling but
assumes all clusters are spherical and thus routinely delivers poor results; it's trivial to generate
a toy dataset where k-means will perform very badly. Hierarchical clustering is a useful tool for
small datasets but its O(N^2) scaling makes it impractical for larger datasets. Density-based
clustering (aka DBSCAN) is in principle very powerful but in practice is highly sensitive to choice
of hyperparameters, and spectral clustering is too computationally expensive for routine use.
<br><br>
Like all these other techniques, mixture models have advantages and disadvantages and are only optimal
for certain kinds of problems. The most common mixture model is the Gaussian mixture,
which assumes all clusters are approximately normal distributions. The number of components can 
be chosen using an information criterion-based heuristic (a "finite mixture") 
or by fitting an infinite mixture (Dirichlet process) model fit using a variational 
approximation; this approach automatically selects the number of
clusters most appropriate given the data and a user-specified prior. Gaussian mixtures scale
well with number of datapoints although exhibit cubic scaling with increasing dimensionality.
<br><br>
The assumption that all clusters are normal distributions might seem arbitrary and fairly problematic.
One does however encounter a fair number of datasets where this assumption is approximately true or
at least reasonable, and hence a Gaussian mixture is sometimes a useful approach. Its usefulness,
however, is limited by the fact that it can easily be thrown off by the presence of a sprinkling of
outliers. Consider the toy dataset below, clustered using either a mixture of Student's t-distributions
or a Gaussian mixture. Even with 5 restarts, the presence of some outliers was enough to make the 
Gaussian mixture worse than useless.
<br>
![mixture_comp](https://github.com/jlparkI/mix_T/blob/main/Documentation/STM_vs_GMM.png)
<br>
The moral of the story is that most of the time, if you're going to fit a mixture model,
a mixture of Student's t-distributions is a better bet than a Gaussian mixture. If there are no
outliers, you'll get a clustering similar to what you would get from a Gaussian mixture. If there
are outliers...the Gaussian mixture may be useless, while the Student's t-mixture can still perform
quite well. Unfortunately scikit-learn and the other usual suspects don't provide tools for fitting
a mixture of Student's t-distributions. This package provides classes for fitting a finite mixture
(currently available) and an infinite mixture (in the works).

# FiniteMixtures

Documentation not added yet; coming soon.

# InfiniteMixtures

Not added yet, coming soon.
