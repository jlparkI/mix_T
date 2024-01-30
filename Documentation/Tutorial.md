Fitting mixtures of t-distributions -- a tutorial
===================

- [Introduction](#Introduction)
- [Fitting finite mixtures with EM](#Expectation-Maximization)
- [The variational mean-field alternative](#VariationalMean-Field)
- [Variational mean-field: choosing hyperparameters](#HyperparameterSelection)

### Additional Background
- [What is mean-field?](https://jlparki.github.io/mean_field.pdf)<br>

# Introduction

Like all other clustering techniques, mixture models have advantages and disadvantages and are only optimal
for certain kinds of problems. The most common mixture model is the Gaussian mixture,
which assumes all clusters are approximately normal distributions. The number of components can 
be chosen using an information criterion-based heuristic (a "finite mixture") 
or by fitting either finite or infinite mixtures using a variational mean-field approximation; 
this approach automatically selects the number of
clusters most appropriate given the data and a user-specified prior. Alternatively, mixture models
can be fitted using MCMC sampling techniques (at considerably greater computational expense).
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
The moral of the story is that most of the time, if you're going to cluster data using a mixture model,
a mixture of Student's t-distributions is a better bet than a Gaussian mixture. If there are no
outliers, you'll get a clustering similar to what you would get from a Gaussian mixture. If there
are outliers...the Gaussian mixture may be useless, while the Student's t-mixture can still perform
well.

This package does not provide functions or classes for MCMC-based fitting at this time because the excellent
PyMC3 package already provides good tools for that purpose. Rather, it focuses
on fitting using either EM or variational mean-field. These approaches are much less computationally expensive,
less well adapted to inference and significantly more suitable for making predictions. Just 
like scitkit-learn, then, these classes focus on prediction rather than inference.
