# mix_T
Scipy and sklearn do not at present have any resources for fitting multivariate Student t-distributions or mixtures thereof, even though these are widely used for modeling heavy-tailed data. This package (in progress, finished soon) will provide classes for:

1) fitting a dataset to a finite mixture of multivariate Student T distributions 
with a scikit-learn-like interface and calculating key statistics (BIC, AIC etc.);
2) fitting a dataset to a mixture of multivariate Student T distributions using
a variational approximation.
