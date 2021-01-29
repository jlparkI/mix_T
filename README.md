# mix_T
Scipy and sklearn do not at present have many resources for fitting data using Student t-distributions, even though these are widely used for modeling heavy-tailed data. This package (in progress, finished soon) will provide classes for:

1) fitting a dataset to a mixture of multivariate Student T distributions with a scikit-learn-like interface so that user can sample from the mixture, calculate AIC, BIC etc.;
2) fitting a single multivariate student T to a dataset and sampling from it;

May eventually merge this with MAP-DPMM...
