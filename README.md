# studenttmixture

Mixtures of multivariate Student's t distributions are widely used for clustering
data that may contain outliers, but scipy and scikit-learn do not at present
offer classes for fitting Student's t mixture models. This package provides classes
for:

1) Modeling / clustering a dataset using a finite mixture of multivariate Student's
t distributions fit via the EM algorithm. You can select the number of components
using either prior knowledge or the information criteria calculated by the model
(e.g. AIC, BIC).
2) Modeling / clustering a dataset using a mixture of multivariate Student's 
t distributions fit via the variational mean-field approximation. Depending on the
hyperparameters you select, the fitting process will automatically "choose" an 
appropriate number of clusters, so the number of components in this case acts
as an upper bound.
3) An infinite mixture of Student's t-distributions (i.e. a Dirichlet process). In practice,
this model is fitted using some small modifications to the mean-field recipe and has
some of the same advantages and limitations.

(1) and (2) are currently available; (3) will be available in version 0.0.3.

Unittests for the package are in the tests folder.

### Installation

    pip install studenttmixture

### Usage

- [EMStudentMixture](https://github.com/jlparkI/mix_T/blob/main/Documentation/Finite_Mixture_Docs.md)<br>
- [VariationalStudentMixture](https://github.com/jlparkI/mix_T/blob/main/Documentation/Variational_Mixture_Docs.md)<br>
- [Tutorial: Modeling with mixtures of t-distributions](https://github.com/jlparkI/mix_T/blob/main/Documentation/Tutorial.md)<br>


### Background

- [Deriving the mean-field formula](https://github.com/jlparkI/mix_T/blob/main/Documentation/variational_mean_field.pdf)<br>
