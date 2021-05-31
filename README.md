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
as an upper bound. In many cases this can be a significant advantage, but of course
the hyperparameters may require some tuning, and the variational approach makes
some subtle assumptions that may have impact the quality of the fit, especially for
small datasets. Nonetheless, for some problems the ability to automatically select the
number of clusters can make this a powerful tool.

(1) is available in version 0.0.1, (2) will be available in version 0.0.2.

Unittests for the package are in the tests folder.

### Installation

    pip install studenttmixture

### Usage

- [EMStudentMixture](https://github.com/jlparkI/mix_T/blob/main/Documentation/Finite_Mixture_Docs.md)<br>
- [VariationalStudentMixture (coming soon)](https://github.com/jlparkI/mix_T/blob/main/Documentation/Variational_Mixture_Docs.md)<br>
- [Student's T mixture modeling user guide](https://github.com/jlparkI/mix_T/blob/main/Documentation/Tutorial.md)<br>


### Background

- [What is variational mean-field?](https://github.com/jlparkI/mix_T/blob/main/Documentation/variational_mean_field.pdf)<br>
