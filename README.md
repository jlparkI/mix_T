# studenttmixture

NOTE: As of version 0.0.2.2, this package uses scikit-learn's KMeans clustering
to initialize component locations. This adds an additional dependency (in
addition to numpy and scipy) but 
gives faster convergence, since KMeans provides a good way to choose initial
cluster centers that are refined by the mixture model.

Mixtures of multivariate Student's t distributions are widely used for clustering
data that may contain outliers, but scipy and scikit-learn do not at present
offer classes for fitting Student's t mixture models. This package provides classes
for:

1) Modeling / clustering a dataset using a finite mixture of multivariate Student's
t distributions fit via the EM algorithm. You can select the number of components
using either prior knowledge or the information criteria calculated by the model
(AIC, BIC).
2) Modeling / clustering a dataset using a mixture of multivariate Student's 
t distributions fit via the variational mean-field approximation. Depending on the
hyperparameters you select, the fitting process may kill off unneeded clusters, 
so the number of components in this case acts as an upper bound.
3) Modeling / clustering an infinite mixture of Student's t-distributions (i.e. a Dirichlet process). In practice,
this model is fitted using some small modifications to the mean-field recipe and has
some of the same advantages and limitations.

(1) and (2) are currently available; (3) will be available in version 0.0.3.

Unittests for the package are in the tests folder.

### Installation

    pip install studenttmixture

Note that starting in version 0.0.2.3, this package contains C extensions and is therefore
distributed as a source distribution which is automatically compiled on install. 
This is a little less convenient but provides a large speed increase.

It is unusual but problems with source distribution pip packages that contain C extensions are occasionally
observed on Windows, e.g. an error similar to this:

    error: Microsoft Visual C++ 14.0 is required.

in the unlikely event you encounter this, I recommend the solution described under this 
[StackOverflow and links](https://stackoverflow.com/questions/44951456/pip-error-microsoft-visual-c-14-0-is-required).

Finally, if you for whatever reason prefer the pure Python version, install version 0.0.2.2, i.e.:

    pip install studenttmixture==0.0.2.2

training for mixture models will run slower but no compilation is required.

### Usage

- [EMStudentMixture](https://github.com/jlparkI/mix_T/blob/main/Documentation/Finite_Mixture_Docs.md)<br>
- [VariationalStudentMixture](https://github.com/jlparkI/mix_T/blob/main/Documentation/Variational_Mixture_Docs.md)<br>
- [Tutorial: Modeling with mixtures](https://github.com/jlparkI/mix_T/blob/main/Documentation/Tutorial.md)<br>

### Background

- [Deriving the mean-field formula](https://jlparki.github.io/mean_field.pdf)<br>

### Upcoming in future versions

- [Planned for version 0.0.3](https://github.com/jlparkI/mix_T/blob/main/Documentation/planned_mods.md)
