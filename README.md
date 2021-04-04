# studenttmixture

Mixtures of multivariate Student's t distributions are widely used for clustering
data that may contain outliers, but scipy and scikit-learn do not at present
offer classes for fitting Student's t mixture models. This package provides classes
for:

1) Modeling / clustering a dataset using a finite mixture of multivariate Student's
t distributions;
2) Modeling / clustering a dataset using an infinite mixture of Student's t-
distributions (a Dirichlet process; fit using a variational approximation).

(1) is available in version 0.0.1.1, (2) will come in version 0.0.2.

### Installation

    pip install studenttmixture

### Usage

- [FiniteStudentMixture](https://github.com/jlparkI/mix_T/blob/main/Documentation/Finite_Mixture_Docs.md)<br>
- [VariationalStudentMixture (coming soon)](https://github.com/jlparkI/mix_T/blob/main/Documentation/Variational_Mixture_Docs.md)<br>
- [Student's T mixture modeling user guide](https://github.com/jlparkI/mix_T/blob/main/Documentation/Tutorial.md)<br>

