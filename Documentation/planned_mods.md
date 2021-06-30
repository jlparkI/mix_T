### Planned modifications for version 0.0.2.2 - 0.0.3

- Add the ability to initialize cluster centers using scikit-learn's kmeans.
- Improve the k++ algorithm for initializing cluster centers in this package.
- Both EMStudentMixture and VariationalStudentMixture will be modified to inherit
from a base class to avoid redundancy (since some functions are shared between them).
- The infinite mixture option will be added to VariationalStudentMixture.
- The variational lower bound calculation needs to be simplified and reorganized.
- Parts of the EM and variational class will be rewritten to provide
additional optimization / speed, switching to C where necessary. (As of version
0.0.2.1, the squared mahalanobis distance calculation is already rewritten in C,
providing a large speed increase for training, especially if df is fixed.)
