### Planned modifications for version 0.0.3

- Both EMStudentMixture and VariationalStudentMixture will be modified to inherit
from a base class to avoid redundancy (since some functions are shared between them).
- The infinite mixture option will be added to VariationalStudentMixture.
- For large D, the rate-limiting factor becomes the Cholesky decomposition of the
scale matrices. For large N and small D, by contrast, the rate limiting factor becomes
the calculation of the squared mahalanobis distance. The squared mahalanobis distance
calculation will be rewritten in C and wrapped using Cython to provide improved speed
(there is a simple optimization we can add to this calculation in C that should 
reduce the number of operations needed by half).
- The variational lower bound calculation needs to be simplified -- right now
it's a little messy.
- Parts of the EM and variational class will be rewritten using Cython to provide
additional optimization / speed.
