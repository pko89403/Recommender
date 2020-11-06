import numpy as np
from scipy import sparse

a = np.array([[1, 2, 0], [0, 0, 3], [1, 0, 4]])
b = np.array([[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8]]
		)
print("org a")
print(a)

sparse_a = sparse.csr_matrix(a)
sparse_b = sparse.csr_matrix(b)
print("sparse.csr_matrix(a)")
print(sparse_a)
print("sparse.csr_matrix(b)")
print(sparse_b)

identity_csr_a = sparse.identity(10, format='csr')
identity_csr_b = sparse.identity(10, format='csr')

print("sparse.identity(10, format='csr')")
print(identity_csr_a)



identity_dia_a = sparse.identity(10, format='dia')
print("sparse.identity(10, format='dia')")
print(identity_dia_a)

zero_csr_a = sparse.csr_matrix((sparse_a.shape[0], sparse_b.shape[1]), dtype=sparse_a.dtype)
print(zero_csr_a)

hstack_csr_a = sparse.hstack([sparse_a, zero_csr_a], format='csr')
print(hstack_csr_a)

zero_csr_b = sparse.csr_matrix((sparse_b.shape[0], sparse_a.shape[1]), dtype=sparse_b.dtype)
print(zero_csr_b)

hstack_csr_b = sparse.hstack([sparse_b, zero_csr_b], format='csr')
print(hstack_csr_b)

print(hstack_csr_a.todense())
print(hstack_csr_b.todense())
