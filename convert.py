import mxnet as mx
import time
import scipy.sparse as spsp
import numpy as np

csr_mx = mx.nd.load('/home/ubuntu/dgl/examples/mxnet/sse/5_5_csr.nd.patch')[0]
csr_np = csr_mx.asscipy()
# sort by nnz
nnz_per_row = csr_np.getnnz(axis=1)
sorted_nnz = np.flip(np.sort(nnz_per_row, kind='mergesort'), axis=0);
original_idx = np.flip(np.argsort(nnz_per_row, kind='mergesort'), axis=0)
start = csr_np.indptr[original_idx]
end = start + sorted_nnz
end2 = start + nnz_per_row[original_idx]
# sanity check
assert np.all(end2 == end)

out_indices = np.zeros_like(csr_np.indices)
# mapping to convert node id back to its original value
reverse_lookup = np.zeros_like(csr_np.indices)
for i in range(len(original_idx)):
    reverse_lookup[original_idx[i]] = i

offset = 0
count = 0
t0 = time.time()
for s, e in zip(start, end):
    idx = csr_np.indices[s:e]
    # convert col idx based on the mapping, and sort it (as is expected by mxnet..)
    out_indices[offset:offset+e-s] = np.sort(reverse_lookup[idx], axis=0)
    offset += e - s
    count += 1
    if count % 100000 == 0:
        print(count, time.time() - t0)
assert offset == len(out_indices)

t1 = time.time()

# get indptr
out_indptr = np.zeros((csr_np.indptr.size,))
out_indptr[1:] = sorted_nnz
out_indptr = np.cumsum(out_indptr)

# compose scipy csr matrix
final_indices = out_indices
final_data = csr_np.data
final_indptr = out_indptr

out_csr = spsp.csr_matrix((final_data, final_indices, final_indptr), shape=csr_np.shape)
np.save(open('5_5_csr.np.sorted.npy', 'w'), out_csr)

# compose mxnet csr matrix
final_indices_mx = mx.nd.array(out_csr.indices, dtype='int64')
final_data_mx = csr_mx.data
final_indptr_mx = mx.nd.array(final_indptr, dtype='int64')

out_csr_mx = mx.nd.sparse.csr_matrix((final_data_mx, final_indices_mx, final_indptr_mx),
                                     shape=csr_mx.shape, dtype=csr_mx.dtype)

mx.nd.save('5_5_csr.nd.sorted', out_csr_mx)
print(t1 - t0)
print(csr_np)
