# %%
from numba import cuda
import numpy as np
print(cuda)

@cuda.jit
def inner_prod_cuda(arr1, arr2, arr_out, L):
    x = cuda.grid(1)
    if x < L:
        arr_out[x] = arr1[x] * arr2[x]


def inner_prod(arr1, arr2):
    arr_out = np.zeros(arr1.shape[0])
    d_arr1 = cuda.to_device(arr1)
    d_arr2 = cuda.to_device(arr2)
    d_arr_out = cuda.to_device(arr_out)
    shape = arr1.shape[0]

    threads_per_block = 64
    blocks_per_grid = (shape // threads_per_block) + 1
    inner_prod_cuda[blocks_per_grid, threads_per_block](d_arr1, d_arr2, d_arr_out, shape)
    arr_out = d_arr_out.copy_to_host()
    return np.sum(arr_out)

# %%
# %%
