import numpy

import cupy
from cupy import cuda
from cupy.cuda import device
from cupy.linalg import _util as util
from cupy.cuda import cusparse

def batched_gtsv(dl, d, du, B, algo='cyclic_reduction'):
    """Solves multiple tridiagonal systems (This is a bang method for B.)

    Args:
        dl, d, du (cupy.ndarray): Lower, main and upper diagonal vectors with last-dim sizes of N-1, N and N-1, repsectively.
            Only two dimensional inputs are supported currently.
            The first dim is the batch dim.
        B (cupy.ndarray): Right-hand side vectors
            The first dim is the batch dim and the second dim is N.
        algo (str): algorithm, choose one from four algorithms; cyclic_reduction, cuThomas, LU_w_pivoting and QR.
            cuThomas is numerically unstable, and LU_w_pivoting is the LU algorithm with pivoting.
    """
    if algo not in ["cyclic_reduction", "cuThomas", "LU_w_pivoting", "QR"]:
        raise ValueError(f"Unknown algorithm [{algo}]")

    util._assert_cupy_array(dl)
    util._assert_cupy_array(d)
    util._assert_cupy_array(du)
    util._assert_cupy_array(B)
    if dl.ndim != 2 or d.ndim != 2 or du.ndim != 2 or B.ndim!= 2:
        raise ValueError('dl, d, du and B must be 2-d arrays')
    
    batchsize = d.shape[0]
    if batchsize != dl.shape[0] or batchsize != du.shape[0] or batchsize != B.shape[0]:
        raise ValueError('The first dims of dl, du and B must match that of d.')
    N = d.shape[1] # the size of the linear system
    if dl.shape[1] != N-1 or du.shape[1] != N-1 or B.shape[1] != N:
        raise ValueError('The second dims of dl, du and B must match the second dim of d.')

    # the first element must be zero of dl
    padded_dl = cupy.ascontiguousarray(cupy.pad(dl, ((0,0),(1,0)), mode='constant', constant_values=0.0))
    # the last element must be zero of du
    padded_du = cupy.ascontiguousarray(cupy.pad(du, ((0,0),(0,1)), mode='constant', constant_values=0.0))
    # contiguous
    d = cupy.ascontiguousarray(d)
    B = cupy.ascontiguousarray(B)
    
    # Cast to float32 or float64
    if d.dtype == 'f' or d.dtype == 'd':
        dtype = d.dtype
    else:
        dtype = numpy.find_common_type((d.dtype, 'f'), ())
    
    handle = device.get_cusparse_handle()
    
    if dtype == 'f':
        if algo == "cyclic_reduction":
            gtsv2 = cusparse.sgtsv2StridedBatch
            get_buffer_size = cusparse.sgtsv2StridedBatch_bufferSizeExt
            #
            buffer_size = numpy.empty(1,numpy.int32)
            get_buffer_size(handle, N, padded_dl.data.ptr, d.data.ptr, padded_du.data.ptr, B.data.ptr, batchsize, N, buffer_size.ctypes.data)
            buffer_size = int(buffer_size)
            buffer = cupy.zeros((buffer_size,), dtype=cupy.uint8)
            gtsv2(
                handle, N, padded_dl.data.ptr, d.data.ptr, padded_du.data.ptr, B.data.ptr, batchsize, N, buffer.data.ptr
            )
        else:
            raise NotImplementedError
            if algo == "cuThomas":
                algo_num = 0
            elif algo ==  "LU_w_pivoting":
                algo_num = 1
            elif algo == "QR":
                algo_num = 2
            else:
                raise ValueError
            gtsv2 = cusparse.sgtsvInterleavedBatch
            get_buffer_size = cusparse.sgtsvInterleavedBatch_bufferSizeExt
            #
            buffer_size = get_buffer_size(handle, algo_num, N, padded_dl.data.ptr, d.data.ptr, padded_du.data.ptr, B.data.ptr, batchsize)
            buffer = cupy.zeros((buffer_size,), dtype=cupy.uint8)
            gtsv2(
                handle, algo_num, N, padded_dl.data.ptr, d.data.ptr, padded_du.data.ptr, B.data.ptr, batchsize, buffer.data.ptr
            )
    else:
        raise NotImplementedError
    return B
