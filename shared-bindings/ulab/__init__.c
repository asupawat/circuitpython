/*
 * This file is part of the micropython-ulab project,
 *
 * https://github.com/v923z/micropython-ulab
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2019-2020 Zoltán Vörös
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "py/runtime.h"
#include "py/binary.h"
#include "py/obj.h"
#include "py/objarray.h"
#include "py/objproperty.h"

#include "extmod/ulab/code/compat.h"
#include "extmod/ulab/code/ndarray.h"
#include "extmod/ulab/code/linalg.h"
#include "extmod/ulab/code/vectorise.h"
#include "extmod/ulab/code/poly.h"
#include "extmod/ulab/code/fft.h"
#include "extmod/ulab/code/filter.h"
#include "extmod/ulab/code/numerical.h"

//|
//| :mod:`ulab` --- Manipulate numeric data similar to numpy
//| ========================================================
//|
//| .. module:: ulab
//|    :synopsis: Manipulate numeric data similar to numpy
//|
//| `ulab` is a numpy-like module for micropython, meant to simplify and
//| speed up common mathematical operations on arrays. The primary goal was to
//| implement a small subset of numpy that might be useful in the context of a
//| microcontroller. This means low-level data processing of linear (array) and
//| two-dimensional (matrix) data.
//|
//| `ulab` is adapted from micropython-ulab, and the original project's
//| documentation can be found at
//| https://micropython-ulab.readthedocs.io/en/latest/
//|
//| `ulab` is modeled after numpy, and aims to be a compatible subset where
//| possible.  Numpy's documentation can be found at
//| https://docs.scipy.org/doc/numpy/index.html
//|
//| .. contents::
//|

//| .. attribute:: __version__
//|
//| The closest corresponding version of micropython-ulab
//|
STATIC MP_DEFINE_STR_OBJ(ulab_version_obj, "0.32.0");

//| ulab.array -- 1- and 2- dimensional array
//| --------------------------------------------------
//| .. class:: ulab.array(values, *, dtype=float)
//|
//|   :param sequence values: Sequence giving the initial content of the array.
//|   :param dtype: The type of array values, ``int8``, ``uint8``, ``int16``, ``uint16``, or ``float``
//|
//|   The `values` sequence can either be a sequence of numbers (in which case a
//|   1-dimensional array is created), or a sequence where each subsequence has
//|   the same length (in which case a 2-dimensional array is created).
//|
//|   In many cases, it is more convenient to create an array from a function
//|   like `zeros` or `linspace`.
//|
//|   `ulab.array` implements the buffer protocol, so it can be used in many
//|   places an `array.array` can be used.
//|
//|    .. attribute:: shape
//|
//|       The size of the array, a tuple of length 1 or 2
//|
//|    .. attribute:: size
//|
//|       The number of elements in the array
//|
//|    .. attribute:: itemsize
//|
//|       The number of elements in the array
//|
//|    .. method:: flatten(*, order='C')
//|
//|       :param order: Whether to flatten by rows ('C') or columns ('F')
//|
//|       Returns a new `ulab.array` object which is always 1 dimensional.
//|       If order is 'C' (the default", then the data is ordered in rows;
//|       If it is 'F', then the data is ordered in columns.  "C" and "F" refer
//|       to the typical storage organization of the C and Fortran languages.
//|
//|    .. method:: sort(*, axis=1)
//|
//|       :param axis: Whether to sort elements within rows (0), columns (1), or elements (None)
//|
//|    .. method:: transpose()
//|
//|       Swap the rows and columns of a 2-dimensional array
//|
//|    .. method:: __add__()
//|
//|       Adds corresponding elements of the two arrays, or adds a number to all
//|       elements of the array.  A number must be on the right hand side.  If
//|       both arguments are arrays, their sizes must match.
//|
//|    .. method:: __sub__()
//|
//|       Subtracts corresponding elements of the two arrays, or subtracts a
//|       number from all elements of the array.  A number must be on the right
//|       hand side.  If both arguments are arrays, their sizes must match.
//|
//|    .. method:: __mul__()
//|
//|       Multiplies corresponding elements of the two arrays, or multiplies
//|       all elements of the array by a number.  A number must be on the right
//|       hand side.  If both arguments are arrays, their sizes must match.
//|
//|    .. method:: __div__()
//|
//|       Multiplies corresponding elements of the two arrays, or divides
//|       all elements of the array by a number.  A number must be on the right
//|       hand side.  If both arguments are arrays, their sizes must match.
//|
//|    .. method:: __getitem__()
//|
//|       Retrieve an element of the array.
//|
//|    .. method:: __setitem__()
//|
//|       Set an element of the array.
//|


//| Array type codes
//| ----------------
//| .. attribute:: int8
//|
//|    Type code for signed integers in the range -128 .. 127 inclusive, like the 'b' typecode of `array.array`
//|
//| .. attribute:: int16
//|
//|    Type code for signed integers in the range -32768 .. 32767 inclusive, like the 'h' typecode of `array.array`
//|
//| .. attribute:: float
//|
//|    Type code for floating point values, like the 'f' typecode of `array.array`
//|
//| .. attribute:: uint8
//|
//|    Type code for unsigned integers in the range 0 .. 255 inclusive, like the 'H' typecode of `array.array`
//|
//| .. attribute:: uint8
//|
//|    Type code for unsigned integers in the range 0 .. 65535 inclusive, like the 'h' typecode of `array.array`
//|

//| Array defining functions
//| ------------------------
//|
//| .. method:: eye(size, *, dtype=float)
//|
//|    :param int: size - The number of rows and colums in the matrix
//|
//|    Returns a square matrix with all the diagonal elements set to 1 and all
//|    other elements set to 0
//|
//| .. method:: linspace(start, stop, *, dtype=float, num=50, endpoint=True)
//|
//|    .. param: start
//|
//|       First value in the array
//|
//|    .. param: stop
//|
//|       Final value in the array
//|
//|    .. param int: num
//|
//|       Count of values in the array
//|
//|    .. param: dtype
//|
//|       Type of values in the array
//|
//|    .. param bool: endpoint
//|
//|       Whether the ``stop`` value is included.  Note that even when
//|       endpoint=True, the exact ``stop`` value may not be included due to the
//|       inaccuracy of floating point arithmetic.
//|
//|    Return a new 1-D array with ``num`` elements ranging from ``start`` to ``stop`` linearly.
//|
//| .. method:: ones(shape, *, dtype=float)
//|
//|    .. param: shape
//|       Shape of the array, either an integer (for a 1-D array) or a tuple of 2 integers (for a 2-D array)
//|
//|    .. param: dtype
//|       Type of values in the array
//|
//|    Return a new array of the given shape with all elements set to 1.
//|
//| .. method:: zeros
//|
//|    .. param: shape
//|       Shape of the array, either an integer (for a 1-D array) or a tuple of 2 integers (for a 2-D array)
//|
//|    .. param: dtype
//|       Type of values in the array
//|
//|    Return a new array of the given shape with all elements set to 0.
//|
//| .. method:: eye(size, *, dtype=float)
//|
//|    Return a new square array of size, with the diagonal elements set to 1
//|    and the other elements set to 0.
//|

//| Element-by-element functions
//| ----------------------------
//|
//| These functions can operate on numbers, 1-D arrays, or 2-D arrays by
//| applying the function to every element in the array.  This is typically
//| much more efficient than expressing the same operation as a Python loop.
//|
//| .. method:: acos
//|
//|    Computes the inverse cosine function
//|
//| .. method:: acosh
//|
//|    Computes the inverse hyperbolic cosine function
//|
//| .. method:: asin
//|
//|    Computes the inverse sine function
//|
//| .. method:: asinh
//|
//|    Computes the inverse hyperbolic sine function
//|
//| .. method:: atan
//|
//|    Computes the inverse tangent function
//|
//| .. method:: atanh
//|
//|    Computes the inverse hyperbolic tangent function
//|
//| .. method:: ceil
//|
//|    Rounds numbers up to the next whole number
//|
//| .. method:: cos
//|
//|    Computes the cosine function
//|
//  .. method:: cosh
// 
//     Computes the hyperbolic cosine function
// 
//| .. method:: erf
//|
//|    Computes the error function, which has applications in statistics
//|
//| .. method:: erfc
//|
//|    Computes the complementary error function, which has applications in statistics
//|
//| .. method:: exp
//|
//|    Computes the exponent function.
//|
//| .. method:: expm1
//|
//|    Computes $e^x-1$.  In certain applications, using this function preserves nuemric accuracy better than the `exp` function.
//|
//| .. method:: floor
//|
//|    Rounds numbers up to the next whole number
//|
//| .. method:: gamma
//|
//|    Computes the gamma function
//|
//| .. method:: lgamma
//|
//|    Computes the natural log of the gamma function
//|
//| .. method:: log
//|
//|    Computes the natural log
//|
//| .. method:: log10
//|
//|    Computes the log base 10
//|
//| .. method:: log2
//|
//|    Computes the log base 2
//|
//| .. method:: sin
//|
//|    Computes the sine
//|
//| .. method:: sinh
//|
//|    Computes the hyperbolic sine
//|
//| .. method:: sqrt
//|
//|    Computes the square root
//|
//| .. method:: tan
//|
//|    Computes the tangent
//|
//| .. method:: tanh
//|
//|    Computes the hyperbolic tangent
//|

//| Matrix Functions
//| ----------------
//| .. method:: dot(m1, m2)
//|
//|    :param ~ulab.array m1: a matrix
//|    :param ~ulab.array m2: a matrix
//|
//|    Computes the matrix product of two matrices
//|
//|    **WARNING:** Unlike ``numpy``, this function cannot be used to compute the dot product of two vectors
//|
//| .. method:: inv(m)
//|
//|    :param ~ulab.array m: a square matrix
//|    :return: The inverse of the matrix, if it exists
//|    :raises ValueError: if the matrix is not invertible
//|
//|    Computes the inverse of a square matrix
//|
//| .. method:: eig(m)
//|
//|    :param m: a square matrix
//|    :return tuple (eigenvectors, eigenvaues):
//|
//|    Computes the eigenvalues and eigenvectors of a square matrix
//|
//| .. method:: det
//|
//|    :param: m, a square matrix
//|    :return float: The determinant of the matrix
//|
//|    Computes the eigenvalues and eigenvectors of a square matrix
//|

//| Filtering functions
//| -------------------
//| .. method:: convolve(r, c=None)
//|
//|    :param ulab.array a:
//|    :param ulab.array v:
//|
//|    Returns the discrete, linear convolution of two one-dimensional sequences.
//|    The result is always an array of float.  Only the ``full`` mode is supported,
//|    and the ``mode`` named parameter of numpy is not accepted. Note that all other
//|    modes can be had by slicing a ``full`` result.
//|

//| Frequency-domain functions
//| --------------------------
//|
//| .. method:: fft(r, c=None)
//|
//|    :param ulab.array r: A 1-dimension array of values whose size is a power of 2
//|    :param ulab.array c: An optional 1-dimension array of values whose size is a power of 2, giving the complex part of the value
//|    :return tuple (r, c): The real and complex parts of the FFT
//|
//|    Perform a Fast Fourier Transform from the time domain into the frequency domain
//|
//| .. method:: ifft(r, c=None)
//|
//|    :param ulab.array r: A 1-dimension array of values whose size is a power of 2
//|    :param ulab.array c: An optional 1-dimension array of values whose size is a power of 2, giving the complex part of the value
//|    :return tuple (r, c): The real and complex parts of the inverse FFT
//|
//|    Perform an Inverse Fast Fourier Transform from the frequeny domain into the time domain
//|
//| .. method:: spectrum(r):
//|
//|    :param ulab.array r: A 1-dimension array of values whose size is a power of 2
//|
//|    Computes the spectrum of the input signal.  This is the absolute value of the (complex-valued) fft of the signal.
//|

//| Statistical functions
//| ---------------------
//|
//| Most of these functions take an "axis" argument, which indicates whether to
//| operate over the flattened array (None), rows (0), or columns (1).
//|
//| .. method:: argmax(array, *, axis=None)
//|
//|    Return the index of the maximum element of the 1D array, as an array.
//|
//| .. method:: argmin(array, *, axis=None)
//|
//|    Return the index of the minimum element of the 1D array, as an array with 1 element
//|
//| .. method:: argsort(array, *, axis=None)
//|
//|    Returns an array which gives indices into the input array from least to greatest.
//|
//| .. method:: max(array, *, axis=None)
//|
//|    Return the maximum element of the 1D array, as an array with 1 element
//|
//| .. method:: mean(array, *, axis=None)
//|
//|    Return the mean element of the 1D array, as a number if axis is None, otherwise as an array.
//|
//| .. method:: min(array, *, axis=None)
//|
//|    Return the minimum element of the 1D array, as an array with 1 element
//|
//| .. method:: std(array, *, axis=None)
//|
//|    Return the standard deviation of the array, as a number if axis is None, otherwise as an array.
//|
//| .. method:: sum(array, *, axis=None)
//|
//|    Return the sum of the array, as a number if axis is None, otherwise as an array.
//|
//| .. method:: diff(array, *, axis=1)
//|
//|    Return the numerical derivative of successive elements of the array, as
//|    an array.  axis=None is not supported.
//|
//| .. method:: size(array)
//|
//|    Return the total number of elements in the array, as an integer.
//|

//| Polynomial functions
//| --------------------
//|
//| .. method:: polyfit([x, ] y, degree)
//|
//|    Return a polynomial of given degree that approximates the function
//|    f(x)=y.  If x is not supplied, it is the range(len(y)).
//|
//| .. method:: polyval(p, x)
//|
//|    Evaluate the polynomial p at the points x.  x must be an array.
//|

//| Reordering functions
//| --------------------
//|
//| .. method:: sort(array, *, axis=0)
//|
//|    Sort the array along the given axis, or along all axes if axis is None.
//|    The array is modified in place.
//|
//| .. method:: flip(array, *, axis=None)
//|
//|    Returns a new array that reverses the order of the elements along the
//|    given axis, or along all axes if axis is None.
//|
//| .. method:: roll(array, distance, *, axis=None)
//|
//|    Shift the content of a vector by the positions given as the second
//|    argument. If the ``axis`` keyword is supplied, the shift is applied to
//|    the given axis.  The array is modified in place.
//|

MP_DEFINE_CONST_FUN_OBJ_1(ndarray_get_shape_obj, ndarray_shape);
MP_DEFINE_CONST_FUN_OBJ_1(ndarray_get_size_obj, ndarray_size);
extern mp_obj_t ndarray_itemsize(mp_obj_t);
MP_DEFINE_CONST_FUN_OBJ_1(ndarray_get_itemsize_obj, ndarray_itemsize);
MP_DEFINE_CONST_FUN_OBJ_KW(ndarray_flatten_obj, 1, ndarray_flatten);

MP_DECLARE_CONST_FUN_OBJ_1(linalg_transpose_obj);
MP_DECLARE_CONST_FUN_OBJ_2(linalg_reshape_obj);
MP_DECLARE_CONST_FUN_OBJ_KW(linalg_size_obj);
MP_DECLARE_CONST_FUN_OBJ_1(linalg_inv_obj);
MP_DECLARE_CONST_FUN_OBJ_2(linalg_dot_obj);
MP_DECLARE_CONST_FUN_OBJ_KW(linalg_zeros_obj);
MP_DECLARE_CONST_FUN_OBJ_KW(linalg_ones_obj);
MP_DECLARE_CONST_FUN_OBJ_KW(linalg_eye_obj);
MP_DECLARE_CONST_FUN_OBJ_1(linalg_det_obj);
MP_DECLARE_CONST_FUN_OBJ_1(linalg_eig_obj);

MP_DECLARE_CONST_FUN_OBJ_1(vectorise_acos_obj);
MP_DECLARE_CONST_FUN_OBJ_1(vectorise_acosh_obj);
MP_DECLARE_CONST_FUN_OBJ_1(vectorise_asin_obj);
MP_DECLARE_CONST_FUN_OBJ_1(vectorise_asinh_obj);
MP_DECLARE_CONST_FUN_OBJ_1(vectorise_atan_obj);
MP_DECLARE_CONST_FUN_OBJ_1(vectorise_atanh_obj);
MP_DECLARE_CONST_FUN_OBJ_1(vectorise_ceil_obj);
MP_DECLARE_CONST_FUN_OBJ_1(vectorise_cos_obj);
// MP_DECLARE_CONST_FUN_OBJ_1(vectorise_cosh_obj);
MP_DECLARE_CONST_FUN_OBJ_1(vectorise_erf_obj);
MP_DECLARE_CONST_FUN_OBJ_1(vectorise_erfc_obj);
MP_DECLARE_CONST_FUN_OBJ_1(vectorise_exp_obj);
MP_DECLARE_CONST_FUN_OBJ_1(vectorise_expm1_obj);
MP_DECLARE_CONST_FUN_OBJ_1(vectorise_floor_obj);
MP_DECLARE_CONST_FUN_OBJ_1(vectorise_gamma_obj);
MP_DECLARE_CONST_FUN_OBJ_1(vectorise_lgamma_obj);
MP_DECLARE_CONST_FUN_OBJ_1(vectorise_log_obj);
MP_DECLARE_CONST_FUN_OBJ_1(vectorise_log10_obj);
MP_DECLARE_CONST_FUN_OBJ_1(vectorise_log2_obj);
MP_DECLARE_CONST_FUN_OBJ_1(vectorise_sin_obj);
MP_DECLARE_CONST_FUN_OBJ_1(vectorise_sinh_obj);
MP_DECLARE_CONST_FUN_OBJ_1(vectorise_sqrt_obj);
MP_DECLARE_CONST_FUN_OBJ_1(vectorise_tan_obj);
MP_DECLARE_CONST_FUN_OBJ_1(vectorise_tanh_obj);


STATIC const mp_obj_property_t ndarray_shape_obj = {
    .base.type = &mp_type_property,
    .proxy = {(mp_obj_t)&ndarray_get_shape_obj,
              (mp_obj_t)&mp_const_none_obj,
              (mp_obj_t)&mp_const_none_obj},
};

STATIC const mp_obj_property_t ndarray_size_obj = {
    .base.type = &mp_type_property,
    .proxy = {(mp_obj_t)&ndarray_get_size_obj,
              (mp_obj_t)&mp_const_none_obj,
              (mp_obj_t)&mp_const_none_obj},
};

STATIC const mp_obj_property_t ndarray_itemsize_obj = {
    .base.type = &mp_type_property,
    .proxy = {(mp_obj_t)&ndarray_get_itemsize_obj,
              (mp_obj_t)&mp_const_none_obj,
              (mp_obj_t)&mp_const_none_obj},
};

STATIC const mp_rom_map_elem_t ulab_ndarray_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_shape), MP_ROM_PTR(&ndarray_shape_obj) },
    { MP_ROM_QSTR(MP_QSTR_size), MP_ROM_PTR(&ndarray_size_obj) },
    { MP_ROM_QSTR(MP_QSTR_itemsize), MP_ROM_PTR(&ndarray_itemsize_obj) },
    { MP_ROM_QSTR(MP_QSTR_flatten), MP_ROM_PTR(&ndarray_flatten_obj) },
    { MP_ROM_QSTR(MP_QSTR_transpose), MP_ROM_PTR(&linalg_transpose_obj) },
    { MP_ROM_QSTR(MP_QSTR_reshape), MP_ROM_PTR(&linalg_reshape_obj) },
    { MP_ROM_QSTR(MP_QSTR_sort), MP_ROM_PTR(&numerical_sort_inplace_obj) },
};

STATIC MP_DEFINE_CONST_DICT(ulab_ndarray_locals_dict, ulab_ndarray_locals_dict_table);

const mp_obj_type_t ulab_ndarray_type = {
    { &mp_type_type },
    .name = MP_QSTR_ndarray,
    .print = ndarray_print,
    .make_new = ndarray_make_new,
    .subscr = ndarray_subscr,
    .getiter = ndarray_getiter,
    .unary_op = ndarray_unary_op,
    .binary_op = ndarray_binary_op,
    .locals_dict = (mp_obj_dict_t*)&ulab_ndarray_locals_dict,
    // .attr = ndarray_attributes,
};

STATIC const mp_map_elem_t ulab_globals_table[] = {
    { MP_OBJ_NEW_QSTR(MP_QSTR___name__), MP_OBJ_NEW_QSTR(MP_QSTR_ulab) },
    { MP_ROM_QSTR(MP_QSTR___version__), MP_ROM_PTR(&ulab_version_obj) },
    { MP_OBJ_NEW_QSTR(MP_QSTR_array), (mp_obj_t)&ulab_ndarray_type },
    { MP_OBJ_NEW_QSTR(MP_QSTR_size), (mp_obj_t)&linalg_size_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_inv), (mp_obj_t)&linalg_inv_obj },
    { MP_ROM_QSTR(MP_QSTR_dot), (mp_obj_t)&linalg_dot_obj },
    { MP_ROM_QSTR(MP_QSTR_zeros), (mp_obj_t)&linalg_zeros_obj },
    { MP_ROM_QSTR(MP_QSTR_ones), (mp_obj_t)&linalg_ones_obj },
    { MP_ROM_QSTR(MP_QSTR_eye), (mp_obj_t)&linalg_eye_obj },
    { MP_ROM_QSTR(MP_QSTR_det), (mp_obj_t)&linalg_det_obj },
    { MP_ROM_QSTR(MP_QSTR_eig), (mp_obj_t)&linalg_eig_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_acos), (mp_obj_t)&vectorise_acos_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_acosh), (mp_obj_t)&vectorise_acosh_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_asin), (mp_obj_t)&vectorise_asin_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_asinh), (mp_obj_t)&vectorise_asinh_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_atan), (mp_obj_t)&vectorise_atan_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_atanh), (mp_obj_t)&vectorise_atanh_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_ceil), (mp_obj_t)&vectorise_ceil_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_cos), (mp_obj_t)&vectorise_cos_obj },
    // { MP_OBJ_NEW_QSTR(MP_QSTR_cosh), (mp_obj_t)&vectorise_cosh_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_erf), (mp_obj_t)&vectorise_erf_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_erfc), (mp_obj_t)&vectorise_erfc_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_exp), (mp_obj_t)&vectorise_exp_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_expm1), (mp_obj_t)&vectorise_expm1_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_floor), (mp_obj_t)&vectorise_floor_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_gamma), (mp_obj_t)&vectorise_gamma_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_lgamma), (mp_obj_t)&vectorise_lgamma_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_log), (mp_obj_t)&vectorise_log_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_log10), (mp_obj_t)&vectorise_log10_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_log2), (mp_obj_t)&vectorise_log2_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_sin), (mp_obj_t)&vectorise_sin_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_sinh), (mp_obj_t)&vectorise_sinh_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_sqrt), (mp_obj_t)&vectorise_sqrt_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_tan), (mp_obj_t)&vectorise_tan_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_tanh), (mp_obj_t)&vectorise_tanh_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_linspace), (mp_obj_t)&numerical_linspace_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_sum), (mp_obj_t)&numerical_sum_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_mean), (mp_obj_t)&numerical_mean_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_std), (mp_obj_t)&numerical_std_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_min), (mp_obj_t)&numerical_min_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_max), (mp_obj_t)&numerical_max_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_argmin), (mp_obj_t)&numerical_argmin_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_argmax), (mp_obj_t)&numerical_argmax_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_roll), (mp_obj_t)&numerical_roll_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_flip), (mp_obj_t)&numerical_flip_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_diff), (mp_obj_t)&numerical_diff_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_sort), (mp_obj_t)&numerical_sort_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_argsort), (mp_obj_t)&numerical_argsort_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_polyval), (mp_obj_t)&poly_polyval_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_polyfit), (mp_obj_t)&poly_polyfit_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_fft), (mp_obj_t)&fft_fft_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_ifft), (mp_obj_t)&fft_ifft_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_spectrum), (mp_obj_t)&fft_spectrum_obj },
    // filter functions
    { MP_OBJ_NEW_QSTR(MP_QSTR_convolve), (mp_obj_t)&filter_convolve_obj },
    // class constants
    { MP_ROM_QSTR(MP_QSTR_uint8), MP_ROM_INT(NDARRAY_UINT8) },
    { MP_ROM_QSTR(MP_QSTR_int8), MP_ROM_INT(NDARRAY_INT8) },
    { MP_ROM_QSTR(MP_QSTR_uint16), MP_ROM_INT(NDARRAY_UINT16) },
    { MP_ROM_QSTR(MP_QSTR_int16), MP_ROM_INT(NDARRAY_INT16) },
    { MP_ROM_QSTR(MP_QSTR_float), MP_ROM_INT(NDARRAY_FLOAT) },
};

STATIC MP_DEFINE_CONST_DICT (
    mp_module_ulab_globals,
    ulab_globals_table
);

const mp_obj_module_t ulab_module = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t*)&mp_module_ulab_globals,
};
