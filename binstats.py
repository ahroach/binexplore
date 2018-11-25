import numpy
import math

def _cast_uint8_ndarray(a):
    """ Attempt to cast everything as a numpy.ndarray of type uint8

    Parameters
    ----------
    a : array_like or bytestring
        Object containing data

    Returns
    -------
    result : numpy.ndarray with dtype numpy.uint8
        Re-cast result
    """

    if isinstance(a, numpy.ndarray):
        if a.dtype == 'uint8':
            return a
        else:
            return numpy.array(a, dtype=numpy.uint8)
    elif isinstance(a, bytes):
        return numpy.array(bytearray(a), dtype=numpy.uint8)
    else:
        return numpy.array(a, dtype=numpy.uint8)

def byte_count(a, frac=False):
    """ Count the occurrence of each byte value

    Parameters
    ----------
    a : array_like or bytestring
        The array for which to count byte occurrences
    frac : Boolean
        Return results as a floating point fraction array

    Returns
    -------
    result : numpy.ndarray with size 256
        Number of byte occurrences counted in a. Returned dtype
        is numpy.uint64 for frac=False, and floating point for
        frac=True.
    """

    a = _cast_uint8_ndarray(a)

    count = numpy.zeros((256), dtype=numpy.uint64)
    for i in range(256):
        count[i] = numpy.count_nonzero(a == i)
    if frac:
        return 1.0*count/count.sum(axis=None)
    else:
        return count

def diff_bit(a, b):
    """ Calculate the number of bit differences between the two arrays

    Arrays must be the same size

    Parameters
    ----------
    a : array_like or bytestring
        First input object
    b : array_like or bytestring
        Second input object

    Returns
    -------
    diff : int
        Number of bits different between a and b
    """

    a = _cast_uint8_ndarray(a)
    b = _cast_uint8_ndarray(b)

    return numpy.count_nonzero(numpy.unpackbits(numpy.bitwise_xor(a, b)))

def diff_byte(a, b):
    """ Calculate the number of byte differences between the two arrays

    Arrays must be the same size

    Parameters
    ----------
    a : array_like or bytestring
        First input object
    b : array_like or bytestring
        Second input object

    Returns
    -------
    diff : int
        Number of bytes different between a and b
    """

    a = _cast_uint8_ndarray(a)
    b = _cast_uint8_ndarray(b)

    return numpy.count_nonzero(numpy.bitwise_xor(a, b))

def digraph_count(a, frac=False):
    """ Count the occurrence of each pair of bytes

    Parameters
    ----------
    a : array_like or bytestring
        The array for which to count byte pair occurrences
    frac : Boolean
        Return results as a floating point fraction array

    Returns
    -------
    result : numpy.ndarray with size 256x256
        Number of byte pair occurrences counted in a. Returned dtype is
        numpy.uint64 for frac=False, and floating point for frac=True.
    """

    a = _cast_uint8_ndarray(a).flatten()
    result = numpy.zeros((256, 256), dtype=numpy.uint64)

    for i in range(a.size-1):
        result[a[i], a[i+1]] += 1

    if frac:
        return 1.0*result/result.sum(axis=None)
    else:
        return result

def entropy(a):
    """ Calculate the Shannon entropy

    Parameters
    ----------
    a : array_like or bytestring
        The array over which to calculate entropy

    Returns
    -------
    h : float
        Shannon entropy
    """

    a = _cast_uint8_ndarray(a)

    h = 0.0
    for i in range(256):
        count = numpy.count_nonzero(a == i)
        if count != 0:
            p = 1.0*count/a.size
            h -= p * math.log2(p)

    return h

def repeating_xor(src, mask):
    """ Apply repeating xor mask to array

    Arrays greater than 1-D will have the mask applied in the standard C
    order produced by numpy flatten() method (row-major order)

    Parameters
    ----------
    src : array_like or bytestring
        The source data to which the xor mask will be applied
    mask : array_like or bytestring
        Xor mask

    Returns
    -------
    xor_result : numpy.ndarray with dtype uint8
        Result of repeated xor operation, with the same shape as src
    """
    
    # Cast src and mask to binarray
    src = _cast_uint8_ndarray(src)
    mask = _cast_uint8_ndarray(mask)

    result = src.flatten()
    
    # Apply the entire mask where we can
    for i in range(result.size//mask.size):
        s = i*mask.size
        e = (i+1)*mask.size
        result[s:e] = numpy.bitwise_xor(result[s:e], mask)

    # Apply any remainder
    remain = result.size % mask.size
    if remain != 0:
        result[-remain:] = numpy.bitwise_xor(result[-remain:], mask[:remain])

    return result.reshape(src.shape)

def trigraph_count(a, frac=False):
    """ Count the occurrence of each triplet of bytes

    Parameters
    ----------
    a : array_like or bytestring
        The array for which to count byte triplet occurrences
    frac : Boolean
        Return results as a floating point fraction array

    Returns
    -------
    result : numpy.ndarray with size 256x256x256
        Number of byte triplet occurrences counted in a. Returned dtype is
        numpy.uint64 for frac=False, and floating point for frac=True.
    """

    a = _cast_uint8_ndarray(a).flatten()
    result = numpy.zeros((256, 256, 256), dtype=numpy.uint64)

    for i in range(a.size-2):
        result[a[i], a[i+1], a[i+2]] += 1

    if frac:
        return 1.0*result/result.sum(axis=None)
    else:
        return result

