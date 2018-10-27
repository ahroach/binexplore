import math
import numpy
import numpy.fft
import matplotlib.pyplot as pyplot
import binexplore.binstats as binstats

def _cast_uint8_ndarray(a):
    """ Attempt to cast everything as a numpy.ndarray of type uint8 """
    if isinstance(a, numpy.ndarray):
        if a.dtype == 'uint8':
            return a
        else:
            return numpy.array(a, dtype=numpy.uint8)
    elif isinstance(a, bytes):
        return numpy.array(bytearray(a), dtype=numpy.uint8)
    else:
        return numpy.array(a, dtype=numpy.uint8)

def byte_freq_progression(a, **kwargs):
    """ Plot progression of byte frequencies through the array

    Produces a heatmap of byte frequencies as a function of block number
    through the provided array.

    Parameters
    ----------
    a : array_like or bytestring
        Contains data to plot
    bs : int, optional
        blocksize over which to calculate frequencies
    ax : matploitlib.axes._subplots.AxesSubplot instance
        Axes on which to show results

    Other Parameters
    ----------------
    **kwargs : `~matplotlib.artist.Artist` properties, optional
        Additional kwargs will be passed on to the imshow() function

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot instance
        Axes on which the results were shown
    """

    a = _cast_uint8_ndarray(a)

    if 'bs' in kwargs:
        bs = kwargs.pop('bs')
    else:
        bs = a.size//0x100
        if bs < 0x200:
            bs = 0x200

    blocks = a.size//bs + (0 if a.size % bs == 0 else 1)
    bf = numpy.zeros((blocks, 256))
    for i in range(blocks):
        bf[i,:] = binstats.byte_count_frac(a.flatten()[i*bs:(i+1)*bs])

    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
    else:
        fig = pyplot.figure()
        ax = fig.add_subplot(1,1,1)
        # Black background for any empty cells from log_2 calculation
        ax.set_facecolor('black')
        ax.set_xlabel("Byte value")
        ax.set_ylabel("Block (size = %i bytes)" % bs)

    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'hot'
    if 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'nearest'

    ax.imshow(numpy.log2(bf), **kwargs)

    return ax

def byte_values(a, **kwargs):
    """ Plot byte values in the provided array

    Show an image produced by assigning each byte of the provided array
    to a color

    Parameters
    ----------
    a : array_like or bytestring
        Contains data to plot
    width : int, optional
        X-dimension of the array; used to resize array to 2-D
    ax : matploitlib.axes._subplots.AxesSubplot instance
        Axes on which to show results

    Other Parameters
    ----------------
    **kwargs : `~matplotlib.artist.Artist` properties, optional
        Additional kwargs will be passed on to the imshow() function

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot instance
        Axes on which the results were shown
    """

    a = _cast_uint8_ndarray(a)

    if 'width' in kwargs:
        width = kwargs.pop('ax')
    else:
        width = int(math.sqrt(a.size))

    remainder = a.size % width
    dim1 = a.size//width + (1 if remainder != 0 else 0)
    data = numpy.resize(a, (dim1, width))
    # numpy.resize() repeats the array contents if the new array is larger
    # than the original. I would rather have null bytes.
    if remainder != 0:
        data[-1,-(width-remainder):] = 0

    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
    else:
        fig = pyplot.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_ylabel("Block (size = %i bytes)" % width)
        ax.set_xlabel("Index in block")

    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'hot'
    if 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'nearest'
    if 'vmin' not in kwargs:
        kwargs['vmin'] = 0
    if 'vmax' not in kwargs:
        kwargs['vmax'] = 255

    ax.imshow(data, **kwargs)

    return ax

def digraphs(a, **kwargs):
    """ Plot digraph counts of the provided array

    Show a heatmap corresponding to the count of digraphs in the provided
    array. The heatmap values are chosen from the colormap the log base 2 of
    the digraph counts.

    Parameters
    ----------
    a : array_like or bytestring
        Contains data to plot
    ax : matploitlib.axes._subplots.AxesSubplot instance
        Axes on which to show results

    Other Parameters
    ----------------
    **kwargs : `~matplotlib.artist.Artist` properties, optional
        Additional kwargs will be passed on to the imshow() function

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot instance
        Axes on which the results were shown

    See also
    --------
    binstats.digraph_count()
    """

    dg = binstats.digraph_count(a)

    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
    else:
        fig = pyplot.figure()
        ax = fig.add_subplot(1,1,1)
        # Black background for any empty cells from log_2 calculation
        ax.set_facecolor('black')
        ax.set_ylabel("Byte 1")
        ax.set_xlabel("Byte 2")

    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'hot'
    if 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'nearest'

    ax.imshow(numpy.log2(dg), **kwargs)

    return ax

def entropy(a, **kwargs):
    """ Plot entropy as a function of block number
    
    Parameters
    ----------
    a : array_like or bytestring
        Contains data to plot
    bs : int, optional
        blocksize over which to calculate entropy
    ax : matploitlib.axes._subplots.AxesSubplot instance
        Axes on which to show results

    Other Parameters
    ----------------
    **kwargs : `matplotlib.pyplot.Line2D` properties, optional
        Additional kwargs will be passed on to the plot() function

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot instance
        Axes on which the results were shown
    
    See also
    --------
    binstats.entropy() : Calculate Shannon entropy

    Notes
    -----
    Blocksizes that are too small (less than about 0x100 bytes or so) will
    be subject to significant amounts of random noise, possibly leading to
    false estimates of the variation of the entropy. There are 0x100 possible
    symbols in the set, so a blocksize that is at least a few hundred bytes
    should be chosen for best results.
    """

    a = _cast_uint8_ndarray(a)

    if 'bs' in kwargs:
        bs = kwargs.pop('bs')
    else:
        bs = a.size//1024
        if bs < 0x200:
            bs = 0x200

    blocks = a.size//bs + (0 if a.size % bs == 0 else 1)
    entropy = numpy.zeros(blocks)
    for i in range(blocks):
        entropy[i] = binstats.entropy(a.flatten()[i*bs:(i+1)*bs])

    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
    else:
        fig = pyplot.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel("Block (size=%i bytes)" % bs)
        ax.set_ylabel("Shannon entropy")
        ax.set_ylim(bottom=-0.1, top=8.1)

    ax.plot(entropy, **kwargs)

    return ax

def repetition_power(a, **kwargs):
    """ Plot power spectrum as a function of repetition period
    
    Parameters
    ----------
    a : array_like or bytestring
        Contains data to plot
    ax : matploitlib.axes._subplots.AxesSubplot instance
        Axes on which to show results

    Other Parameters
    ----------------
    **kwargs : `matplotlib.pyplot.Line2D` properties, optional
        Additional kwargs will be passed on to the plot() function

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot instance
        Axes on which the results were shown
    
    Notes
    -----
    Results are returned on a log scale for ease of visualization. The user may
    decide that a linear scale is more appropriate.

    The zero-frequency (infinite repetition period) componenet is removed,
    since it doesn't really fit on the x scale, and the fact that all values
    are unsigned characters means that the magnitude is very large compared to
    the other components.
    """

    a = _cast_uint8_ndarray(a)

    fft = numpy.fft.fft(a.flatten())
    ps = numpy.abs(fft)**2
 
    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
    else:
        fig = pyplot.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel("Period")
        ax.set_ylabel("Power")
        ax.set_xscale('log')

    # We'll just use the positive frequency space, also excluding the zero
    # frequency component.
    # Frequencies are spaced 1/n apart; Periods are the inverse of these
    # frequencies.
    mid = a.size//2
    periods = [a.size/i for i in range(1,mid)]

    ax.plot(periods, ps[1:mid], **kwargs)

    return ax

