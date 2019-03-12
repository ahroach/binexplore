import math
import numpy
import numpy.fft
import matplotlib.pyplot as pyplot
import matplotlib.ticker as ticker
import binexplore.binstats as binstats

def _hex_tick_str(x, pos):
    return "%x" % int(x)

_hex_tick_formatter = ticker.FuncFormatter(_hex_tick_str)

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
    log : Boolean, optional
        Show log of counts (defaults to False)

    Other Parameters
    ----------------
    **kwargs : `~matplotlib.artist.Artist` properties, optional
        Additional kwargs will be passed on to the imshow() function

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot instance
        Axes on which the results were shown
    """

    a = binstats._cast_uint8_ndarray(a)

    if 'bs' in kwargs:
        bs = kwargs.pop('bs')
    else:
        bs = a.size//0x100
        if bs < 0x200:
            bs = 0x200

    blocks = a.size//bs + (0 if a.size % bs == 0 else 1)
    bf = numpy.zeros((blocks, 256))
    for i in range(blocks):
        bf[i,:] = binstats.byte_count(a.flatten()[i*bs:(i+1)*bs], frac=True)

    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
    else:
        fig = pyplot.figure()
        ax = fig.add_subplot(1,1,1)
        # Black background for any empty cells from log_2 calculation
        ax.set_facecolor('black')
        ax.set_xlabel("Byte value [hex]")
        ax.set_ylabel("Block [size = %i (0x%x) bytes]" % (bs, bs))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0x10))
        ax.xaxis.set_major_formatter(_hex_tick_formatter)

    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'hot'
    if 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'nearest'

    log = kwargs.pop('log', False)
    if log:
        ax.imshow(numpy.log2(bf), **kwargs)
    else:
        ax.imshow(bf, **kwargs)

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

    a = binstats._cast_uint8_ndarray(a)

    if 'width' in kwargs:
        width = kwargs.pop('width')
    else:
        width = math.ceil(int(math.sqrt(a.size))/256)*256

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
        ax.set_ylabel("Block [size = %i bytes]" % width)
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
    log : Boolean, optional
        Show log of counts (defaults to False)

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
        ax.set_ylabel("Byte 1 [hex]")
        ax.set_xlabel("Byte 2 [hex]")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0x10))
        ax.xaxis.set_major_formatter(_hex_tick_formatter)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0x10))
        ax.yaxis.set_major_formatter(_hex_tick_formatter)

    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'hot'
    if 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'nearest'

    log = kwargs.pop('log', False)
    if log:
        if 'vmin' not in kwargs:
            kwargs['vmin'] = -1.0
        ax.imshow(numpy.log2(dg), **kwargs)
    else:
        ax.imshow(dg, **kwargs)

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

    a = binstats._cast_uint8_ndarray(a)

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

def entropy_digraph(a, **kwargs):
    """ Plot digraph entropy as a function of block number

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
    Blocksizes that are too small (less than about 0x10000 bytes) will be
    subject to significant amounts of random noise, possibly leading to false
    estimates of the variation of the entropy. There are 0x10000 possible
    symbols in the set, so a blocksize that is at least a hundred kilobytes or
    so should be chosen for best results.
    """

    a = binstats._cast_uint8_ndarray(a)

    if 'bs' in kwargs:
        bs = kwargs.pop('bs')
    else:
        bs = a.size//1024
        if bs < 0x20000:
            bs = 0x20000

    blocks = a.size//bs + (0 if a.size % bs == 0 else 1)
    entropy = numpy.zeros(blocks)
    for i in range(blocks):
        entropy[i] = binstats.entropy_digraph(a.flatten()[i*bs:(i+1)*bs])

    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
    else:
        fig = pyplot.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel("Block (size=%i bytes)" % bs)
        ax.set_ylabel("Shannon entropy over digraphs")
        ax.set_ylim(bottom=-0.1, top=16.2)

    ax.plot(entropy, **kwargs)

    return ax

def autocorrelation(a, mode='direct', **kwargs):
    """ Plot autocorrelation-like computation

    Plots a variant of the autocorrelation. It can be calculated directly by
    xoring with shifted version of signal, and summing to determine number of
    zero components. Or it can be calculated indirectly by taking an FFT of the
    signal, multiplying the FFT by its complex conjugate, then applying the
    inverse FFT operation.
    
    Parameters
    ----------
    a : array_like or bytestring
        Contains data to plot
    mode : string (optional)
        Computation mode. Valid values are 'direct' (default) or 'fft'.
    ax : matploitlib.axes._subplots.AxesSubplot instance (optional)
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

    The FFT calculation of the autocorrelation is sensitive to window size in a
    way that the direct calculation is not. In particular, spurious peaks may
    appear in the FFT-generated autocorrelation if the window size is not an
    even multiple of the natural data period. These peaks tend to appear as
    sidebands from the real peaks, at a position dependent on the difference
    between the the chosen data window size and the nearest even multiple of
    the natural data period. Some iteration may be useful to help identify real
    peaks. Varying the window width slightly will cause these spurious peaks to
    shift location, and disappear altogether if the window width is chosen
    correctly. This detection process could be automated, but is currently left
    to the user to perform manually.

    We could operate on the bitstream for better results in noisy channels with
    the possibility of single-bit errors, but for now we'll stick to byte
    comparisons. The user could send in their own bitstream with
    numpy.unpackbits() if they wanted this type of operation.
    """

    a = binstats._cast_uint8_ndarray(a)

    if mode == 'fft':
        rfft = numpy.fft.rfft(a.flatten())
        autocorr = numpy.fft.irfft(rfft*rfft.conj())
    else:
        # Pre-fill the array with the maximum number of zeros that we could
        # observe, and then subtract off the number that we *don't* observe.
        autocorr = numpy.arange(a.size, 0, -1, dtype=numpy.uint64)

        for i in range(1, a.size-1):
            autocorr[i] -= numpy.count_nonzero(numpy.bitwise_xor(a[:-i],
                                                                 a[i:]))
 
    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
    else:
        fig = pyplot.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel("Shift")
        ax.set_ylabel("Autocorrelation")
        ax.set_xscale('log')

    ax.plot(autocorr, **kwargs)

    return ax

