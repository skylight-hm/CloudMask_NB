import numpy
import six


def get_conv_outsize(size, k, s, p, cover_all=False, d=1):
    """Calculates output size of convolution.

    This function takes the size of input feature map, kernel, stride, and
    pooling of one particular dimension, then calculates the output feature
    map size of that dimension.

    .. seealso:: :func:`~chainer.utils.get_deconv_outsize`

    Args:
        size (int): The size of input feature map. It usually is the length of
            a side of feature map.
        k (int): The size of convolution kernel.
        s (int): The size of stride.
        p (int): The size of padding.
        cover_all (bool): Use ``cover_all`` option or not.
        d (int): The size of dilation.

    Returns:
        int: The expected output size of the convolution operation.

    """
    dk = k + (k - 1) * (d - 1)
    if cover_all:
        return (size + p * 2 - dk + s - 1) // s + 1
    else:
        return (size + p * 2 - dk) // s + 1


def get_deconv_outsize(size, k, s, p, cover_all=False, d=1):
    """Calculates output size of deconvolution.

    This function takes the size of input feature map, kernel, stride, and
    pooling of one particular dimension, then calculates the output feature
    map size of that dimension.

    .. seealso:: :func:`~chainer.utils.get_conv_outsize`

    Args:
        size (int): The size of input feature map. It usually is the length of
            a side of feature map.
        k (int): The size of deconvolution kernel.
        s (int): The size of stride.
        p (int): The size of padding.
        cover_all (bool): Use ``cover_all`` option or not.
        d (int): The size of dilation.

    Returns:
        int: The expected output size of the deconvolution operation.

    """
    dk = (k - 1) * d + 1
    if cover_all:
        return s * (size - 1) + dk - s + 1 - 2 * p
    else:
        return s * (size - 1) + dk - 2 * p


def im2col_cpu(
    img, kh, kw, sy, sx, ph, pw, pval=0, cover_all=False, dy=1, dx=1,
    out_h=None, out_w=None):
    n, c, h, w = img.shape
    if out_h is None:
        out_h = get_conv_outsize(h, kh, sy, ph, cover_all, dy)
    assert out_h > 0, 'Height in the output should be positive.'
    if out_w is None:
        out_w = get_conv_outsize(w, kw, sx, pw, cover_all, dx)
    assert out_w > 0, 'Width in the output should be positive.'

    img = numpy.pad(img,
                    ((0, 0), (0, 0), (ph, ph + sy - 1), (pw, pw + sx - 1)),
                    mode='constant', constant_values=(pval,))
    col = numpy.ndarray((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    for j in six.moves.range(kh):
        jdy = j * dy
        j_lim = jdy + sy * out_h
        for i in six.moves.range(kw):
            idx = i * dx
            i_lim = idx + sx * out_w
            col[:, :, j, i, :, :] = img[:, :, jdy:j_lim:sy, idx:i_lim:sx]

    return col


def cal_nxn_indices(array, n=2, func=numpy.max):
    array = array[numpy.newaxis, numpy.newaxis]
    k_h = 2 * n + 1
    k_w = 2 * n + 1
    p_h = n
    p_w = n
    s_h = 1
    s_w = 1

    col = im2col_cpu(array, k_h, k_w, s_h, s_w, p_h, p_w, pval=65535)
    col_m = im2col_cpu(array.mask, k_h, k_w, s_h, s_w, p_h, p_w, pval=True)

    array_indices = numpy.ma.masked_array(col, col_m)
    array_indices = func(array_indices, (0, 1, 2, 3))
    return array_indices
