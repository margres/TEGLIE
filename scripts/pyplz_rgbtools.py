import numpy as np
from PIL import Image

def marshall16_pil_format(images, scales=(1., 1., 1.), alpha=1., Q=1.):
    """
    Parameters:
    - images: List of image channels (R, G, B)
    - scales: Tuple of scaling factors for each channel
    - alpha: Parameter for the arcsinh function
    - Q: Parameter for the arcsinh function

    Output:
    - List of RGB tuples in PIL format
    """
    r = images[0] * scales[0]
    g = images[1] * scales[1]
    b = images[2] * scales[2]

    I = (r + g + b)
    f = np.arcsinh(alpha * I * Q)

    R = r * f / (I * Q)
    G = g * f / (I * Q)
    B = b * f / (I * Q)

    R = R * 255.
    G = G * 255.
    B = B * 255.

    R[R < 0.] = 0.
    G[G < 0.] = 0.
    B[B < 0.] = 0.

    R[R > 255.] = 255.
    G[G > 255.] = 255.
    B[B > 255.] = 255.

    flatlist = []
    for img in [R, G, B]:
        img = np.uint8(img.round())
        img = np.flipud(img)
        flatlist.append(img.flatten())

    result = []
    for i in range(images[0].size):
        result.append((flatlist[0][i], flatlist[1][i], flatlist[2][i]))

    return result


def lupton04_pil_format(images, scales=(1., 1., 1.), beta=1.):
    """
    Parameters:
    - images: List of image channels (R, G, B)
    - scales: Tuple of scaling factors for each channel
    - beta: Parameter for the arcsinh function

    Output:
    - List of RGB tuples in PIL format
    """
    r = images[0] * scales[0]
    g = images[1] * scales[1]
    b = images[2] * scales[2]

    I = (r + g + b) / 3.
    f = np.arcsinh(I / beta)

    R = r * f / I
    G = g * f / I
    B = b * f / I

    M = max(R.max(), G.max(), B.max())

    R = R / M * 255.
    G = G / M * 255.
    B = B / M * 255.

    R[R < 0.] = 0.
    G[G < 0.] = 0.
    B[B < 0.] = 0.

    flatlist = []
    for img in [R, G, B]:
        img = np.uint8(img.round())
        img = np.flipud(img)
        flatlist.append(img.flatten())

    result = []
    for i in range(images[0].size):
        result.append((flatlist[0][i], flatlist[1][i], flatlist[2][i]))

    return result


def make_crazy_pil_format(data, cuts):
    """
    Parameters:
    - data: List of image channels (R, G, B)
    - cuts: Tuple of percentiles for each channel

    Output:
    - List of RGB tuples in PIL format
    """
    newlist = []
    for i in range(3):
        flatd = np.flipud(data[i]).flatten()
        flatd[flatd < 0.] = 0.
        flatd *= 255. / cuts[i]
        flatd[flatd > 255.] = 255.
        flatd = np.uint8(flatd.round())
        newlist.append(flatd)

    l = []

    for i in range(data[0].size):
        l.append((newlist[0][i], newlist[1][i], newlist[2][i]))

    return l


def make_one_rgb(sci, model, scheme='LINEAR', cuts=(99., 99., 99.), scales=None):
    """
    Parameters:
    - sci: List of data images (R, G, B)
    - model: List of model components
    - scheme: Color mapping scheme ('LINEAR' or 'M16')
    - cuts: Tuple of percentiles for each channel
    - scales: Scaling factors for the 'M16' scheme

    Output:
    - PIL Image containing the RGB representation
    """
    data = []
    fullresid = []
    fullmodel = []

    for i in range(3):
        data.append(sci[i])
        tmp_fullmodel = 0. * sci[i]
        for comp in model:
            tmp_fullmodel += comp[i]
        fullmodel.append(tmp_fullmodel)
        fullresid.append(sci[i] - tmp_fullmodel)

    if scheme == 'LINEAR':
        auto_cuts = []
        for i in range(3):
            cut = np.percentile(sci[i], cuts[i])
            auto_cuts.append(cut)

        dlist = make_crazy_pil_format(data, auto_cuts)
        fmlist = make_crazy_pil_format(fullmodel, auto_cuts)
        frlist = make_crazy_pil_format(fullresid, auto_cuts)

    elif scheme == 'M16':
        dlist = marshall16_pil_format(data, scales)
        fmlist = marshall16_pil_format(fullmodel, scales)
        frlist = marshall16_pil_format(fullresid, scales)

    s = (data[0].shape[1], data[0].shape[0])
    im = Image.new('RGB', (len(data) + len(model) + len(fullresid)) * s[1], s[0], 'black')

    dim = Image.new('RGB', s, 'black')
    fmim = Image.new('RGB', s, 'black')
    frim = Image.new('RGB', s, 'black')

    dim.putdata(dlist)
    fmim.putdata(fmlist)
    frim.putdata(frlist)

    im.paste(dim, (0, 0,))
    im.paste(fmim, (1 * s[1], 0))
    im.paste(frim, ((len(data) + len(model) - 1) * s[1], 0))

    for n in range(len(model)):
        rgb_list = []
        for i in range(3):
            rgb_list.append(model[n][i])
        if scheme == 'LINEAR':
            dlist = make_crazy_pil_format(rgb_list, auto_cuts)
        elif scheme == 'M16':
            dlist = marshall16_pil_format(rgb_list, scales)

        dim = Image.new('RGB', s, 'black')
        dim.putdata(dlist)
        im.paste(dim, ((2 + n) * s[1], 0,))

    return im


def make_full_rgb(sci_list, comp_list, outname='rgb.png', scheme='LINEAR', cuts=None, scales=None):
    """
    Parameters:
    - sci_list: List of data images
    - comp_list: List of model components
    - outname: Output file name for the combined RGB image
    - scheme: Color mapping scheme ('LINEAR' or 'M16')
    - cuts: Tuple of percentiles for each channel
    - scales: Scaling factors for the 'M16' scheme

    Output:
    - Save the combined RGB image
    """
    rgbsets = []

    ntotbands = len(sci_list)
    if ntotbands == 1:
        rgbsets.append((0, 0, 0))
    elif ntotbands == 2:
        rgbsets.append((1, 1, 0))
        rgbsets.append((0, 0, 0))
        rgbsets.append((1, 1, 1))
    elif ntotbands == 3:
        rgbsets.append((2, 1, 0))
        rgbsets.append((0, 0, 0))
        rgbsets.append((1, 1, 1))
        rgbsets.append((2, 2, 2))
    else:
        nsets = ntotbands - 2
        for i in range(nsets):
            rgbsets.append((i + 2, i + 1, i))
        for i in range(ntotbands):
            rgbsets.append((i, i, i))

    nsets = len(rgbsets)

    ncomp = len(comp_list[0])
    ncol = 3 + ncomp

    s = (len(sci_list[0][0]), len(sci_list[0]))
    fullim = Image.new('RGB', (ncol * len(sci_list[0]), nsets * len(sci_list[0][0])), 'black')

    for i in range(nsets):
        sci_here = []
        for ind in rgbsets[i]:
            sci_here.append(sci_list[ind])
        if scheme == 'M16':
            scales_here = []
            for ind in rgbsets[i]:
                scales_here.append(scales[ind])
        else:
            scales_here = None

        model_list = []
        for n in range(ncomp):
            model_here = []
            for ind in rgbsets[i]:
                model_here.append(comp_list[ind][n])
            model_list.append(model_here)

        im = make_one_rgb(sci_here, model_list, scheme=scheme, cuts=cuts, scales=scales_here)

        fullim.paste(im, (0, i * len(sci_list[0])))

    fullim.save(outname)

