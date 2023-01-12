import numpy as np

def random_cutout(img, min_cut=1,max_cut=4):
    """
        args:
        imgs: np.array shape (B,H,W)
        min / max cut: int, min / max size of cutout 
        returns np.array
    """
    h, w = img.shape
    w1 = np.random.randint(min_cut, max_cut)
    h1 = np.random.randint(min_cut, max_cut)
    
    cutout = np.empty((h, w), dtype=img.dtype)

    cut_img = img.copy()
    cut_img[ h1:h1 + h1, w1:w1 + w1] = -1
    cutout = cut_img

    return cutout

def random_translate(img, size = 8, return_random_idxs=False, h1s=None, w1s=None):
    h, w = img.shape
    assert size >= h and size >= w
    out = np.zeros(( size, size), dtype=img.dtype)
    h1s = np.random.randint(0, size - h + 1) if h1s is None else h1s
    w1s = np.random.randint(0, size - w + 1) if w1s is None else w1s

    out[ h1s:h1s + h, w1s:w1s + w] = img
    return out

def random_crop(img, out=6):
    """
        args:
        imgs: np.array shape (B,H,W)
        out: output size (e.g. 84)
        returns np.array       """


    h, w = img.shape
    crop_max = h - out + 1
    w1 = np.random.randint(0, crop_max)
    h1 = np.random.randint(0, crop_max)
    cropped = np.empty(( out, out), dtype=img.dtype)

    cropped = img[ h1:h1 + out, w1:w1 + out]

    return cropped