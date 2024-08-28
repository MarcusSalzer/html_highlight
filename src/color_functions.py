import numpy as np
from colorsys import hsv_to_rgb

def palette_from_background(
    n_colors: int, hsv_background=(0.0, 0.5, 0.5), spread=(0.0, 0.0, 0.0), noise=0
):
    """Generate a palette contrasting background.
    TODO: S and V spread"""
    
    rng = np.random.default_rng()

    # opposite to background
    h_primary = (hsv_background[0] + 0.5) % 1.0

    # linearly spaced hues
    h_vec = np.linspace(h_primary - spread[0] / 2, h_primary + spread[0] / 2, n_colors)
    h_vec += noise * rng.normal(size = n_colors)
    h_vec = np.round(h_vec % 1.0, 8)

    s_vec = 0.6 * np.ones(n_colors)
    v_vec = np.ones(n_colors)
    return [hsv_to_rgb(*hsv) for hsv in zip(h_vec, s_vec, v_vec)]


def rgb2hex(r, g, b):
    """Convert rgb color to hex.

    ## Parameters
    r,g,b: int (0-255) or float(0-1)"""

    if isinstance(r, int):
        f = 1
    else:
        f = 255
    return "#" + "".join(f"{round(i*f):02x}" for i in (r, g, b))


def random_rgb():
    """Generate a random 3-tuple"""
    rng = np.random.default_rng()
    return tuple(rng.random(size=3))
