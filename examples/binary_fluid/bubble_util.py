"""Utilities to analyze Taylor flow simulations."""

import numpy as np
from collections import namedtuple
from scipy import interpolate
from scipy import optimize

Bubble = namedtuple('Bubble', 'start end len width mid')

def FindBubble(phi):
    """Locates the bubble in the simulation domain.

    $\phi = 0$ is used as the bubble boundary.

    Returns:
      start, end, length, midpoint width, bubble midpoint
    """
    # Drop wall nodes.
    real_phi = phi[2:-2,:]
    mid = real_phi.shape[0] // 2 + 1
    # Streamwise profile.
    profile = real_phi[mid,:]
    x = np.arange(len(profile))
    # cubic here can cause scipy to hang..
    prof_int = interpolate.interp1d(x, profile)

    w = np.where(profile < 0.0)
    x0 = np.min(w)
    x1 = np.max(w)

    # Bubble at the start and end of the subdomain, with liquid
    # phase in the middle.
    if x0 == 0:
        w = np.where(profile > 0.0)
        x1 = np.min(w)
        x0 = np.max(w)

    x0 = optimize.newton(prof_int, x0)
    x1 = optimize.newton(prof_int, x1)
    if x1 > x0:
        mid = int(round(0.5 * (x0 + x1)))
        blen = x1 - x0
    else:
        rem = len(profile) - x0
        mid = int(round(0.5 * (x1 + rem)))
        blen = x1 + rem

        if mid < rem:
            mid += x0
        else:
            mid -= rem

    width = EstimateBubbleWidth(real_phi[:,mid])
    return Bubble(x0, x1, blen, width, mid)


def EstimateBubbleWidth(profile):
    # Profile is expected to be stripped of nan's (wall nodes).
    assert not np.any(np.isnan(profile))
    # Physical position is shifted due to wall location between
    # the fluid and no-slip node.
    x = np.arange(len(profile)) + 0.5
    prof_int = interpolate.interp1d(x, profile, kind='cubic')
    y0 = optimize.brenth(prof_int, 0.5, len(profile) / 2)
    y1 = optimize.brenth(prof_int, len(profile) / 2, x[-1])
    return y1 - y0


def EstimateBubbleVelocity(a, b, dt):
    x0, _, _, _, _ = FindBubble(a)
    x1, _, _, _, _ = FindBubble(b)
    if x0 > x1:
        return (x1 + a.shape[1] - x0) / float(dt)
    else:
        return (x1 - x0) / float(dt)
