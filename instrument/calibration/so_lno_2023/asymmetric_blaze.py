# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:55:20 2024

@author: iant

ASYMMETRIC BLAZE FROM LOIC
"""

import numpy as np
# import matplotlib.pyplot as plt

# from instrument.nomad_lno_instrument_v02 import nu_mp


def compute_beta(wvn, order, sigma, gamma, alpha):
    return np.arcsin(order/wvn/sigma/np.cos(gamma)-np.sin(alpha))


# px_ixs = np.arange(320.0)
# order = 191.0
# px_nus = nu_mp(order, px_ixs, 0.0) - 2.0
# channel = "lno"


def asymmetric_blaze(channel, order, px_nus):
    """Loic asymmetric blaze function"""
    if channel.lower() == "so":
        gamma, alphab = 0.04538, -0.0003491
    elif channel.lower() == "lno":
        gamma, alphab = 0.04764, 0.0
    thetab = 1.10706
    sigma = 0.024792

    alpha = alphab + thetab
    beta = compute_beta(px_nus, order, sigma, gamma, alpha)

    sarg = (px_nus * sigma * np.cos(gamma) * np.cos(alpha) / np.cos(alphab) * (np.sin(alphab) + np.sin(beta - thetab)))
    sinc2 = np.sinc(sarg)**2.0

    effect = np.zeros_like(px_nus)
    mask1 = [i for i, bi in enumerate(beta) if alpha >= bi]
    mask2 = [i for i, bi in enumerate(beta) if alpha < bi]
    effect[mask1] = 1.0
    effect[mask2] = (np.cos(beta[mask2]) / np.cos(alpha))**2.0
    # lamb and beta are vectors
    blazefct = effect * sinc2

    # plt.plot(scan_line)
    # plt.plot(sinc2)
    # plt.plot(blazefct*280000)

    return blazefct
