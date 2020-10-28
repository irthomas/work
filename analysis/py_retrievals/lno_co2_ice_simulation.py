# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 09:45:31 2020

@author: iant

LNO AOTF SIMULATION WITH GIANCARLO CO2 ICE PROFILE
"""


import numpy as np
import matplotlib.pyplot as plt

from instrument.nomad_lno_instrument import nu_mp, F_blaze, F_aotf_goddard19draft
from tools.plotting.colours import get_colours

def norm(x):
    return x / np.max(x)


ice_profile_raw = np.asfarray([
    [4200.0000,1.04571],
    [4248.1224,1.04571],
    [4249.1655,1.04454],
    [4249.8331,1.04517],
    [4250.7928,1.0428 ],
    [4251.6273,1.03621],
    [4252.5869,1.02902],
    [4253.1711,1.02121],
    [4253.8387,0.99413],
    [4254.089,0.96522 ],
    [4254.3394,0.93511],
    [4254.548,0.89597 ],
    [4254.8401,0.86766],
    [4255.3616,0.8466 ],
    [4256.0501,0.84362],
    [4256.5299,0.8599 ],
    [4256.9471,0.88341],
    [4257.5104,0.90572],
    [4258.3449,0.90877],
    [4259.2211,0.92447],
    [4260.1808,0.93535],
    [4261.1822,0.94382],
    [4262.6634,0.95051],
    [4264.7914,0.94879],
    [4265.9805,0.95124],
    [4266.8359,0.95309],
    [4267.8999,0.96096],
    [4268.9013,0.96943],
    [4269.8609,0.97791],
    [4270.6328,0.98456],
    [4271.6551,0.99244],
    [4272.6565,0.99971],
    [4273.4284,1.00516],
    [4274.5341,1.01062],
    [4275.5355,1.01548],
    [4276.5369,1.01914],
    [4277.58,1.02279  ],
    [4278.5814,1.02524],
    [4279.6245,1.02649],
    [4300.0000,1.02649]
])

nu_grid = np.arange(4200., 4300., 0.01)

ice_profile = np.interp(nu_grid, ice_profile_raw[:, 0], ice_profile_raw[:, 1])

# ice_profile -= 0.7
ice_profile /= np.max(ice_profile)

flat_profile = np.ones_like(ice_profile)

central_order = 189
pixels = np.arange(320.0)
temperature = -10.0
n_orders = 1

colours = get_colours(n_orders *2 + 1)

plt.figure(figsize=(15, 6))
plt.title("CO2 ice spectrum, applying blaze function and order addition")
plt.xlabel("Wavenumbers cm-1")
plt.ylabel("Normalised response")
aotf_grid = F_aotf_goddard19draft(central_order, nu_grid, temperature)
# plt.plot(nu_grid, aotf_grid)
# plt.plot(nu_grid, ice_profile)

pixel_response_ice = np.zeros(len(pixels))
pixel_response_flat = np.zeros(len(pixels))

nu_px_centre = nu_mp(central_order, pixels, temperature)

for order_index, diffraction_order in enumerate(range(central_order-n_orders, central_order+n_orders+1, 1)):

    nu_pixels = nu_mp(diffraction_order, pixels, temperature)
    blaze = F_blaze(diffraction_order, pixels, temperature)

    #interpolate ice onto pixel grid
    ice_pixels = np.interp(nu_pixels, nu_grid, ice_profile)
    flat_pixels = np.interp(nu_pixels, nu_grid, flat_profile)
    
    aotf_pixels = np.ones_like(nu_pixels)
    # aotf_pixels = np.interp(nu_pixels, nu_grid, aotf_grid)
    
    # plt.plot(nu_pixels, blaze, color=colours[order_index])
    plt.plot(nu_pixels, ice_pixels-order_index/10.0, color=colours[order_index], label="Order %i: CO2 ice spectrum (w/ offset)" %diffraction_order)
    plt.fill_betweenx([0., 1.], nu_pixels[0], x2=nu_pixels[-1], color=colours[order_index], alpha=0.1, label="Order %i: spectral range" %diffraction_order)
    
    # plt.plot(nu_pixels, ice_pixels * blaze, color=colours[order_index], linestyle=":")
    plt.plot(nu_pixels, blaze * aotf_pixels, color=colours[order_index], linestyle=":", label="Order %i: Flat spectrum * blaze function" %diffraction_order)
    plt.plot(nu_pixels, ice_pixels * blaze * aotf_pixels, color=colours[order_index], linestyle="--", label="Order %i: Ice spectrum * blaze function" %diffraction_order)

    pixel_response_ice += ice_pixels * blaze * aotf_pixels
    pixel_response_flat += blaze * aotf_pixels

plt.legend()
plt.tight_layout()
plt.savefig("co2_ice_order_addition.png")
plt.figure(figsize=(14, 5))
plt.xlabel("Wavenumbers cm-1")
plt.ylabel("Normalised response")

# plt.plot(nu_px_centre, pixel_response_ice)
# plt.plot(nu_px_centre, pixel_response_flat)
plt.plot(nu_px_centre, norm(norm(pixel_response_ice)/norm(pixel_response_flat)), color="g")
plt.xlim([4248., 4280.])
plt.tight_layout()
plt.savefig("co2_ice_blaze_function_spectrum.png")
