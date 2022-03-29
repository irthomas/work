# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 08:51:39 2022

@author: iant

CHECK UVIS SO SPECTRA FOR OSCILLATIONS
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from tools.file.hdf5_functions import make_filelist, get_file


regex = re.compile("20181029_203917_.*_UVIS_E")

file_level = "hdf5_level_1p0a"


hdf5_files, hdf5_filenames, _ = make_filelist(regex, file_level)

h5 = hdf5_files[0]
h5_filename = hdf5_filenames[0]


df = pd.DataFrame(np.array(h5['Science/Y']).T, index=np.array(h5['Science/X'][0,:]), columns=np.array(h5['Geometry/Point0/TangentAltAreoid'][:, 0]))

df_500 = df[(df.index > 500) & (df.index < 600)]

df_rolling = df_500.rolling(5, axis=0, center=True).mean()




df_poly = df_500.copy()
for i in df_500.columns:
    df_poly[i] = np.polyval(np.polyfit(np.arange(df_500.shape[0]), df_500[i], 6), np.arange(df_500.shape[0]))

df_rolling_sub = df_500 - df_rolling
df_poly_sub = df_500 - df_poly


# fig, ax = plt.subplots()
# df.plot(colormap=plt.cm.get_cmap("Spectral"), ax=ax)
# df_rolling.plot(ax=ax)
# df_poly.plot(ax=ax)

# fig, ax = plt.subplots()
# df_rolling_sub.plot(ax=ax)
# df_poly_sub.plot(ax=ax)




rolling_std = df_rolling_sub.std()
poly_std = df_poly_sub.std()

sigma = poly_std / rolling_std


oscillations = np.where(sigma>1.5)[0]

osc_alt = df.columns[oscillations[0]]

# df_rolling_sub.loc[:, sigma>1.5].plot()
# df_poly_sub.loc[:, sigma>1.5].plot()







h5_filename_3b, h5_3b = get_file(h5_filename.replace("1p0a", "0p3b"), "hdf5_level_0p3b", 0)

y_3b = h5_3b["Science/Y"][...]
x_3b = h5_3b["Science/X"][0, :]

alts_3b = h5_3b["Geometry/Point0/TangentAltAreoid"][:, 0]

i_3b = np.where(alts_3b == osc_alt)[0][0] -1

# fig, axes = plt.subplots(nrows=3)
# for i, index in enumerate(range(i_3b-1, i_3b+2)):
#     axes[i].imshow(y_3b[index, :, :]/y_3b[index-1, :, :])
    
image1_3b = y_3b[i_3b, :, 659:893]
image2_3b = y_3b[i_3b-1, :, 659:893]
x500_3b = x_3b[659:893]

# plt.figure()
# for i in range(10, 30):
#     plt.plot(x500_3b, image1_3b[i, :] / image2_3b[i, :], label=i)
# plt.legend()


mean_ill_3b = np.mean(image1_3b[15:25, :], axis=0) / np.mean(image2_3b[15:25, :], axis=0)

poly_val = np.polyval(np.polyfit(x500_3b, mean_ill_3b, 3), x500_3b)

plt.figure()
plt.plot(x500_3b, mean_ill_3b / poly_val)






h5_filename_2a, h5_2a = get_file(h5_filename.replace("1p0a", "0p2a"), "hdf5_level_0p2a", 0)

y_2a = h5_2a["Science/Y"][...]
x_2a = h5_2a["Science/X"][0, :]

v_start = h5_2a["Channel/VStart"][0]

alts_2a = h5_2a["Geometry/Point0/TangentAltAreoid"][:, 0]

i_2a = np.where(alts_2a == osc_alt)[0][0] -1

# fig, axes = plt.subplots(nrows=3)
# for i, index in enumerate(range(i_3b-1, i_3b+2)):
#     axes[i].imshow(y[index, :, :]/y[index-1, :, :])
    
image1_2a = y_2a[i_2a, :, 659:893]
image2_2a = y_2a[i_2a-1, :, 659:893]
x500_2a = x_2a[659:893]

plt.figure()
for i in range((v_start+10), (v_start+30)):
    plt.plot(x500_2a, image1_2a[i, :] / image2_2a[i, :], label=i)
plt.legend()


mean_ill_2a = np.mean(image1_2a[(v_start+15):(v_start+25), :], axis=0) / np.mean(image2_2a[(v_start+15):(v_start+25), :], axis=0)

poly_val = np.polyval(np.polyfit(x500_2a, mean_ill_2a, 3), x500_2a)

plt.figure()
plt.plot(x500_2a, mean_ill_2a / poly_val)