from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

file_path = Path(r"C:\Users\iant\Documents\DOCUMENTS\science\yannick_o3\Willame_JGR2025_withO3.txt")
file_path = Path(r"C:\Users\iant\Documents\DOCUMENTS\science\yannick_o3\snippet.txt")

print("Loading file")
data = np.genfromtxt(
    file_path,
    delimiter=",",
    names=True,
    dtype=None,
    encoding=None,
    autostrip=True,
)

print("Finding columns")
# columns = {column.strip().lower(): column for column in data.dtype.names}
lat_in, lon_in, o3_in, ls_in = data["lat_Â°"], data["lon_Â°"], data["O3_Dobson"], data["Ls_Â°"]


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


print("Sorting data")
lat = np.fromiter((to_float(value) for value in lat_in), dtype=float)
lon = np.fromiter((to_float(value) for value in lon_in), dtype=float)
o3 = np.fromiter((to_float(value) for value in o3_in), dtype=float)
ls = np.fromiter((to_float(value) for value in ls_in), dtype=float)
valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(o3) & np.isfinite(ls)

print("Plotting Ls", np.min(ls), "to", np.max(ls))
fig, ax = plt.subplots(figsize=(9, 6))
plot = ax.scatter(lon[valid], lat[valid], c=o3[valid], cmap="viridis", vmin=0, vmax=2)
fig.colorbar(plot, ax=ax, label="O3")
ax.set(xlabel="Longitude", ylabel="Latitude", title="UVIS Nadir O3 (Yannick), Ls %0.1f to %0.1f" % (np.min(ls), np.max(ls)))
plt.tight_layout()
plt.show()
