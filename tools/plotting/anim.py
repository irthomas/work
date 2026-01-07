# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 12:27:50 2015

@author: iant

TOOLS FOR MAKING ANIMATIONS
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

from tools.file.paths import paths


def make_line_anim(d):
    """
    TO WORK, MUST KEEP THE RETURNED ANIM OBJECT IN MEMORY e.g anim = make_line_anim(d)
    pass dictionary to function to make a line plot animation. Keys are:
    'x':{name:1d or 2d data}, #name = dataset name, 1d or 2d data array
    'y':{name:2d data},  #name = dataset name, 2d data array
    'text':[], #text to add to each animation frame, required
    'text_position':[x,y], #text position in data coords, required
    'xlabel', #optional
    'ylabel', #optional
    'xlim', #[min, max], optional
    'ylim', #[min, max], optional
    'save', #True = save to file
    'filename', #filename to save to
    'legend':{}, 
    'keys':[], #keys to plot from x and y, leave blank to plot all
    'title', 
    'x_params':{'1d':True} #1d = is x[key] a 1d array for all keys?
    """

    if "format" in d.keys():
        anim_format = d["format"]
    else:
        anim_format = "html"

    ext = {"ffmpeg": "mp4", "html": "html", "pillow": "gif"}[anim_format]

    # set up formatting for movie files
    # print("Setting up animation")
    # Writer = animation.writers[anim_format]
    # writer = Writer(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    # print("Animation setup done")

    # get list of keys or make list of all keys in data
    if "keys" in d.keys():
        keys = d["keys"]
    else:
        keys = list(d["y"].keys())[:]
    print("Plotting keys:", keys)

    xlabel = ""
    if "xlabel" in d.keys():
        xlabel = d["xlabel"]

    ylabel = ""
    if "ylabel" in d.keys():
        ylabel = d["ylabel"]

    if "xlim" in d.keys():
        xlim = d["xlim"]
    else:
        xlim = [np.nanmin([d["x"][key] for key in keys]), np.nanmax([d["x"][key] for key in keys])]
        ylim = [np.nanmin([d["y"][key] for key in keys]), np.nanmax([d["y"][key] for key in keys])]

    if "interval" in d.keys():
        interval = int(d["interval"])
    else:
        interval = 50

    ylim[0] /= 1.01
    ylim[1] *= 1.01

    # # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    print("xlim:", xlim, "ylim:", ylim, "xlabel:", xlabel, "ylabel:", ylabel)
    ax = plt.axes(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)

    if "title" in d.keys():
        ax.set_title(d["title"])
        print(d["title"])

    # check if x is 1d or 2d
    x_1d = False
    if "x_params" in d:
        if "1d" in d["x_params"]:
            if d["x_params"]["1d"]:
                x_1d = True
    print("is x 1D:", x_1d)

    lines = []
    for key in keys:
        lines.append(ax.plot([], [], lw=2, label=key)[0])
    text = ax.text(d["text_position"][0], d["text_position"][1], "")

    if "legend" in d.keys():
        if "on" in d["legend"].keys():
            if d["legend"]["on"]:
                if "loc" in d["legend"].keys():
                    ax.legend(loc=d["legend"]["loc"])
                else:
                    ax.legend()

    # initialization function: plot the background of each frame
    def init():
        for line in lines:
            line.set_data([], [])
        text.set_text("")

        artists = lines + [text]
        return artists

    # animation function.  This is called sequentially
    def animate(i):
        if np.mod(i, 100) == 0:
            print(i)

        for line, key in zip(lines, keys):
            if x_1d:
                x = d["x"][key]
            else:
                x = d["x"][key][i]

            line.set_data(x, d["y"][key][i, :])
        text.set_text(d["text"][i])

        artists = lines + [text]
        return artists

    n_frames = len(d["y"][list(d["y"].keys())[0]])
    print("N frames = %i" % n_frames)
    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=interval, blit=True)

    if "save" in d.keys() and "filename" in d.keys():
        if d["save"]:
            try:
                cwd = os.getcwd()
                os.chdir(paths["ANIMATION_DIRECTORY"])
                anim.save("%s.%s" % (d["filename"], ext), writer=anim_format)
            except Exception as e:
                print(e)
            finally:
                os.chdir(cwd)
    plt.show()

    return anim


def make_frame_anim(list_of_frames, zmin, zmax, filename, ymax=256):
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib import animation

    # Set up formatting for the movie files
    Writer = animation.writers['html']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    plt.axes(xlim=(0, 320), ylim=(0, ymax), xlabel="Spectral Pixel", ylabel="Spatial Pixel")
    im = plt.imshow(list_of_frames[0], interpolation='none', aspect=int(np.floor(256/ymax)), origin='upper', cmap=plt.cm.Spectral)
    plt.clim(zmin, zmax)
    plt.colorbar()

    # initialization function: plot the background of each frame

    def init():
        im.set_data(list_of_frames[0])
        return [im]

    # animation function.  This is called sequentially
    def animate(i):
        im.set_data(list_of_frames[i])
#        print np.mean(list_of_frames[i])
        return [im]

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(list_of_frames), interval=50, blit=True)

    cwd = os.getcwd()
    os.chdir(paths["ANIMATION_DIRECTORY"])
    anim.save(filename+".html", writer=writer)

    os.chdir(cwd)
    plt.show()
    return 0


# """test with basemap"""
# from mpl_toolkits.basemap import Basemap
# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.animation as animation

# map = Basemap(resolution='l',projection='ortho',lon_0=0,lat_0=0)
# map.drawcoastlines()
# map.drawcountries()
# map.fillcontinents(color = 'gray')
# map.drawmapboundary()
# map.drawmeridians(np.arange(0, 360, 30))
# map.drawparallels(np.arange(-90, 90, 30))

# x,y = map(0, 0)
# point = map.plot(x, y, 'ro', markersize=5)[0]

# def init():
#     point.set_data([], [])
#     return point,

# # animation function.  This is called sequentially
# def animate(i):
#     lons, lats =  np.random.random_integers(-130, 130, 2)
#     x, y = map(lons, lats)
#     point.set_data(x, y)
#     return point,

# # call the animator.  blit=True means only re-draw the parts that have changed.
# anim = animation.FuncAnimation(plt.gcf(), animate, init_func=init,
#                                frames=20, interval=500, blit=True)

# plt.show()
