# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 12:27:50 2015

@author: iant
"""
def make_slice_anim(list_of_slices,y_min,y_max,filename):
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib import animation
    
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 320), ylim=(y_min, y_max), xlabel="Detector Pixel")
    line, = ax.plot([], [], lw=2)
    
    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,
    
    # animation function.  This is called sequentially
    def animate(i):
        x = np.arange(0.0,320.0,1.0)
        y = list_of_slices[i]
        line.set_data(x, y)
        return line,
    
    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,frames=len(list_of_slices), interval=50, blit=True)
    
    anim.save(filename+'.mp4', fps=30)
#    plt.show()
    return 0
    
    
def make_frame_anim(list_of_frames,z_min,z_max,filename):
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib import animation
    
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    plt.axes(xlim=(0, 320), ylim=(0, 256), xlabel="Spectral Pixel", ylabel="Spatial Pixel")
    im=plt.imshow(list_of_frames[0],interpolation='none')
    plt.clim(z_min,z_max)
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
    anim = animation.FuncAnimation(fig, animate, init_func=init,frames=len(list_of_frames), interval=50, blit=True)
    anim.save(filename+'.mp4')
    plt.show()
    return 0

#frames = []
#for loop in range(3):
#    frames.append(np.random.random((256,320)))
#    
#make_frame_anim(frames,0,0,'test2')






"""test with basemap"""
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

map = Basemap(resolution='l',projection='ortho',lon_0=0,lat_0=0)
map.drawcoastlines()
map.drawcountries()
map.fillcontinents(color = 'gray')
map.drawmapboundary()
map.drawmeridians(np.arange(0, 360, 30))
map.drawparallels(np.arange(-90, 90, 30))

x,y = map(0, 0)
point = map.plot(x, y, 'ro', markersize=5)[0]

def init():
    point.set_data([], [])
    return point,

# animation function.  This is called sequentially
def animate(i):
    lons, lats =  np.random.random_integers(-130, 130, 2)
    x, y = map(lons, lats)
    point.set_data(x, y)
    return point,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(plt.gcf(), animate, init_func=init,
                               frames=20, interval=500, blit=True)

plt.show()