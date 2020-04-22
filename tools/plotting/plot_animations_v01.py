# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 15:36:34 2018

@author: iant


FUNCTIONS TO ANIMATE DATA
"""
#import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animateLines(lines, text="", label="", speed=100):
    
#    colours = ["r","g","b","k","c"]
    
    x = np.arange(len(lines[0][0,:]))
    n_frames = len(lines[0][:,0])
    n_lines = len(lines)
    max_value = np.nanmax(lines)
    min_value = np.nanmin(lines)
    fig, ax = plt.subplots(1, figsize=(10,8))
    num = 0
    
    plt.ylim((min_value*0.9,max_value*1.1))
    if label == "":
        line_array = [plt.plot(x, lines[index][num,:], animated=True)[0] for index in range(n_lines)]
    else:
        line_array = [plt.plot(x, lines[index][num,:], animated=True, label=label[index])[0] for index in range(n_lines)]
        plt.legend()
        
    text_y = 0.99
    if text == "":
        plottext = ax.text(10, text_y, "Frame %i" %(num))
    else:
        plottext = ax.text(10, text_y, text[num])

    def updatefig(num): #always use num, which is sent by the animator. a loop variable will keep increasing as the animation is repeated!
#        global plot #,plottitle#,detector_data,variable_changing,what_to_animate,line_number,sum_vertically
        if np.mod(num,50)==0:
            print(num)
            
        for index in range(n_lines):
            line_array[index].set_data(x, lines[index][num,:])
        
        if text == "":
            plottext.set_text("Frame %i" %(num))
        else:
            plottext.set_text(text[num])
        artists = line_array + [plottext]
        return artists
            
    ani = animation.FuncAnimation(fig, updatefig, frames=n_frames, interval=speed, blit=True)
#    if save_figs: ani.save(title+"_detector_%s.mp4" %what_to_animate, fps=20, extra_args=['-vcodec', 'libx264'])
    plt.show()
    return ani