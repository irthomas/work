# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:35:43 2024

@author: iant

EPO ACTIVITY: CALIBRATE GEOMETRY AND RADIOMETRY OF VMC IMAGES

READ IN ARSIA MONS CLOUD GIF

CALIBRATE LAT LON WITH 

https://commons.wikimedia.org/wiki/File:Evolution_of_the_Arsia_Mons_Elongated_Cloud_ESA23201331.gif


"""


import tkinter as tk
from PIL import Image, ImageSequence, ImageTk
import numpy as np
# import matplotlib.pyplot as plt


gif_path = "C:/Users/iant/Downloads/Evolution_of_the_Arsia_Mons_Elongated_Cloud_ESA23201331.gif"



def get_frames(first, last):
    with Image.open(gif_path) as im:
        frames = []
        times = []
        times_all = [""]*25 + ["07:20", "07:25", "07:28", "07:33", "07:37", "07:42", "07:47", "07:51", "07:56", "08:04"]
        for i, frame in enumerate(ImageSequence.Iterator(im)):
            
            if i<first or i>=last:
                continue
            
            frame = np.array(frame.copy().convert('RGB').getdata(), dtype=np.uint8).reshape(frame.size[1],
                                                                                            frame.size[0],
                                                                                            3)
            frames.append(frame)
            times.append(times_all[i])
    return frames, times



if "frames" not in globals():
    print("Getting image data")
    frames, times = get_frames(25, 35)
    
    #cut borders to fit the frame better
    frames = [frame[80:700, 200:1550, :] for frame in frames]




class MainWindow():
    
    def __init__(self, main, frames, times):

        n_frames = len(frames)
        
        self.imgtks = []
        for frame in frames:
            print("Converting")
            im = Image.fromarray(frame[:, :, :])
            imgtk = main.imgtk = ImageTk.PhotoImage(image=im)
            self.imgtks.append(imgtk)
            
            # print(imgtk.width(), imgtk.height())
            # plt.imshow(frame)
            
        self.geom_d = {"x":[], "y":[], "lat":[], "lon":[]}


        self.top_left = tk.Frame(main, bg='black', width=1200, height=600)
        self.top_right = tk.Frame(main, bg='black', width=100, height=400)
        self.bottom_left = tk.Frame(main, bg='black', width=1200, height=100)
        self.bottom_right = tk.Frame(main, width=100, height=100)

        self.top_left.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
        self.top_right.grid(row=0, column=1, padx=10, pady=10, sticky="ne")
        self.bottom_left.grid(row=1, column=0, padx=10, pady=10, sticky="sw")
        self.bottom_right.grid(row=1, column=1, padx=10, pady=10, sticky="se")
        
        
        
        # self.label = tk.Label(self.top_left, image=self.imgtks[1], bg='black')
        # self.label.pack()
        
        self.canvas = tk.Canvas(self.top_left,width=1350,height=620)
        self.canvas.pack()
        self.canvas_img = self.canvas.create_image((0,0), image=self.imgtks[0], anchor='nw')
        
        self.labeltext = tk.StringVar()
        self.labeltext.set("Mouse coordinates: 0, 0")
        
        self.label = tk.Label(self.bottom_left, textvariable=self.labeltext)
        self.label.pack()


        def geomWindow():
            self.geomWindow = tk.Toplevel(main)
            self.geomWindow.title("Geometry calibration")
            self.geomWindow.geometry("300x300")
            
            self.geomlabel = tk.Label(self.geomWindow, text="Enter mouse coordinates:")
            self.geomlabel.pack()
            
            self.geomlabel2 = tk.Label(self.geomWindow, text="x:")
            self.geomlabel2.pack()
            
            self.geomentryx = tk.Entry(self.geomWindow)
            self.geomentryx.pack()
            
            self.geomlabel3 = tk.Label(self.geomWindow, text="y:")
            self.geomlabel3.pack()

            self.geomentryy = tk.Entry(self.geomWindow)
            self.geomentryy.pack()
            
            self.geomlabel4 = tk.Label(self.geomWindow, text="Enter corresponding latitude/longitude:")
            self.geomlabel4.pack()
            
            self.geomlabel5 = tk.Label(self.geomWindow, text="Longitude:")
            self.geomlabel5.pack()
            
            self.geomentrylat = tk.Entry(self.geomWindow)
            self.geomentrylat.pack()
            
            self.geomlabel6 = tk.Label(self.geomWindow, text="Latitude:")
            self.geomlabel6.pack()

            self.geomentrylon = tk.Entry(self.geomWindow)
            self.geomentrylon.pack()
            
            self.geomsubmit = tk.Button(self.geomWindow, text="Submit")
            self.geomsubmit.pack()

        
        self.geom_button = tk.Button(
                        self.bottom_right,
                        text="Geometry calibration",
                        command=geomWindow
                    )
        self.geom_button.pack()

        self.col_button = tk.Button(
                        self.bottom_right,
                        text="Scale colours",
                        # command=self.colours()
                    )
        self.col_button.pack()

        
        
        def change_pic(i):
            self.canvas.itemconfig(self.canvas_img, image=self.imgtks[i])


        self.buttons = []
        for i in range(n_frames):
            button = tk.Button(
                self.top_right,
                text="%s" %times[i],
                command=lambda i=i: change_pic(i)
            )
            button.pack()
            self.buttons.append(button)

        def motion(event):
            x, y = event.x, event.y
            self.labeltext.set('Mouse coordinates: {}, {}'.format(x, y))
            # print()

        main.bind('<Motion>', motion)



# root.grid_rowconfigure(1, weight=1)
# root.grid_columnconfigure(0, weight=1)




# A root window for displaying objects
root = tk.Tk()
root.title("Mars Express VMC images of a cloud at Arsia Mons")
root.geometry("1600x800")

main = MainWindow(root, frames, times) #must assign to a variable otherwise garbage collected!
root.mainloop() # Start the GUI





