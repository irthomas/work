# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 14:37:53 2023

@author: iant


FUNCTIONS TO SAVE AND LOAD INTERACTIVE FIGURE WINDOWS
"""

import pickle


def save_ifig(fig, filepath):
    """save the figure to a pkl file"""
    with open(filepath, "wb") as f:
        pickle.dump(fig, f)


def load_ifig(filepath):
    """load the figure from a pkl file"""
    with open(filepath, "rb") as f:
        fig = pickle.load(f)
        return fig


# open dialog box to select pickle
if __name__ == "__main__":
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename
    root = Tk()
    root.wm_attributes('-topmost', 1)
    root.withdraw()
    filepath = askopenfilename(parent=root)
    root.update()

    print("Loading", filepath)
    load_ifig(filepath)
