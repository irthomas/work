# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:14:14 2023

@author: iant
"""


# import numpy as np
# import matplotlib.pyplot as plt


# a = np.arange(100)
# b = np.random.rand(100)

# plt.plot(a, b)
# plt.show()


import sys
from PyQt5 import QtWidgets

app = QtWidgets.QApplication(sys.argv)
windows = QtWidgets.QWidget()

windows.resize(500,500)
windows.move(100,100)
windows.show()
sys.exit(app.exec_())