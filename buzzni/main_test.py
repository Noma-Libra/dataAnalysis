
"""

"""
import os
import pandas as pd
import datetime
import sys
from tkinter import *
import tkinter.scrolledtext as tkst
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'log')

userlog_1907 = pd.read_csv(os.path.join(DATA_DIR, 'log_1907.csv'))

record = pd.DataFrame(userlog_1907[userlog_1907.hash == keyword],
                      columns=[
    'idx', 'date', 'hash', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'Y'
]).sort_values(by=['date'])

record_list = record.values.tolist()
