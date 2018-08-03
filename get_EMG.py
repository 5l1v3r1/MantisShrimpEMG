
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 13:02:01 2018

@author: Dell
"""

import plotting
from scipy.io.wavfile import read
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pypaths.pypaths as pypaths
from scipy.signal import butter, filtfilt
from tkinter import filedialog
import json

#%% Housekeeping
#%matplotlib qt
plt.clf() # clear current fig
pp = pypaths.Pypath() # initialize my class

#%% CLI functionality
# get path of file
path = filedialog.askopenfilename(title='')

path = pp.to_native(path)
wav_file = read(path)

sr = wav_file[0]


if wav_file[1].ndim == 2:
    audio = wav_file[1][:,1]
    data = wav_file[1][:,0]
else:
    audio = 0 # equivalent to None here
    data = wav_file[1]

b,a=butter(1,[100 / (.5*sr),600/(.5*sr)],btype='band')
data = filtfilt(b,a,data) # instead of calling it filtered data, I am seeing if this helps find spikes

# show whole dataset
plotting.home_plot(plt, data, audio=audio)


#%% iterative region acquisition
x_regions = [] # tuples of x values
x_vals = list(range(len(data)))
y_vals = 0
x1 = 0
x2 = 0
command = ''

while command != 'done':
    command = input('What would you like to do? done/snip/z/r\n')

    if  command == 'snip': # y for yes to SAVE, shows whole trace, runs to end of program
        if x1 != 0 and x2 != 0:
            x_regions.append((x1, x2))
        # plot whole data with the correct area highlighted
        plotting.home_plot(plt, data, audio=audio, x_vals=x_vals, y_vals=y_vals)
        plt.draw()
        plt.show()
        plt.pause(0.600)
        # Save Data
        json_data = {'x_regions': x_regions}
        db_path_name = path[:-4] + '-analysis.json'
        with open(db_path_name, 'w') as write_file:
            json.dump(json_data, write_file)
        #% Other commands
    elif command == 'r': # restart, show previous
        plotting.home_plot(plt, data, audio=audio, x_vals=x_vals, y_vals=y_vals)

    elif command == 'z': # for zoom in
        print('Flank EMG regions of interest')
        x = plt.ginput(2, timeout=-1)
        x1 = int(x[0][0])
        x2 = int(x[1][0])

        # make sure sorted
        if x1 > x2:
            temp = x1
            x1 = x2
            x2 = temp

        x_vals = list(range(x1, x2+1)) # section of x axis we want
        y_vals = data[x1:x2+1] # values corresponding to the section of x axis we want

        plt.close()
        plt.plot(x_vals, y_vals)
#        plt.plot(x_vals, (1000 + audio[x1:x2+1] / 100))
        tick_range = np.arange(x_vals[0], x_vals[-1], step= sr * 1)
        tick_labels = np.floor(tick_range / 10000)
        plt.xticks(tick_range, tick_labels)
        plt.title('file: ' + path.split('\\')[-1]
              + ', x range:' + str(x1)
              + ', ' + str(x2))
        plotting.speed_plot(plt)

    if len(x_regions) > 0:
        print('clicked x pairs', x_regions[-1]) # x = [(x, y), (x, y), ... , (x, y)]

print("you were just looking at: " + path)
#%

#plt.plot(rms * 5)
#plt.xticks(       np.arange(0, len(data), step= sr * 1), # odering of one second
#           1/sr * np.arange(0, len(data), step= sr * 1))
#plt.show()
