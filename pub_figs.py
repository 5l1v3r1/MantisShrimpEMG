# -*- coding: utf-8 -*-

import json
import matplotlib.pyplot as plt

import pypaths.pypaths as pypaths
from tkinter import filedialog

import plotting
import numpy as np
from scipy.io.wavfile import read
from scipy.signal import butter, filtfilt

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

#%%
#(mantis shrimp data from 11.38.47, cricket from 13.38.41, cockroach from 17.03.52)
#%% Housekeeping
#%matplotlib qt
plt.clf() # clear current fig
pp = pypaths.Pypath() # initialize my class
#%%
analysis_path = ''
while (analysis_path[-1*len('-analysis.json'):] != '-analysis.json'):
    analysis_path = filedialog.askopenfilename(title='Select an -analysis.json') # , initialdir=path)

data_path = analysis_path[:-1 * len('-analysis.json')] + '.wav'

#open metaanalysis
analysis_path = pp.to_native(analysis_path)
json_data = json.load(open(analysis_path, "r"))
threshold = json_data['threshold']
sr = json_data['sr']
x_regions = json_data['x_regions']
timestamps = json_data['timestamps']
onset = json_data['onset']
offset = json_data['offset']
first_spike = json_data['first_spike']
spikes = json_data['spikes']

#open data raw
data_path = pp.to_native(data_path)
wav_file = read(data_path)
if wav_file[1].ndim == 2:
    audio = wav_file[1][:,1]
    raw_data = wav_file[1][:,0]
else:
    audio = 0 # equivalent to None here
    raw_data = wav_file[1]
b,a=butter(1,[100 / (.5*sr),600/(.5*sr)],btype='band')
filtered_data = filtfilt(b,a,raw_data)


def pretty_up(interval=0.001):
    # ticks
#    tick_range = np.arange(0, window_size, step= sr * interval) # sr * 0.001 is a ms, is 10 points
#    tick_labels = np.floor(np.arange(0, len(tick_range)) * interval * 1000)
#    tick_labels = tick_labels.astype(int)
#    plt.xticks(tick_range, tick_labels)
#    plt.xlabel('ms')

    # scalebar
    bar = AnchoredSizeBar(plt.gca().transData, sr * interval,
                          str(int(interval * 1000)) + ' ms', 4, frameon=False)
    plt.gca().add_artist(bar)

    plt.gca().axes.get_yaxis().set_visible(False) # removing y axis
    plt.gca().axes.get_xaxis().set_visible(False) # removing y axis
    plt.gca().spines['left'].set_visible(False) # remove frame...
    plt.gca().spines['right'].set_visible(False) # remove frame...
    plt.gca().spines['top'].set_visible(False) # remove frame...
    plt.gca().spines['bottom'].set_visible(False) # remove frame...


#%% For each distinct EMG event,
# i.e., number of x_regions
for i in range(len(json_data['spikes'])):
#
    x1 = x_regions[i][0]
    x2 = x_regions[i][1]

    #%% First, waveforms
    plt.figure()
    curr_spikes = np.array(json_data['spikes'][i][0])
    curr_spikes = curr_spikes[~np.isnan(curr_spikes).any(axis=1)] # removing any rows with nans in them
    WINDOW_SIZE = np.shape(curr_spikes)[1]
    numel_spikes = len(curr_spikes)
#    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.winter(np.linspace(0, 1, numel_spikes))))
#
#    for j in range(numel_spikes): # numel spikes
#        plt.plot(np.transpose(curr_spikes))
#
#    pretty_up()
#
#    savepath = analysis_path[:-1*len('.json')]+'-PUBLISHEDEMGwaveform-'+str(i+1)
#    plt.savefig((savepath + '.png'))
#    plt.close()
#
#
#    #%% Then, zoom out and show the artifact and audio traces
#    # artifact
#    a_reg_len =  x2-x1
#    plt.figure()
#    if sr != 44100: # ie, if a mantis not cockroach or cricket
#        artifact_region = filtered_data[x1:x1+a_reg_len*2]
#        plt.plot(artifact_region, color="black")
#        # plot each spike
#        offset = np.max(artifact_region) * 1.01
#        plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.winter(np.linspace(0, 1, numel_spikes))))
#        for j in range(len(timestamps[i])):
#            plt.plot(timestamps[i][j], offset, "*")#plt.plot(timestamps[i], [5000 for k in range(31)], "*")
#    else: # if is a cricket or cockroach
#        tozero = lambda ind: 0 if ind < 0 else ind # know I should make another one for the other side but I'm busy
#        artifact_region = filtered_data[tozero(x1-a_reg_len):x1+a_reg_len]
#        plt.plot(artifact_region, color="black")
#        offset = np.max(artifact_region) * 1.01
#        plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.winter(np.linspace(0, 1, numel_spikes))))
#        for j in range(len(timestamps[i])):
#            # note: I'm adjusting the indices because we are getting before and after as well.
#            plt.plot((timestamps[i][j] + a_reg_len) if x1-a_reg_len >= 0 else timestamps[i][j], offset, "*")
#    pretty_up(0.10)
#    savepath = analysis_path[:-1*len('.json')]+'-PUBLISHEDEMGartifact-'+str(i+1)
#    plt.savefig((savepath + '.png'))
#    plt.close()
#
#    #audio
#    if type(audio) != int: # if there is an audio trace
#        plt.figure()
#        audio_region = audio[x1:x1+a_reg_len * 2]
#        plt.plot(audio_region, color='m')
#        pretty_up(0.10)
#        savepath = analysis_path[:-1*len('.json')]+'-PUBLISHEDEMGaudio-'+str(i+1)
#        plt.savefig((savepath + '.png'))
#        plt.close()
#
#
    #%% Finally, entire x_region
    plt.figure()
    curr_filt = filtered_data[x1:x2]
    plt.plot(curr_filt, color="black")

    offset = np.max(curr_filt) * 1.1
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.winter(np.linspace(0, 1, numel_spikes))))
    for j in range(len(timestamps[i])):
        plt.plot(timestamps[i][j], offset, "*") # plt.plot(timestamps[i], [5000 for k in range(31)], "*")
    pretty_up(0.050)
    savepath = analysis_path[:-1*len('.json')]+'-PUBLISHEDEMGregion-'+str(i+1)
#    plt.savefig((savepath + '.png'))
#    plt.close()

    #%% histogram
    # make negative, relative to occurance.
#    ms = lambda int_or_array : int_or_array/sr * 1000
#
#    ts = np.sort(np.array(list(set(timestamps[i]))))
#    diff = np.diff(ts)
#    diff = ms(diff)
#    plt.hist(diff, bins = np.arange((max(diff) + 10), step=5), color = "black")
#    plt.gca().spines['right'].set_visible(False) # remove frame...
#    plt.gca().spines['top'].set_visible(False) # remove frame...
#    maxy = int(plt.gca().get_ybound()[1]) # yields tuple of lower and higher, gets second value, higher
#    # fixing the y axis
#    tick_range = np.arange(0, maxy+1)
#    tick_labels = np.arange(0, maxy+1)
#    tick_labels = tick_labels.astype(int)
#    plt.yticks(tick_range, tick_labels)
#
#    savepath = analysis_path[:-1*len('.json')]+'-PUBLISHEDEMGhisto' + str(i + 1)
#    plt.savefig((savepath + '.png'))
#    plt.close()
