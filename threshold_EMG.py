# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 10:00:00 2018

@author: Dell
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pypaths.pypaths as pypaths
import plotting # my module
from scipy.io.wavfile import read
from scipy.signal import butter, filtfilt
from tkinter import filedialog
import sys

#%% Housekeeping
#%matplotlib qt
plt.close() # clear current fig
pp = pypaths.Pypath() # initialize my class

def try_threshold():
    xy = plt.ginput(1, timeout=-1)
    threshold = xy[0][1]
    plt.axhline(y=threshold, color = 'r', linestyle='-')
    plotting.speed_plot(plt)
    return threshold

def buffer(data_snip, nan_array):
    temp_region = np.concatenate((nan_array.copy(),
                                     data_snip,
                                     nan_array.copy()))
    return temp_region

def generator_threshold(input_data, curr_threshold):
    zip_gen = zip(list(range(len(input_data))),input_data) # a zip generator
    ids = list(zip_gen) # ids = indexed data snip
    # If threshold is below mean, must sort in other direction
    if np.nanmean(input_data) < curr_threshold:
        generator = ((i,v) if v > curr_threshold else (i, np.nan) for (i, v) in ids) # making thresholding generator
    else:
        generator = ((i,v) if v < curr_threshold else (i, np.nan) for (i, v) in ids) # making thresholding generator
    coords = list(generator) # a list of tuples
#    indices, vals = map(list, zip(*coords)) # unzipping list of tuples
    return map(list, zip(*coords)) # unzipping list of tuples

def run(analysis_path = ''):
    #%% Finding and opening the file
    #root = tk.Tk()
    #root.withdraw()

    # r'C:\Users\Dell\Documents\BYB\BYB_Recording_2018-07-06_13.05.15-analysis.txt'
     # ghetto regex, I'm pretty proud of it

    while (analysis_path[-1*len('-analysis.json'):] != '-analysis.json'):
        analysis_path = filedialog.askopenfilename(title='Select an -analysis.json') # , initialdir=path)

    data_path = analysis_path[:-1 * len('-analysis.json')] + '.wav'
    analysis_path = pp.to_native(analysis_path)
    data_path = pp.to_native(data_path)

    ''' Extracting x_regions tuples '''
    with open(analysis_path, "r") as read_file:
        json_data = json.load(read_file)

    x_regions = json_data['x_regions']


    '''Read raw data with scipy'''
    wav_file = read(data_path)

    sr = wav_file[0]
    # Filtering raw data

    if wav_file[1].ndim == 2:
        audio = wav_file[1][:,1]
        raw_data = wav_file[1][:,0]
    else:
        audio = 0 # equivalent to None here
        raw_data = wav_file[1]

    b,a=butter(1,[100 / (.5*sr),600/(.5*sr)],btype='band')
    filtered_data = filtfilt(b,a,raw_data)


    for emg_event in range(len(x_regions)):
        emg_bounds = x_regions[emg_event]
        data_snip = filtered_data[emg_bounds[0]:emg_bounds[1]]

        bound_size = emg_bounds[1] - emg_bounds[0]
        artifact_thres_region = raw_data[emg_bounds[0]:emg_bounds[1]+bound_size]

        if wav_file[1].ndim == 2:
            audio_thres_region = audio[emg_bounds[0]:emg_bounds[1]+bound_size]
        else:
            audio_thres_region = 0




        '''
                    DATA
        ------------------------------

                    X_REGIONS: data_snip
           ----------     ----------

                    SNIPPETS: to_add
            --  --          --   --

                    SPIKES: peaks
            .   .           .    .


        '''
        plotting.home_plot(plt, data_snip)

        ''' Thresholding '''
        threshold, cmd, indices, vals = 0, '', 0, 0 # instantiating in larger scope so they persist outside of while loop
        while cmd not in ['y', 'no data', 'elim']:
            threshold = try_threshold()
            indices, vals = generator_threshold(data_snip, threshold)
            # show the parts you are thresholding!
            plotting.home_plot(plt, data_snip, x_vals=indices, y_vals=vals)
            cmd = input('Is threshold ok? (y/n/no data)\n')
            if cmd == 'no data':
                continue # skips this loop b/c no data
            elif cmd == 'elim':
                xy=plt.ginput()
                # Here, I am making a list of tuples corresponding to values above threshold, nan else


        plt.close()
        # list((x,y),(x,y),(x,y),...,(x,y),(x,y))

        # We have the threshold, now we need to consolidate into spikes
        WINDOW_SIZE = int(sr * 0.020) # number of points that encapsulates a spike
        nan_array = np.empty(WINDOW_SIZE)
        nan_array[:] = np.nan
        spikes = nan_array.copy() # horizontal stack of nans, will eliminate shortly
        timestamps = []

        found_first_spike = False
        for i in indices:
            if ~np.isnan(vals[i]) and i > 0 and np.isnan(vals[i-1]):

                # Getting the first spike of a sequence
                # INDEX OF SPIKE WITHIN THIS DATA SNIPPET:
                if ~found_first_spike:
                    first_spike = i
                    found_first_spike = True

                # Getting spike waveforms
                if len(indices) < WINDOW_SIZE:
                    to_add = np.empty((1,WINDOW_SIZE))
                    to_add[:] = np.nan
                else:
                    temp_region = buffer(data_snip, nan_array)
                    # index_adjust is CRUCIAL
                    index_adjust = len(nan_array)
                    # note: peak is local to the slice. We need to put it into "context"
                    if threshold > 0:
                        peak = np.argmax(temp_region[int(i-WINDOW_SIZE/2+index_adjust):int(i+WINDOW_SIZE/2+index_adjust)])
                    else:
                        peak = np.argmin(temp_region[int(i-WINDOW_SIZE/2+index_adjust):int(i+WINDOW_SIZE/2+index_adjust)])
                    # start of spike region in temp_region, putting into "context"
                    spike_reg_start = i - WINDOW_SIZE / 2
                    temp_reg_peak = spike_reg_start + peak
                    # It's too easy to double count, so let's just do this:
                    if temp_reg_peak not in timestamps:
                        timestamps.append(temp_reg_peak)
                        to_add = temp_region[int(temp_reg_peak-WINDOW_SIZE/2+index_adjust):
                            int(temp_reg_peak+WINDOW_SIZE/2+index_adjust)]

                spikes = np.vstack((spikes, to_add))

        #spikes = spikes[~np.all(spikes == np.nan, axis=1)] # remove np.nan (originally zero) rows

        plt.close()
        avg = np.nanmean(spikes, axis=0) # like np.average but ignores nans
        std = np.nanstd(spikes, axis=0)  # like np.std but ignores nans
        #https://stackoverflow.com/questions/21001088/how-to-add-different-graphs-as-an-inset-in-another-python-graph
        fig,ax1 = plt.subplots()
        tick_range = np.arange(0, WINDOW_SIZE, step= sr * .001) # sr * 0.001 is a ms, is 10 points
        tick_labels = np.floor(tick_range / 10)
        plt.xticks(tick_range, tick_labels)
        plt.xlabel('ms')

        ### Plotting data_snip
        left, bottom, width, height = [0.70, 0.65, 0.2, 0.2]
        ax2 = fig.add_axes([left, bottom, width, height])
        ax2.plot(data_snip)
        ax2.axhline(y=threshold, color = 'r', linestyle='-')
        #https://stackoverflow.com/questions/2176424/hiding-axis-text-in-matplotlib-plots
        ax2.axes.get_xaxis().set_visible(False)
        ax2.axes.get_yaxis().set_visible(False)

        ### Plotting full audio and artifact region
        left, bottom, width, height = [0.70, 0.15, 0.2, 0.2]
        ax3 = fig.add_axes([left, bottom, width, height])

        if type(audio) != 'int':
            ax3.plot(audio_thres_region)

        ax3.plot(artifact_thres_region)
        ax3.axes.get_xaxis().set_visible(False)
        ax3.axes.get_yaxis().set_visible(False)
        ### Plotting
        ax1.plot(np.transpose(spikes), color='grey') # replacing plt with ax1
        ax1.plot(avg, color='red')
        ax1.plot(avg+std, color='g')
        ax1.plot(avg-std, color='g')
        # Plot threshold info
        ax1.axhline(y=threshold, color = 'r', linestyle='-')
        # Title metadata
        ax1.set_title(str(len(spikes)) + ' spikes from ' + analysis_path.split('\\')[-1])

        plotting.speed_plot(plt)

        base_out_path = analysis_path[:-1*len('.json')]+'-EMGregion-'+str(emg_event+1)
        plt.savefig((base_out_path + '.png'))
        #%% Save JSON
        # Preallocate space in keys' lists if they are listy
        relevant_keys = ['spikes','first_spike','offset','onset','threshold', 'timestamps'] # irrelevant includes 'sr' b/c that doesn't change between trials
        for key in relevant_keys:
            if (key not in json_data) or (len(json_data[key]) != len(x_regions)):
                json_data[key] = [None] * len(x_regions)


        ### SPIKES
        json_data['spikes'][emg_event] = [spikes.tolist()] # to undo, np.array(spikes.tolist())


        # Here, if there isn't audio data, no need to get audio or artefact phase info
        if wav_file[1].ndim == 2:
            ### AUDIO
            # relative to the same point as first_spike
            # at index 'offset', a list of each event's offset (audio)

            plotting.home_plot(plt, audio_thres_region, title='Select threshold for audio') # we are going twice the length
            audio_threshold = try_threshold() # (x,y)
            aud_indices, aud_vals = generator_threshold(audio_thres_region, audio_threshold)
            for i in aud_indices:
                if ~np.isnan(aud_vals[i]) and i > 0 and np.isnan(aud_vals[i-1]):
                    json_data['offset'][emg_event] = i
                    plt.close()
                    break

            ### ARTIFACT
            # relative to the same point as first_spike
            # DRY, I know I know
            plotting.home_plot(plt, artifact_thres_region, title='Select threshold for artifact') # we are going twice the length
            artifact_threshold = try_threshold() # (x,y)

            art_indices, art_vals = generator_threshold(artifact_thres_region, artifact_threshold)
            for i in art_indices:
                if ~np.isnan(art_vals[i]) and i > 0 and np.isnan(art_vals[i-1]):
                    json_data['onset'][emg_event] = i
                    plt.close()
                    break
        else:
            json_data['offset'][emg_event] = 0
            json_data['onset'][emg_event] = 0



        ###
        json_data['first_spike'][emg_event] = first_spike # in points
        ###
        json_data['threshold'][emg_event] = threshold
        ###
        json_data['timestamps'][emg_event] = timestamps
        ###
        json_data['sr'] = sr

        with open(analysis_path, 'w') as write_file:
            json.dump(json_data, write_file, indent=4)

if __name__ == "__main__":
    run()