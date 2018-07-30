# -*- coding: utf-8 -*-
'''
So, I'm trying to see if there are spectral characteristics I can filter out
'''
import json
import matplotlib.pyplot as plt
import pypaths.pypaths as pypaths
import plotting # my module
from scipy.io.wavfile import read
from scipy.signal import butter, filtfilt, periodogram
from tkinter import filedialog
import numpy as np
import re
import os

def freq_compare():
    #%% Housekeeping
    #%matplotlib qt
    plt.close() # clear current fig
    pp = pypaths.Pypath() # initialize my class

    #%% Finding and opening the file
    analysis_path = ''

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

    ### Quick function definition
    def plot_sexy(section, max_f=1200):
        # to be used with spikes and artifact
        plt.clf()
        plt.subplot(211)
        plt.plot(section)
        plt.title(analysis_path.split('\\')[-1])
        tick_range = np.arange(0, len(section), step= sr * .05) # sr * 0.001 is a ms, is 10 points
        tick_labels = (tick_range / 10).astype('int')
        plt.xlabel('ms')
        plt.xticks(tick_range, tick_labels)


        plt.subplot(212)
        f, Px_denn = periodogram(section, sr)
        f = f[f<max_f]
        Px_denn = Px_denn[0:len(f)]
        plt.plot(f, Px_denn)
        plt.title('Periodogram')
        plt.xlabel('Frequency (Hz)')
        plotting.speed_plot(plt)



    ### Filtering raw data
    if wav_file[1].ndim == 2:
        raw_data = wav_file[1][:,0]
    else:
        raw_data = wav_file[1]
    b,a=butter(1,[200 / (.5*sr),1200/(.5*sr)],btype='band')
    filtered_data = filtfilt(b,a,raw_data)

    data = filtered_data # raw_data # filtered_data

    cmd = input('get region or x region (' + str(len(x_regions)) + ' regions) (#/space/q)')
    while cmd not in ['q', 'quit']:

        if re.match('\d\d?', cmd):
            # note: must input position not index
            reg = x_regions[int(cmd)-1]
#            p = Process(target=plot_sexy(data[reg[0]:reg[1]]))
#            p.start()
#            p.join()
            x = (reg[0], reg[1])
            plot_sexy(data[x[0]:x[1]])
        elif cmd == 's':
            pathlist = analysis_path.split('\\')
            filename = pathlist[-1][:-1 * len('.json')] + '-periodogram_' + str(x[0]) + '-' + str(x[1]) + '.png'

            savepath = ''
            pathlist = pathlist[:-1]
            pathlist.append( filename)
            for elem in pathlist:
                savepath = savepath + elem + os.sep

            print(savepath[:-1])
            plt.savefig(savepath[:-1]) # cut off \ at end

        else:
            plotting.home_plot(plt, data)
            xy = plt.ginput(2, timeout=-1)

            plt.close()
#            p = Process(target=plot_sexy(data[int(xy[0][0]):int(xy[0][1])]))
#            p.start()
#            p.join()
            x = (int(xy[0][0]), int(xy[1][0]))
            if x[0] > x[1]:
                x = (x[1], x[0])

            plot_sexy(data[x[0]:x[1]])


        cmd = input('get region or x region (#/s/space/q)')



    print(analysis_path)
#    with open(analysis_path[-5]+'trace', 'wb') as fid:
#        pickle.dump(ax1, fid)
#    with open(analysis_path[-5]+'periodogram', 'wb') as fid:
#        pickle.dump(ax2, fid)

#    with open('myplot.pkl','rb') as fid:
#        ax = pickle.load(fid)
#        plt.show()




if __name__ == '__main__':
    freq_compare()