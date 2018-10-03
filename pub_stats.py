# -*- coding: utf-8 -*-

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')

xlfile = pd.ExcelFile(r"E:\BYB\dataannotations.xlsx")
dfs = pd.read_excel(xlfile, sheet_name="Sheet1")
# note: dfs.loc[dfs['notes']=='underwater', ['title']]
# returns df containing only rows of the title column for which the notes column reads
# 'underwater'
categories = ['cockroach', 'Mantis Shrimp', 'cricket']
#files = dfs.loc[dfs['include']=='yes', ['title', 'include']].values.tolist() # returns list([['asdf'],['asdf'],['asdf']])
 # returns list([['asdf', ' Mantis Shrimp'],['asdf', 'Mantis Shrimp'],['asdf', 'Mantis Shrimp']])
def run_analysis_df(organism='cockroach'):
    # init lists for analysis
    histo_data = []
    numspikes = []
    cocontraction_durations = []
    gap_phase_duration = []
    initial_hundred = []
    final_hundred = []
    ms = lambda int_or_array : int_or_array/sr * 1000
    files = dfs.loc[(dfs['organism']==organism) & ~dfs['include'].isnull(), ['title', 'include']].values.tolist()
#    print(files)
    for metainfo in files:
        # need to chop and screw this piece so it reads the right piece
        
        path = metainfo[0][:-1 * len('.wav')] + '-analysis.json'
        path = 'E:\\BYB\\' + path.split('\\')[-1]
        # casting as string bc sometimes it is interpreted as an int.
        trials = str(metainfo[1]).split() # returns list of POSITIONS of relevant trials. decrement for index
        jsondata = json.load(open(path, 'r'))
#        print(metainfo)
        sr = jsondata['sr']
        for indexstr in trials:
            index= int(indexstr)-1
            # for big ass ISI histogram
            ts = ms(np.sort(np.array(list(set(jsondata['timestamps'][index])))))
            diff = np.diff(ts)
            histo_data = np.append(histo_data, diff)
##            ###******************************************* For making nice black histos
#            plt.hist(diff, bins = np.arange((max(diff) + 10), step=5), color = "grey")
#            plt.gca().spines['right'].set_visible(False) # remove frame...
#            plt.gca().spines['top'].set_visible(False) # remove frame...
#            maxy = int(plt.gca().get_ybound()[1]) # yields tuple of lower and higher, gets second value, higher
#            # fixing the y axis
#            tick_range = np.arange(0, maxy+1)
#            tick_labels = np.arange(0, maxy+1)
#            tick_labels = tick_labels.astype(int)
#            plt.yticks(tick_range, tick_labels)
#
#            savepath = path[:-1*len('.json')]+'-BLACKPUBLISHEDEMGhisto' + indexstr
#            plt.savefig((savepath + '.png'))
#            plt.close()
##            ###*********************************************
            # cocontraction duration
            cocontraction_durations.append(ms(jsondata['timestamps'][index][-1] - jsondata['timestamps'][index][0]))
            # num spikes in cocontraction
            numspikes.append(len(ts))
            # initial and final hundred milliseconds of coactivation phase, numspikes

            initial_hundred = np.append(initial_hundred, len(np.where(ts < int(ts[-1]/2))[0])) # initial_hundred = np.append(initial_hundred, len(ts[ts<100]))
            final_hundred = np.append(final_hundred, len(np.where(ts > int(ts[-1]/2))[0])) # final_hundred = np.append(final_hundred,len(ts[ts>ts[-1]-100]))

            # gap phase duration
            if organism == 'Mantis Shrimp':
                gap_phase_duration = np.append(gap_phase_duration, ms(jsondata['onset'][index]) - ts[-1])

    return (histo_data, numspikes, cocontraction_durations, gap_phase_duration, initial_hundred, final_hundred)
ns_cc_df = pd.DataFrame()
hd_df = pd.DataFrame()
gpd_df = pd.DataFrame()
if_df = pd.DataFrame() # initial/final df

# setting up dataframes for analysis
(hd, ns, cc, gpd, ih, fh) = run_analysis_df('Mantis Shrimp')
ns_cc_df = pd.concat([ns_cc_df, pd.DataFrame({'organism': 'Mantis Shrimp', 'cc':cc,'ns':ns,})])
hd_df = pd.concat([hd_df, pd.DataFrame({'organism': 'Mantis Shrimp', 'hd':hd})])
gpd_df = pd.concat([gpd_df, pd.DataFrame({'organism': 'Mantis Shrimp', 'gpd':gpd})])
if_df = pd.concat([if_df, pd.DataFrame({'organism': 'Mantis Shrimp', 'period': 'first half',
                                        'numsp':ih, 'ind':list(range(len(ih)))})])
if_df = pd.concat([if_df, pd.DataFrame({'organism': 'Mantis Shrimp', 'period': 'second half',
                                        'numsp':fh, 'ind':list(range(len(ih)))})])
# So it only plots all points in pointplot if we say hue=ind. But if ind is the same range 
# for all things, then it things all points are the same color. They shouldn't 
# be the same color, so we need to change ind.
# this is an annoying workaround so that the ind variables are different for each damn organism so the hues look different.

ms_ih = len(ih)
    
(hd, ns, cc, gpd, ih, fh) = run_analysis_df('cricket')
ns_cc_df = pd.concat([ns_cc_df, pd.DataFrame({'organism': 'cricket', 'cc':cc,'ns':ns,})])
hd_df = pd.concat([hd_df, pd.DataFrame({'organism': 'cricket', 'hd':hd})])
gpd_df = pd.concat([gpd_df, pd.DataFrame({'organism': 'cricket', 'gpd':gpd})])
if_df = pd.concat([if_df, pd.DataFrame({'organism': 'cricket', 'period': 'first half',
                                        'numsp':ih, 'ind':list(range(ms_ih, len(ih)+ms_ih))})])
if_df = pd.concat([if_df, pd.DataFrame({'organism': 'cricket', 'period': 'second half',
                                        'numsp':fh, 'ind':list(range(ms_ih, len(ih)+ms_ih))})])
cricket_ih = len(ih) + ms_ih
    
(hd, ns, cc, gpd, ih, fh) = run_analysis_df('cockroach')
ns_cc_df = pd.concat([ns_cc_df, pd.DataFrame({'organism': 'cockroach', 'cc':cc,'ns':ns,})])
hd_df = pd.concat([hd_df, pd.DataFrame({'organism': 'cockroach', 'hd':hd})])
gpd_df = pd.concat([gpd_df, pd.DataFrame({'organism': 'cockroach', 'gpd':gpd})])
if_df = pd.concat([if_df, pd.DataFrame({'organism': 'cockroach', 'period': 'first half',
                                        'numsp':ih, 'ind':list(range(cricket_ih, len(ih)+cricket_ih))})])
if_df = pd.concat([if_df, pd.DataFrame({'organism': 'cockroach', 'period': 'second half',
                                        'numsp':fh, 'ind':list(range(cricket_ih, len(ih)+cricket_ih))})])

def pretty_up(axis, title='', y_axis = ''):
    axis.spines['left'].set_visible(False) # remove frame...
    axis.spines['right'].set_visible(False) # remove frame...
    axis.spines['top'].set_visible(False) # remove frame...
    axis.spines['bottom'].set_visible(False) # remove frame...
    axis.set_ylabel(y_axis)
    axis.set_xlabel('')
    axis.tick_params(axis='x',          # changes apply to the x-axis
                     which='both',      # both major and minor ticks are affected
                     bottom=False,      # ticks along the bottom edge are off
                     top=False)         # ticks along the top edge are off

# make dataframe for analysis
#ax3 = sns.swarmplot(x='species', y='cc', color="0.2", data=interspec)
#ax3.set_title('Number of spikes, final hundred ms')
#pretty_up(ax3, title='', y_axis='spikes')

##
#ax = sns.boxplot(x='organism', y='cc', width=0.4, data=ns_cc_df)
#ax = sns.swarmplot(x='organism', y='cc', data=ns_cc_df, color='.02')
#ax.set_title('burst durations (ms)')
#pretty_up(ax, title='', y_axis='ms')
#
#ax1 = sns.boxplot(x='organism', y='ns', width=0.4, data=ns_cc_df)
#ax1 = sns.swarmplot(x='organism', y='ns', color="0.2", data=ns_cc_df)
#ax1.set_title('Number of spikes per burst')
#pretty_up(ax1, title='', y_axis='spikes')

#ax2 = sns.boxplot(x='organism', y='ih', width=0.4, data=ih_df)
#ax2 = sns.swarmplot(x='organism', y='ih', color="0.2", data=ih_df)
#ax2.set_title('Number of spikes, initial hundred ms')
#pretty_up(ax2, y_axis='spikes')
#
#ax3 = sns.boxplot(x='organism', y='fh', width=0.4, data=fh_df)
#ax3 = sns.swarmplot(x='organism', y='fh', color="0.2", data=fh_df)
#ax3.set_title('Number of spikes, final hundred ms')
#pretty_up(ax3, title='', y_axis='spikes')

# histogram SUM
#for org in ['cockroach', 'cricket', 'Mantis Shrimp']:
#
#    org_diff = hd_df.loc[hd_df['organism']==org, ['hd']].values
#    org_diff = list(map(lambda x: x[0], org_diff))
#    plt.figure()
#    plt.hist(org_diff, bins = np.arange((max(org_diff) + 10), step=1), color = "grey")
#    plt.gca().spines['right'].set_visible(False) # remove frame...
#    plt.gca().spines['top'].set_visible(False) # remove frame...
#    maxy = int(plt.gca().get_ybound()[1]) # yields tuple of lower and higher, gets second value, higher
#    # fixing the y axis
#    tick_range = np.arange(0, maxy+1, 10)
#    tick_labels = np.arange(0, maxy+1, 10)
#    tick_labels = tick_labels.astype(int)
#    plt.yticks(tick_range, tick_labels)
#    savepath = "C:\\Users\\Dell\\Documents\\BYB\\sum_histo_" + org + "_full.png"
#    plt.savefig(savepath)
#
#    plt.title(org + ' ISI')
#    rightbound = plt.gca().get_xbound()[1]
#    plt.xlim(0, 35)
#
#    savepath = "C:\\Users\\Dell\\Documents\\BYB\\sum_histo_" + org + ".png"
#    plt.savefig(savepath)


#interspec = pd.DataFrame({'species': 'Mantis Shrimp', 'cc':cc,'ns':ns,})

asl = lambda a, s: [a-s, a, a+s]

avg1, avg2, avg3, avg4, avg5, avg6 = 370, 243, 375, 383, 248, 376
std1, std2, std3, std4, std5, std6 = 84, 130, 37, 82, 58, 83
# total number of extensor spikes during cocontraction
us1, us2, us3, us4, us5, us6 = 46.1, 13.4, 30.0, 68.8, 20.5, 22.3 # avg spikes
ss1, ss2, ss3, ss4, ss5, ss6 = 14.7, 8.5, 4.3, 14.5, 5.9, 4.1 # std spikes

meta_df = pd.DataFrame()
# cocontraction duration
ms_cc_data = list(map(lambda x: x[0], # wrangling so it comes in the right shape
              np.array(ns_cc_df.loc[ns_cc_df['organism']=='Mantis Shrimp',['cc']])))
# numspikes
ms_ns_data = list(map(lambda x: x[0], # wrangling so it comes in the right shape
              np.array(ns_cc_df.loc[ns_cc_df['organism']=='Mantis Shrimp',['ns']])))
# adding data to df
meta_df = pd.concat([meta_df, pd.DataFrame(
        {'Individual':'0','Species':'Gonodactylus smithii', 'Duration': ms_cc_data,
         'Extensor spikes': ms_ns_data})])
meta_df = pd.concat([meta_df, pd.DataFrame(
        {'Individual':'1','Species':'Neogonodactylus bredini', 'Duration': asl(avg1, std1),
         'Extensor spikes':asl(us1, ss1)})])
meta_df = pd.concat([meta_df, pd.DataFrame(
        {'Individual':'2','Species':'Neogonodactylus bredini', 'Duration': asl(avg2, std2),
         'Extensor spikes':asl(us2, ss2)})])
meta_df = pd.concat([meta_df, pd.DataFrame(
        {'Individual':'3','Species':'Neogonodactylus bredini', 'Duration': asl(avg3, std3),
         'Extensor spikes':asl(us3, ss3)})])
meta_df = pd.concat([meta_df, pd.DataFrame(
        {'Individual':'4','Species':'Neogonodactylus bredini', 'Duration': asl(avg4, std4),
         'Extensor spikes':asl(us4, ss4)})])
meta_df = pd.concat([meta_df, pd.DataFrame(
        {'Individual':'5','Species':'Neogonodactylus bredini', 'Duration': asl(avg5, std5),
         'Extensor spikes':asl(us5, ss5)})])
meta_df = pd.concat([meta_df, pd.DataFrame(
        {'Individual':'6','Species':'Neogonodactylus bredini', 'Duration': asl(avg6, std6),
         'Extensor spikes':asl(us6, ss6)})])

# Make plots
plt.figure()
ax23 = sns.pointplot( x='Individual', y='Duration', hue='Species', join=False, sharex=True, data=meta_df, legend=False)
ax23.set(xticklabels=[])
ax23.legend_.remove()
ax23.set(title='Cocontraction durations between species')
plt.figure()
ax24 = sns.pointplot( x='Individual', y='Extensor spikes', hue='Species', join=False, sharex=True, data=meta_df)
ax24.set(xticklabels=[])
ax24.set(title='Number of spikes')

#meta_df  =pd.DataFrame({
#        'xbar': [370, 243, 375, 383,
#                 248, 376],
#        'std' : [84, 130, 37, 82, 58, 83]})
#i can do this
#i wil do this
#everything wil be good


## lineplot ACROSS TAXA
ax6 = sns.catplot(x='period', y='numsp', hue='ind',
                  kind='point', col='organism', capsize=0.1, data=if_df, legend=False)
#pretty_up(ax6, title='', y_axis='Number of spikes')



