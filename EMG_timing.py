# -*- coding: utf-8 -*-

import json
import matplotlib.pyplot as plt
import numpy as np

import pypaths.pypaths as pypaths
from tkinter import filedialog
import threshold_EMG

### functions
def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

stringify = lambda exp : str(truncate(ms(exp), 3))

### Housekeeping
plt.close()
pp = pypaths.Pypath()

#root = tk.Tk()
#root.withdraw()

# r'C:\Users\Dell\Documents\BYB\BYB_Recording_2018-07-06_13.05.15-analysis.txt'

analysis_path = ''
while (analysis_path[-1*len('-analysis.json'):] != '-analysis.json'):
    analysis_path = filedialog.askopenfilename(title='Select an -analysis.json') # , initialdir=path)

data_path = analysis_path[:-1 * len('-analysis.json')] + '.wav'
analysis_path = pp.to_native(analysis_path)
data_path = pp.to_native(data_path)

json_data = json.load(open(analysis_path, "r"))

if 'timestamps' not in json_data.keys():
	print('need to redo this one')
	threshold_EMG.run(analysis_path)
	json_data = json.load(open(analysis_path, "r"))

###
for trial in range(len(json_data['spikes'])):
	timestamps= np.sort(np.array(list(set(json_data['timestamps'][trial]))))
	diff = np.diff(timestamps)
	diff = ms(diff)
	plt.hist(diff, bins = np.arange((max(diff) + 10), step=5))
	title = ('EMG event length: '
		+ stringify(timestamps[-1] - timestamps[0])
		+ ' ms')

	title += (', b/h length: '
		+ stringify(json_data['offset'][trial] - json_data['onset'][trial])
		+ ' ms')

	plt.title(title)


	base_out_path = analysis_path[:-1*len('.json')]+'-histo, region ' + str(trial + 1)
	plt.savefig((base_out_path + '.png'))
	plt.close()
	print(base_out_path)
