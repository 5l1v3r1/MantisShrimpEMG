# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 11:39:41 2018

@author: Dell
"""

def speed_plot(plt, pausetime=0.600):
    plt.draw()
    #plt.show()
    plt.pause(pausetime)


    # plots whole data plus the regions you want to plot in another plot
def home_plot(plt, data, audio=0, x_vals=0, y_vals=0, color='r', title=''):
    plt.clf()
    plt.plot(data)
    if type(audio) != int: # audio can be np.ndarray
        plt.plot( audio + 10000)

    plt.plot(x_vals, y_vals, color=color)
    plt.title(title)
    speed_plot(plt)

def home_plot_many(plt, data_list, color='g'):
    for elem in data_list:
        plt.plot(elem)

    speed_plot(plt)
