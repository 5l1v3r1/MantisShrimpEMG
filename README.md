# MantisShrimpEMG
Suite for analysis of EMGs from invertebrates using the backyard brains data acquisition system


All .wav EMG files, .json data analysis files, and .png data visualization files, along with various excel files, can be found in the /data folder.
The three recordings used for Figure 3 are from days 7-23-18; recording 11.38.47 (mantis shrimp), 5-30-18; recording 13.38.41 (cricket), and 5-31-18; recording 
17.03.52 (cockroach).


A note for zooming in matplotlib window for selecting EMG regions, thresholding, or anything else that requires the user to click inside of a figure window: The program is waiting for one or two left clicks. If you select the button outlined in red here, you can zoom in and out by holding down the right mouse button. Experiment with this functionality. If you are selecting EMGs, you can hold down the right mouse button to pan. This will create a little red cross, indicating that if you left click in the figure again, it will zoom in. You can right-click to cancel this first data point, allowing you to eventually left-click select a region somewhere else. This way, you can pan across a recording unhindered. The pattern is left-click hold-> pan, right click.

If you're getting a threshold, you can only right-click zoom, which should be all the functionality you need. You usually don't need to pan when you threshold. Once you've zoomed in enough, left-click at you desired threshold.

![button](https://github.com/BackyardBrains/MantisShrimpEMG/blob/master/img/threshold_annotated.PNG)
