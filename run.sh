#!/bin/bash

###Documentation

##shows the output; Convolution and Sobel(X,Y and Combined)
python3 cons.py inp.png

##saves to 2 files: 'canny-plot.png' 'canny-final.png'
##also displays the canny-final, along with OpenCV [very similar results!]
##NOTE: different approach yet the same result for calculating Sobel (intermediary stage)
python3 canny.py inp.png

##shows the output: Hough Transformation
##caveat: could take up to 2 minutes
python3 hough.py inp.png

