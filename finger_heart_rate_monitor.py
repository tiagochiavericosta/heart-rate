# -*- coding: utf-8 -*-
"""
Heart Rate Monitor

Simple demonstration of calculating the heart rate using a webcam. Place a
finger on the camera and shine a light through it.

If possible, adjust webcam settings to a high frame rate (20-30 is good). On
linux, guvcview can be used for this. To get good results the finger and
light source must be held very still.

Created on Fri May  3 15:35:20 2013

@author: John Quinn <jquinn@cit.ac.ug>
"""

import cv2
import numpy as np
import time

N = 100
values = [0]*N
timestamps = [None]*N

# valores antigos
window_height = 500
window_width = 500
#window_height, window_width , channels = img.shape
#print height, width, channels

chart_l = 0
chart_r = window_width
chart_t = window_height/2
chart_b = window_height
old_bpm = 0
min_bpm = 30
text_color = (255,255,255)
line_color_red = (0,0,255)
line_color_green = (0,255,0)
line_color_blue = (255,0,0)
try:
    # The zero refers to the first installed webcam, adjust to select between
    # multiple installed webcams if needed.
    capture = cv2.VideoCapture(1)
    # The following operation rarely seems to be supported, probably have to
    # set this with some external software.
    capture.set(cv2.CAP_PROP_FPS, 20)
    
    # new if that's opencv 2.4.8 - true. the cv2.cv module is no more available.
    # the constant you're looking for is: cv2.CAP_PROP_FPS

    capture.set(cv2.CAP_PROP_FPS, 20)

    while True:
        ret, frame = capture.read()

        # Find the mean intensity
        currentval = np.mean(frame)

        # Add the next value and timestamp on to the queue and discard
        # the oldest ones.
        values.append(currentval)
        values.pop(0)
        timestamps.append(time.time())
        timestamps.pop(0)

        # Calculate the heart rate by looking at the peak of the power
        # spectrum.
        x = np.array(values)
        x = x - np.mean(x)
        if timestamps[0]:
            nfft = 512
            spectrum = np.abs(np.fft.fft(x,nfft)[:nfft/8])
            delta = (timestamps[-1]-timestamps[0])/N
            beats_per_second = spectrum.argmax()/(delta*nfft)
            beats_per_minute = 60*beats_per_second
            # If the calculated value is less than min_bpm then it's
            # probably picking up noise rather than a genuine pulse.
            if beats_per_minute > min_bpm:
                if old_bpm==0:
                    old_bpm = beats_per_minute
                # Easy approximation to a Kalman filter: keep a running
                # average. This improves accuracy since we know that the
                # physiology cannot change very quickly.
                beats_per_minute = .1*beats_per_minute + .9*old_bpm
                old_bpm = beats_per_minute
        else:
            beats_per_minute = 0

        # Show a thumbnail of the current webcam input (if the position and
        # lighting are correct, it should be possible to see the pulse).
        view = np.zeros((window_height, window_width, 3),np.uint8)
        thumbnail = cv2.resize(frame, (window_height/2, window_width/2))
        view[:window_height/2,window_width/2:,:] = thumbnail

        # Plot the last N intensity values
        axis_min = np.min(x)
        axis_max = np.max(x)
        coords = np.zeros((N,2),np.int)
        coords[:,0] = chart_r*np.arange(N)/N
        coords[:,1] = -1*(chart_b-chart_t)*np.array(x)/(axis_max-axis_min)
        coords[:,1] += chart_t - np.min(coords[:,1])
        for segment in range(N-1):
            cv2.line(view,tuple(coords[segment,:]),
                     tuple(coords[segment+1,:]),
                     line_color_red,2)

        # Plot the heart rate text
        cv2.putText(view, 'bpm', (30, 70),
            cv2.FONT_HERSHEY_DUPLEX, 1, text_color,
            thickness=1)
        if beats_per_minute > min_bpm:
            bpm_text = '%d' % (beats_per_minute)
        else:
            bpm_text = '-'
        cv2.putText(view, bpm_text, (50, 170),
            cv2.FONT_HERSHEY_DUPLEX, 3.0, text_color,
            thickness=2)

        cv2.imshow('Heart Rate Monitor', view)

        # Escape key to quit
        if cv2.waitKey(33)==27:
            break

finally:
    capture.release()
    cv2.destroyAllWindows()
