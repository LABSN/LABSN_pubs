"""
=============================
Script 'Calculate mean pitch'
=============================

This script loads all the headerless tab-delimited text files in a folder and
finds the filewise means and grand mean.
"""
# Author: Dan McCloy <drmccloy@uw.edu>
#
# License: BSD (3-clause)

import csv
from glob import glob
import os.path as op
import numpy as np

file_dir = 'pitchDataFromManip'
files = glob(op.join(file_dir, '*.tab'))

fname = []
time = []
pitch = []
mean_pitches = []

for f in files:
    file_pitch = []
    with open(f, 'r') as table:
        reader = csv.reader(table, delimiter='\t')
        for t, p in reader:
            #time.append(float(t))
            pitch.append(float(p))
            file_pitch.append(float(p))
            #fname.append(op.splitext(op.basename(f))[0])
    mean_pitches.append(np.mean(file_pitch))

print np.mean(pitch)
print np.mean(mean_pitches)
