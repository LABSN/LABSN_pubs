# -*- coding: utf-8 -*-
"""
============================
Script 'Get Sound Durations'
============================

This script opens a bunch of wav files and writes out their durations to a
JSON file.
"""
# Author: Dan McCloy <drmccloy@uw.edu>
#
# License: BSD (3-clause)

import json
import os.path as op
from glob import glob
from scipy.io import wavfile as wav


indir = '/home/dan/Documents/experiments/drmccloy/divAttnSemantic/rawRecordings/monotonized'
outfile = '/home/dan/Documents/experiments/drmccloy/divAttnSemantic/analysis/wordDurations.json'

stim_files = glob(op.join(indir, '*.wav'))
word_dict = {}
fs_dict = {}
dur_dict = {}
for f in stim_files:
    name = op.splitext(op.basename(f))[0]
    fs, data = wav.read(op.join(indir, f))
    word_dict[name] = data
    fs_dict[name] = fs

for name in word_dict.keys():
    dur_dict[name] = len(word_dict[name]) / float(fs_dict[name])

with open(outfile, 'w') as f:
    json.dump(dur_dict, f)
