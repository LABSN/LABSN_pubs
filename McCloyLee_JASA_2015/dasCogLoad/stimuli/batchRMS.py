# -*- coding: utf-8 -*-
"""
=============================
Script 'DAS-cog-load stimuli'
=============================

This script makes spatially-distributed word streams.
"""
# Author: Dan McCloy <drmccloy@uw.edu>
#
# License: BSD (3-clause)

import os.path as op
from glob import glob
from expyfun.stimuli import rms, read_wav, write_wav

indir = 'monotonizedWords'
outdir = 'normalizedWords'
target_rms = 0.01

files = glob(op.join(indir, '*.wav'))

for f in files:
    fname = op.split(f)[-1]
    wav, fs = read_wav(f)
    new_wav = wav * target_rms / rms(wav)
    write_wav(op.join(outdir, fname), new_wav, fs)
