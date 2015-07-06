# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'Calculate dB masking for each word'
===============================================================================

This script XXX.
"""
# Author: Dan McCloy <drmccloy@uw.edu>
# License: BSD (3-clause)

import os.path as op
import numpy as np
import expyfun as ef
import matplotlib.pyplot as plt
#import scipy.stats as ss
np.set_printoptions(linewidth=150)

worddir = 'stimuli/normalizedWords'

vardict = np.load(op.join('processedData', 'expVariables.npz'))
onset_samp = vardict['tr_onset_samp']
words = vardict['tr_words']
attn = vardict['tr_attn']

allwords = np.unique(words)
output_fs = 24414.

'''
# TESTING
vardict = np.load(op.join('processedData', 'masking.npz'))
db_per_ear = vardict['db_per_ear']

word_max = np.max(db_per_ear, axis=-1)
word_max.shape = word_max.shape + (1,)
better_ear = db_per_ear == np.tile(word_max, 2)
db_better_ear = db_per_ear[better_ear]
db_min = np.min(db_better_ear)
db_max = np.max(db_better_ear)
bins = np.arange(np.floor(db_min), np.ceil(db_max))
plt.hist(db_better_ear, bins=bins)
raise RuntimeError()
'''

#%% READ IN WORD AUDIO
word_fs = dict()
word_audio = dict()
word_len = dict()
word_len[''] = 0
for word in allwords:
    samples, fs = ef.stimuli.read_wav(op.join(worddir, word + '.wav'))
    # resample as needed
    if not np.allclose(fs, output_fs):
        samples = ef.stimuli.resample(samples, output_fs / float(fs), 1)
    word_audio[word] = samples
    word_len[word] = samples.shape[-1]
    word_fs[word] = fs
assert len(set(word_fs.values())) == 1  # resampling anyway, so unnecessary
del (word_fs, samples)

#%% HRTF CONVOLUTION
""" Convolve each word with the 4 different HRTFs in advance, rather than
    convolving each stream after it's assembled (more efficient this way). Also
    convolve a dummy stream to capture any changes to number of samples caused
    by the convolution. The "13" (seconds) in trial_len_samples comes from the
    trial timecourse.
    """
trial_len_samples = np.ceil(13. * output_fs).astype(int)  # 14.25 - 1.25
trial_len_samples = ef.stimuli.convolve_hrtf(np.zeros(trial_len_samples),
                                             output_fs, 0).shape[-1]
angles = ["-60", "-15", "15", "60"]
hrtf_dict = {angle: dict() for angle in angles}
for word in allwords:
    for angle in angles:
        hrtf_dict[angle][word] = ef.stimuli.convolve_hrtf(word_audio[word],
                                                          output_fs,
                                                          int(angle),
                                                          'barb')
del (word_audio, word, angle)

#%% CONSTRUCT STIMS
audio_separate_streams = np.zeros((words.shape[0], 4, 2, trial_len_samples),
                                  dtype=float)
for tr_ix, (tr_words, tr_samps) in enumerate(zip(words, onset_samp)):
    stream_samples = np.zeros((4, 2, trial_len_samples), dtype=float)
    for str_ix, (str_words, str_samps) in enumerate(zip(tr_words, tr_samps)):
        for word_ix, (word, start) in enumerate(zip(str_words, str_samps)):
            l_chan, r_chan = hrtf_dict[angles[str_ix]][word]
            for chan_ix, chan in enumerate((l_chan, r_chan)):
                stop = start + len(chan)
                stream_samples[str_ix, chan_ix, start:stop] += chan
        # equalize mean combined energy of L/R channels across streams
        stream_rms = np.mean(ef.stimuli.rms(stream_samples[str_ix, :, :]))
        stream_samples[str_ix] *= 0.01 / stream_rms
    audio_separate_streams[tr_ix] = stream_samples

#%% CALC MASKING
db_per_ear = np.zeros(words.shape + (2,), dtype=float)
db_per_ear[:] = np.nan
for tr_ix, (tr_words, tr_samps, tr_attn) in \
        enumerate(zip(words, onset_samp, attn)):
    for str_ix, (str_words, str_samps, str_attn) in \
            enumerate(zip(tr_words, tr_samps, tr_attn)):
        if str_attn:
            for word_ix, (word, start) in enumerate(zip(str_words, str_samps)):
                word_samps = hrtf_dict[angles[str_ix]][word]
                word_len = word_samps.shape[-1]
                end = start + word_len
                window = audio_separate_streams[tr_ix, :, :, start:end]
                signal = window[str_ix]
                assert word_samps.shape == signal.shape
                masker = window.copy()
                masker[str_ix] = 0.
                masker = np.nansum(masker, axis=0)
                assert masker.shape == signal.shape
                signal_amp = ef.stimuli.rms(signal)
                masker_amp = ef.stimuli.rms(masker)
                db_per_ear[tr_ix, str_ix, word_ix] = \
                    20 * np.log10(signal_amp / masker_amp)

outvars = dict(db_per_ear=db_per_ear)
np.savez(op.join('processedData', 'masking.npz'), **outvars)
