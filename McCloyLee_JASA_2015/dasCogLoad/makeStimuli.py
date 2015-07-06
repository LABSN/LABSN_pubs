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

import numpy as np
import os.path as op
from itertools import permutations
from scipy.signal import resample
from expyfun import stimuli as stim

wordlist = 'stimSelection/finalWordsAndCategories.tsv'
worddir = 'stimuli/normalizedWords'
stimdir = 'stimuli/finalStims'
outfile = 'processedData/expVariables.npz'


def reshape_sl(sl):
    """to restore the shape of a sliced array
    """
    return sl.reshape((sl.shape[0] // 2, -1) + sl.shape[1:])


# EXPERIMENT PARAMETERS
angles = [-60, -15, 15, 60]
trials = 120  # should be divisible by 60
waves = 12
streams = len(angles)
isi = 0.25  # onset-to-onset delay
output_fs = 24414.0625  # 44100.0
targs_per_trial = [2, 3]
foils_per_trial = [0, 1, 2]
total_per_trial = [3, 4]
write_wavs = True
randomize = True

# RANDOM NUMBER GENERATOR
rng = np.random.RandomState(0)

# READ IN RECORDED WORDS
word_cats = {}
word_wavs = {}
targ_words = []
word_fs = {}
print('Reading in WAVs and possibly resampling')
with open(wordlist, 'r') as wl:
    for line in wl:
        word, category = line.strip().split('\t')
        word_cats[word] = category
        samples, fs = stim.read_wav(op.join(worddir, word + '.wav'))
        num_samples = int(round(samples.shape[-1] * float(output_fs) / fs))
        word_wavs[word] = resample(samples, num_samples, axis=-1)
        word_fs[word] = fs
longest_word_letters = max([len(x) for x in word_cats.keys()])

# REVERSE DICTIONARY FROM word_cats
cat_words = {}
for word, cat in word_cats.items():
    if cat not in cat_words.keys():
        cat_words[cat] = [word]
    else:
        cat_words[cat].append(word)

# EXCLUDE THE LARGEST CATEGORIES FOR THIS EXPERIMENT
categories = cat_words.keys()
cat_sizes = {x: len(cat_words[x]) for x in categories}
large_cats = [x for x in categories if cat_sizes[x] > 6]
categories = list(set(categories) - set(large_cats))
for cat in large_cats:
    del cat_sizes[cat]
for word in word_cats.keys():
    if word_cats[word] in large_cats:
        targ_words.append(word)
        del word_cats[word]
sm_cats = [x for x in categories if cat_sizes[x] == 3]
md_cats = [x for x in categories if cat_sizes[x] == 6]

# REMOVE SOME TARGET WORDS DUE TO POLYSEMY OR NEIGHBORNESS (cf. teeth, tea)
words_to_remove = ('chest', 'trunk', 'teeth', 'tree', 'vine', 'root', 'sock',
                   'ear', 'seed', 'bud')
                   # 'thigh', 'leg', 'goat', 'pig'
targ_words = np.array([x for x in targ_words if x not in words_to_remove])

# WHAT ARE THE CATEGORIES?
tr_cat_size = np.repeat((3, 6), trials // 2)
tr_cats = np.array([sm_cats if x == 3 else md_cats for x in tr_cat_size])

# ATTENTIONAL CONDITIONS (WHICH STREAMS TO ATTEND)
print('Setting up experimental conditions')
sel = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
adj = [(1, 1, 0, 0), (0, 1, 1, 0), (0, 0, 1, 1)]
sep = [(1, 0, 1, 0), (0, 1, 0, 1), (1, 0, 0, 1)]
adj_dup = [(2, 2, 0, 0), (0, 2, 2, 0), (0, 0, 2, 2)]  # these trials will get
sep_dup = [(2, 0, 2, 0), (0, 2, 0, 2), (2, 0, 0, 2)]  # targ streams duplicated
trials_per_cond = trials // 2 // 5  # the 2 is for the 2 category sizes
sel = np.repeat(sel, trials_per_cond // len(sel), axis=0)
adj = np.repeat(adj, trials_per_cond // len(adj), axis=0)
sep = np.repeat(sep, trials_per_cond // len(sep), axis=0)
adj_dup = np.repeat(adj_dup, trials_per_cond // len(adj_dup), axis=0)
sep_dup = np.repeat(sep_dup, trials_per_cond // len(sep_dup), axis=0)
tr_attn = np.concatenate((sel, adj, sep, adj_dup, sep_dup), axis=0)
tr_attn = np.concatenate((tr_attn, tr_attn), axis=0)
assert tr_attn.shape[0] == trials

# DUPLICATE STREAMS AS APPROPRIATE
tr_duplicate_stream = np.tile((0, 1), trials // 2)  # duplicate which stream?
for idx, attn in enumerate(tr_attn):
    if sum(attn) == 4:
        #indices = np.array(zip(*np.where(attn)))  # idiom for > 1D
        indices = np.where(attn)[0]
        word_to_duplicate = tr_cats[idx][indices][tr_duplicate_stream[idx]]
        tr_cats[idx][indices] = word_to_duplicate

# CONSTRUCT WORD LISTS BY STREAM
dt = 'S' + str(longest_word_letters)
tr_words = np.empty(tr_cats.shape + (waves,), dtype=np.dtype(dt))
for rnum, row in enumerate(tr_cats):
    for cnum, col in enumerate(row):
        wds = cat_words[col]
        repeats = waves // len(wds)
        """ the hard way: at most two in a row of the same word """
        stream = rng.permutation(wds)
        for _ in range(repeats - 1):
            stream = np.concatenate((stream, rng.permutation(wds)), 0)
        """ the easy way: may yield three or four of same word in a row """
        #stream = rng.permutation(wds * repeats)
        tr_words[rnum, cnum, :] = stream

# POSSIBLE TARGET LOCATIONS (BY WAVE)
targs_pt = np.concatenate((targs_per_trial, targs_per_trial[::-1]))
tr_total = np.tile(total_per_trial, trials // len(total_per_trial))
tr_targs = np.tile(targs_pt, trials // len(targs_per_trial) // 2)
tr_foils = tr_total - tr_targs
possible = list(set(['1' * x + '2' * y + '0' * (waves - z - 4) for
                     x, y, z in zip(tr_targs, tr_foils, tr_total)]))
possible = [''.join(y) for x in possible for y in list(set(permutations(x)))]
possible = ['000' + x + '0' for x in possible if '11' not in x and '12' not
            in x and '21' not in x and '22' not in x]

# ASSIGN TARGET LOCATIONS
tr_targ_loc = np.zeros_like(tr_words, dtype=int)
tr_foil_loc = np.zeros_like(tr_words, dtype=int)
for tnum in range(trials):
    chosen_scheme = rng.choice([x for x in possible if
                                x.count('1') == tr_targs[tnum] and
                                x.count('2') == tr_foils[tnum]])
    chosen_scheme = [int(x) for x in list(chosen_scheme)]
    # if divided attn, make sure each attended stream gets at least one targ
    targ_indices = [x for x, y in enumerate(chosen_scheme) if y == 1]
    stream_indices = rng.permutation(np.tile([0, 1], 2)[:len(targ_indices)])
    for wnum, wave in enumerate(chosen_scheme):
        if wave == 1:  # a target wave
            if np.sum(tr_attn[tnum]) > 1:  # two attended streams
                attn = np.where(tr_attn[tnum])[0][stream_indices[targ_indices.index(wnum)]]
            else:
                attn = np.where(tr_attn[tnum])[0]  # one attended stream
            tr_targ_loc[tnum, attn, wnum] = 1
        elif wave == 2:  # a foil wave
            igno = rng.choice(np.where(np.logical_not(tr_attn[tnum]))[0])
            tr_foil_loc[tnum, igno, wnum] = 1
assert np.all(np.sum(tr_targ_loc + tr_foil_loc, axis=1) < 2)

# REPLACE STREAM WORDS WITH TARGET WORDS
targ_words = rng.permutation(targ_words)
targ_word_iter = np.nditer(targ_words)
replacements = tr_targ_loc + tr_foil_loc
for tnum in range(trials):
    for snum, stream in enumerate(replacements[tnum]):
        for wnum in range(len(stream)):
            if replacements[tnum, snum, wnum] == 1:
                try:
                    tr_words[tnum, snum, wnum] = targ_word_iter.next()
                except StopIteration:
                    targ_words = rng.permutation(targ_words)
                    targ_word_iter = np.nditer(targ_words)
                    tr_words[tnum, snum, wnum] = targ_word_iter.next()

# ASSIGN TIMING SLOTS
tr_onsets = np.zeros_like(tr_words, dtype=float)
wave_idx = np.repeat(np.atleast_2d(np.arange(waves)), streams, 0) * 4
order = np.zeros((waves, streams), dtype=int)
for tnum in range(trials):
    attn = np.where(tr_attn[tnum])[0]
    igno = np.where(np.logical_not(tr_attn[tnum]))[0]
    for wnum in range(waves):
        # always have attended streams go first
        if wnum == 0:
            attn_idx = np.arange(len(attn))
            igno_idx = np.arange(len(attn), streams)
            order[wnum, attn_idx] = attn
            order[wnum, igno_idx] = igno
        else:
            order[wnum, :] = rng.permutation(streams)
            pen = np.where(order[wnum - 1] == streams - 2)[0]
            ult = np.where(order[wnum - 1] == streams - 1)[0]
            fir = np.where(order[wnum] == 0)[0]
            sec = np.where(order[wnum] == 1)[0]
            while ult in (fir, sec) or pen == fir:
                order[wnum, :] = rng.permutation(streams)
                fir = np.where(order[wnum] == 0)[0]
                sec = np.where(order[wnum] == 1)[0]
    tr_onsets[tnum, :, :] = np.array(isi * (order.T + wave_idx))
assert np.min(np.diff(tr_onsets)) >= 0.75
tr_onset_samp = (tr_onsets * output_fs).round().astype(int)

# CONSTRUCT AUDIO STREAMS
print('Constructing audio streams')
longest_word_samples = np.max([x.shape[-1] for x in word_wavs.values()])
stream_len = np.max(tr_onset_samp) + longest_word_samples
tr_mono = np.zeros((trials, streams, stream_len), dtype=float)
for tnum in range(trials):
    for snum in range(streams):
        for wnum in range(waves):
            word = tr_words[tnum, snum, wnum]
            samps = word_wavs[word][0]
            onset = tr_onset_samp[tnum, snum, wnum]
            offset = onset + len(samps)
            tr_mono[tnum, snum, onset:offset] += samps
del word, samps, onset, offset

# HRTF CONVOLUTION
print('Convolving with HRTFs')
stream_len = stim.convolve_hrtf(np.zeros(stream_len), output_fs, 0).shape[-1]
tr_hrtf = np.zeros((trials, streams, 2, stream_len), dtype=float)
for tnum in range(trials):
    for snum in range(streams):
        tr_hrtf[tnum, snum] = stim.convolve_hrtf(tr_mono[tnum, snum],
                                                 output_fs, angles[snum])
# RENORMALIZE
print('Renormalizing')
tr_original_rms = stim.rms(tr_mono)
tr_convolved_rms = np.mean(stim.rms(tr_hrtf), axis=-1)
multiplier = tr_original_rms / tr_convolved_rms
tr_norm = (tr_hrtf.T * multiplier.T).T  # broadcasting
tr_norm_rms = np.mean(stim.rms(tr_norm), axis=-1)  # TODO: test RMS

# COMBINE L & R CHANNELS ACROSS STREAMS
tr_stim = np.sum(tr_norm, axis=1)
tr_stim_rms = np.mean(stim.rms(tr_stim), axis=-1)  # TODO: test RMS
assert tr_hrtf.shape[0] == tr_stim.shape[0]
assert tr_hrtf.shape[-2] == tr_stim.shape[-2]
assert tr_hrtf.shape[-1] == tr_stim.shape[-1]

# TRAINING STIMULI
print('Constructing training stimuli')
one_idx     = np.where((np.sum(tr_attn, axis=-1) == 1) & (tr_targs == 2) & (tr_cat_size == 3))
two_idx     = np.where((np.sum(tr_attn, axis=-1) == 2) & (tr_targs == 3) & (tr_foils == 1) & (tr_cat_size == 3))
four_a_idx  = np.where((np.sum(tr_attn, axis=-1) == 1) & (tr_targs == 2) & (tr_foils == 2) & (tr_cat_size == 3))
four_aa_idx = np.where((np.sum(tr_attn, axis=-1) == 4) & (tr_targs == 3) & (tr_foils == 1) & (tr_cat_size == 3))
four_ab_idx = np.where((np.sum(tr_attn, axis=-1) == 2) & (tr_targs == 3) & (tr_foils == 1) & (tr_cat_size == 3))

one_attn = tr_attn[one_idx]
two_attn = tr_attn[two_idx]
four_a_attn = tr_attn[four_a_idx]
four_aa_attn = tr_attn[four_aa_idx]
four_ab_attn = tr_attn[four_ab_idx]

one_stim = tr_norm[one_idx][np.where(one_attn)]  # 1 stream 1 attn
two_stim = np.sum(reshape_sl(tr_norm[two_idx][np.where(two_attn)]), 1)
four_a_stim = np.sum(tr_norm[four_a_idx], 1)    # 4 stream 1 attn
four_aa_stim = np.sum(tr_norm[four_aa_idx], 1)  # 4 stream 2 attn (same)
four_ab_stim = np.sum(tr_norm[four_ab_idx], 1)  # 4 stream 2 attn (diff)
for s in (one_stim, two_stim, four_a_stim, four_aa_stim, four_ab_stim):
    assert len(s.shape) == 3
    assert s.shape[-2:] == (2, stream_len)

one_words = tr_words[one_idx][np.where(one_attn)]
two_words = reshape_sl(tr_words[two_idx][np.where(two_attn)])
four_a_words = tr_words[four_a_idx]
four_aa_words = tr_words[four_aa_idx]
four_ab_words = tr_words[four_ab_idx]

one_targ_loc = tr_targ_loc[one_idx][np.where(tr_attn[one_idx])]
two_targ_loc = reshape_sl(tr_targ_loc[two_idx][np.where(tr_attn[two_idx])])
four_a_targ_loc = tr_targ_loc[four_a_idx]
four_aa_targ_loc = tr_targ_loc[four_aa_idx]
four_ab_targ_loc = tr_targ_loc[four_ab_idx]

one_foil_loc = tr_foil_loc[one_idx][np.where(np.logical_not(tr_attn[one_idx]))]
two_foil_loc = reshape_sl(tr_foil_loc[two_idx][np.where(np.logical_not(tr_attn[two_idx]))])
four_a_foil_loc = tr_foil_loc[four_a_idx]
four_aa_foil_loc = tr_foil_loc[four_aa_idx]
four_ab_foil_loc = tr_foil_loc[four_ab_idx]
for s in (two_words, two_targ_loc, four_a_words, four_a_targ_loc,
          four_aa_words, four_ab_words, four_aa_targ_loc, four_ab_targ_loc):
    assert len(s.shape) == 3
    assert s.shape[-1] == waves

one_onset = tr_onsets[one_idx][np.where(one_attn)]
two_onset = reshape_sl(tr_onsets[two_idx][np.where(two_attn)])
four_a_onset = tr_onsets[four_a_idx]
four_aa_onset = tr_onsets[four_aa_idx]
four_ab_onset = tr_onsets[four_ab_idx]

one_cats = tr_cats[one_idx]
one_cats[np.logical_not(one_attn)] = '---'
two_cats = tr_cats[two_idx]
two_cats[np.logical_not(two_attn)] = '---'
four_a_cats = tr_cats[four_a_idx]
four_aa_cats = tr_cats[four_aa_idx]
four_ab_cats = tr_cats[four_ab_idx]

# RANDOMIZE TRIAL ORDER
if randomize:
    print('Randomizing trial order')
    sm_cat_trial_order = rng.permutation(trials // 2)
    md_cat_trial_order = rng.permutation(np.arange(trials // 2, trials))
    trial_order = np.concatenate((sm_cat_trial_order, md_cat_trial_order))
    tr_words = tr_words[trial_order]
    tr_cats = tr_cats[trial_order]
    tr_attn = tr_attn[trial_order]
    tr_targs = tr_targs[trial_order]
    tr_foils = tr_foils[trial_order]
    tr_targ_loc = tr_targ_loc[trial_order]
    tr_foil_loc = tr_foil_loc[trial_order]
    tr_onsets = tr_onsets[trial_order]
    tr_onset_samp = tr_onset_samp[trial_order]
    tr_stim = tr_stim[trial_order]

# WRITE WAV FILES
if write_wavs:
    print('Writing stimuli to disk')
    for tnum, trial in enumerate(tr_stim):
        fname = 'trial-{}-{}.wav'.format(np.char.mod('%03d', tnum),
                                         ''.join(np.char.array(tr_attn[tnum])))
        stim.write_wav(op.join(stimdir, fname), trial, output_fs,
                       overwrite=True)
    # training stims
    for tnum, trial in enumerate(one_stim):
        fname = 'train-one-{}.wav'.format(np.char.mod('%02d', tnum))
        stim.write_wav(op.join(stimdir, fname), trial, output_fs,
                       overwrite=True)
    for tnum, trial in enumerate(two_stim):
        fname = 'train-two-ab-{}.wav'.format(np.char.mod('%02d', tnum))
        stim.write_wav(op.join(stimdir, fname), trial, output_fs,
                       overwrite=True)
    for tnum, trial in enumerate(four_a_stim):
        fname = 'train-four-a-{}.wav'.format(np.char.mod('%02d', tnum))
        stim.write_wav(op.join(stimdir, fname), trial, output_fs,
                       overwrite=True)
    for tnum, trial in enumerate(four_aa_stim):
        fname = 'train-four-aa-{}.wav'.format(np.char.mod('%02d', tnum))
        stim.write_wav(op.join(stimdir, fname), trial, output_fs,
                       overwrite=True)
    for tnum, trial in enumerate(four_ab_stim):
        fname = 'train-four-ab-{}.wav'.format(np.char.mod('%02d', tnum))
        stim.write_wav(op.join(stimdir, fname), trial, output_fs,
                       overwrite=True)

# SAVE VARIABLES
print('Saving experimental variables')
varsToSave = dict(trials=trials, waves=waves, streams=streams, angles=angles,
                  isi=isi, tr_targs=tr_targs, tr_foils=tr_foils,
                  tr_cats=tr_cats, tr_words=tr_words,  tr_attn=tr_attn,
                  total_per_trial=total_per_trial,
                  targs_per_trial=targs_per_trial, tr_targ_loc=tr_targ_loc,
                  foils_per_trial=foils_per_trial, tr_foil_loc=tr_foil_loc,
                  tr_onset_sec=tr_onsets, tr_onset_samp=tr_onset_samp,
                  tn_one_attn=one_attn,
                  tn_two_attn=two_attn,
                  tn_four_a_attn=four_a_attn,
                  tn_four_aa_attn=four_aa_attn,
                  tn_four_ab_attn=four_ab_attn,
                  tn_one_words=one_words,
                  tn_two_words=two_words,
                  tn_four_a_words=four_a_words,
                  tn_four_aa_words=four_aa_words,
                  tn_four_ab_words=four_ab_words,
                  tn_one_cats=one_cats,
                  tn_two_cats=two_cats,
                  tn_four_a_cats=four_a_cats,
                  tn_four_aa_cats=four_aa_cats,
                  tn_four_ab_cats=four_ab_cats,
                  tn_one_onset=one_onset,
                  tn_two_onset=two_onset,
                  tn_four_a_onset=four_a_onset,
                  tn_four_aa_onset=four_aa_onset,
                  tn_four_ab_onset=four_ab_onset,
                  tn_one_targ_loc=one_targ_loc,
                  tn_two_targ_loc=two_targ_loc,
                  tn_four_a_targ_loc=four_a_targ_loc,
                  tn_four_aa_targ_loc=four_aa_targ_loc,
                  tn_four_ab_targ_loc=four_ab_targ_loc,
                  tn_one_foil_loc=one_foil_loc,
                  tn_two_foil_loc=two_foil_loc,
                  tn_four_a_foil_loc=four_a_foil_loc,
                  tn_four_aa_foil_loc=four_aa_foil_loc,
                  tn_four_ab_foil_loc=four_ab_foil_loc)
np.savez(outfile, **varsToSave)
