# -*- coding: utf-8 -*-
"""
=================================
Script 'Make spatialized stimuli'
=================================

This script makes spatially-distributed word streams.
"""
# Author: Dan McCloy <drmccloy@uw.edu>
#
# License: BSD (3-clause)

import json
import socket
import numpy as np
import os.path as op
from glob import glob
from itertools import permutations, combinations, chain
from scipy import signal
from scipy.io import wavfile as wav
from scipy.io import loadmat as mat
from scipy.io import savemat


def rms(data, axis=0):
    """Calculate RMS amplitude along the given axis."""
    return np.sqrt(np.mean(data ** 2, axis))


def intofl(data):
    """int16 to float32 for wav files"""
    return np.ascontiguousarray((data / float(2 ** 15 - 1)), np.float32)


def fltoin(data):
    """float32 to int16 for wav files"""
    return np.ascontiguousarray((data * float(2 ** 15 - 1)), np.int16)


def ttr(seq, length):
    """Tile (to >= length), Truncate (to length, if necessary), Randomize.
    """
    if isinstance(seq, np.ndarray):
        sq = np.tile(seq, int(1 + length / len(seq)))
    else:
        sq = seq * int(1 + length / len(seq))
    sq = sq[:length]
    rng.shuffle(sq)
    return sq


def clean_mat(md, flatten=False):
    """Remove unwanted dict entries in MAT files & optionally flatten values
    """
    for key in ('__globals__', '__header__', '__version__'):
        if key in md:
            del md[key]
    if flatten:
        md = {k: list(chain.from_iterable(v)) for k, v in md.iteritems()}
    return md


# # # # # # # # # # # # # #
# random number generator #
# # # # # # # # # # # # # #
rng = np.random.RandomState(0)

# # # # # # # # # # # # # # # #
# input / output directories  #
# # # # # # # # # # # # # # # #
hn = socket.gethostname()
if hn == 'moctezuma':  # dan's LABSN desktop
    root = '/home/dan/Documents/experiments/drmccloy/divAttnSemantic'
elif hn == 'boulboul':  # dan's laptop
    root = ('/home/dan/Documents/academics/research/auditoryAttention/'
            'drmccloy/divAttnSemantic')
stim_dir = op.join(root, 'rawRecordings', 'monotonized')
vars_dir = op.join(root, 'variables')
out_dir = op.join(root, 'stimuli', 'trials')
train_dir = op.join(root, 'stimuli', 'training')
ctrl_dir = op.join(root, 'stimuli', 'controls')

# # # # # # # # # # # # #
# experiment parameters #
# # # # # # # # # # # # #
write_wavs = True
isi = 0.25  # onset-to-onset delay
output_fs = 24414.0625  # 44100.0
streams = 4
waves = 12
# some calculations...
# targs_per_condition = 30
# trials = len(conditions) * targs_per_condition / np.mean(targs_per_trial)
# trial_dur = isi * waves * streams
# trials_per_block = 600 / trial_dur
# blocks = np.ceil(trials / trials_per_block)
# trials_per_block = trials / blocks
trials = 120
blocks = 4
trials_per_block = 30
ctrl_trials = 60
ctrl_trials_per_block = 15

# dB jitter limits: +/- 1.5 dB
db_low = 10 ** (-1.5 / 20)
db_high = 10 ** (1.5 / 20)

""" for the following parameters, ratios are achieved by repetition: e.g.,
if you want 1/3 of trials to have 1 attended stream, and 2/3 to have 2 attended
streams, then set attn_streams_per_trial = [1, 2, 2] """
targs_per_trial = [2, 3]
dists_per_trial = [1, 2]
attn_streams_per_trial = [1, 2]

# # # # # # # # # # # #
# semantic categories #
# # # # # # # # # # # #
body = ['arm', 'chin', 'foot', 'leg', 'mouth', 'nose']  # 'wrist', 'knee'
fooddrink = ['beer', 'bread', 'meat', 'rice', 'soup', 'wine']  # 'cake','fruit'
furniture = ['bed', 'chair', 'couch', 'desk', 'lamp', 'rug']  # 'sink', 'stove'
plants = ['bark', 'grass', 'leaf', 'root', 'stick', 'tree']  # 'branch', 'moss'
animals = ['bird', 'cat', 'cow', 'mouse', 'pig', 'snake']  # 'fish', 'duck'
weather = ['breeze', 'cloud', 'fog', 'rain', 'sky', 'wind']  # 'hail', 'haze'
clothing = ['belt', 'dress', 'hat', 'scarf', 'shirt', 'suit']  # 'pants','purse'
cat_dict = dict(fooddrink=fooddrink, furniture=furniture, animals=animals,
                plants=plants, weather=weather, body=body, clothing=clothing)
controls = ['pants', 'wrist', 'moss', 'duck', 'cake', 'hail', 'fish', 'sink',
            'stove', 'branch', 'fruit', 'knee']
ctrl_targs = body + fooddrink + furniture + plants + animals + weather \
    + clothing
_ = ctrl_targs.pop(ctrl_targs.index('snake'))  # cake
_ = ctrl_targs.pop(ctrl_targs.index('mouse'))  # moss
_ = ctrl_targs.pop(ctrl_targs.index('suit'))   # fruit
_ = ctrl_targs.pop(ctrl_targs.index('root'))   # fruit

# # # # # # # # # # # # #
# set up randomizations #
# # # # # # # # # # # # #
print 'setting up randomizations'
# choose attended stream(s) for each trial
cond_types = [[1] * x + [0] * (streams - x) for x in attn_streams_per_trial]
conditions = list(chain.from_iterable([list(set(permutations(c)))
                  for c in cond_types]))
attn_streams = ttr(conditions, trials)
# constrain target positions: no targets in waves 1 & 2, no consecutive targets
targ_schemes = list(set(permutations([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])) |
                    set(permutations([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])))
targ_schemes = [(0, 0) + x for x in targ_schemes if '1, 1' not in str(x)]
targ_schemes = ttr(targ_schemes, trials)
# choose number of targets & distractors for each trial
# 1 or 2 distractors if there are 2 targets; 0 or 1 dist. if 3 targets
num_targs = ttr(targs_per_trial, trials)
num_dists = [sum(x) - num_targs[ii] for ii, x in enumerate(targ_schemes)]
total_targs = [x + y for x, y in zip(num_targs, num_dists)]
# determine stream categories / base stream words
stream_combos = list(set(combinations(cat_dict.keys(), streams)))
block_cats = [stream_combos[rng.choice(len(stream_combos))]]
for _ in xrange(streams - 1):
    similarity = np.sum(np.array([[len(set(b) & set(c)) for c in stream_combos]
                                  for b in block_cats]), axis=0).tolist()
    max_diff = [x == min(similarity) for x in similarity]
    choices = [stream_combos[x] for x in np.where(max_diff)[0]]
    chosen = choices[rng.choice(len(choices))]
    block_cats.append(chosen)
trial_cats = list(chain.from_iterable([[x] * trials_per_block for x in
                  block_cats]))
trial_words = [[ttr(cat_dict[y], waves) for y in x] for x in trial_cats]
del b, c, x, y, ii
# target / distractor locations
trial_codes = np.zeros_like(trial_words, dtype=int)
for tnum, trial in enumerate(trial_codes):
    """ for testing:
    # mark attended streams as 2
    for snum, stream in enumerate(trial):
        if attn_streams[tnum][snum] == 1:  # it's an attended stream
            trial_codes[tnum][snum] = [x + 2 for x in trial_codes[tnum][snum]]
        # mark target waves as 3
        for wnum, word in enumerate(stream):
            if targ_schemes[tnum][wnum] == 1:  # it's a target wave
                trial_codes[tnum][snum][wnum] += 3
    """
    # which waves get targs / dists?
    wave_indices = rng.permutation(np.where(targ_schemes[tnum])[0])
    targ_indices = wave_indices[:num_targs[tnum]]
    dist_indices = wave_indices[num_targs[tnum]:]
    # which streams get targs / dists?
    targ_streams = np.where(attn_streams[tnum])[0]
    dist_streams = np.where(np.logical_not(attn_streams[tnum]))[0]
    targ_streams = ttr(targ_streams, len(targ_indices))
    dist_streams = ttr(dist_streams, len(dist_indices))
    # assign target locations
    for ii, _ in enumerate(targ_indices):
        trial_codes[tnum][targ_streams[ii]][targ_indices[ii]] = 1
    for ii, _ in enumerate(dist_indices):
        trial_codes[tnum][dist_streams[ii]][dist_indices[ii]] = 1
del wave_indices, dist_indices, targ_indices, dist_streams, targ_streams
# targets categories by block / by trial, target words by trial
trial_targ_cats = [tuple(np.setdiff1d(cat_dict.keys(), x)) for x in trial_cats]
targ_choices = [list(chain.from_iterable([cat_dict[y] for y in x])) for x in
                trial_targ_cats]
# exclude (near) minimal pairs as targets:
for t in xrange(len(targ_choices)):
    # mouse mouth
    if 'body' in trial_cats[t] and 'animals' not in trial_cats[t]:
        _ = targ_choices[t].pop(targ_choices[t].index('mouse'))
    if 'animals' in trial_cats[t] and 'body' not in trial_cats[t]:
        _ = targ_choices[t].pop(targ_choices[t].index('mouth'))
    # couch cow
    if 'animals' in trial_cats[t] and 'furniture' not in trial_cats[t]:
        _ = targ_choices[t].pop(targ_choices[t].index('couch'))
    if 'furniture' in trial_cats[t] and 'animals' not in trial_cats[t]:
        _ = targ_choices[t].pop(targ_choices[t].index('cow'))
    # cat hat
    if 'animals' in trial_cats[t] and 'clothing' not in trial_cats[t]:
        _ = targ_choices[t].pop(targ_choices[t].index('hat'))
    if 'clothing' in trial_cats[t] and 'animals' not in trial_cats[t]:
        _ = targ_choices[t].pop(targ_choices[t].index('cat'))
    # desk dress
    if 'furniture' in trial_cats[t] and 'clothing' not in trial_cats[t]:
        _ = targ_choices[t].pop(targ_choices[t].index('dress'))
    if 'clothing' in trial_cats[t] and 'furniture' not in trial_cats[t]:
        _ = targ_choices[t].pop(targ_choices[t].index('desk'))
    # soup suit
    if 'clothing' in trial_cats[t] and 'fooddrink' not in trial_cats[t]:
        _ = targ_choices[t].pop(targ_choices[t].index('soup'))
    if 'fooddrink' in trial_cats[t] and 'clothing' not in trial_cats[t]:
        _ = targ_choices[t].pop(targ_choices[t].index('suit'))
    # bed bread
    if 'furniture' in trial_cats[t] and 'fooddrink' not in trial_cats[t]:
        _ = targ_choices[t].pop(targ_choices[t].index('bread'))
    if 'fooddrink' in trial_cats[t] and 'furniture' not in trial_cats[t]:
        _ = targ_choices[t].pop(targ_choices[t].index('bed'))
    # semantic categories too similar: weather/plants (="nature")
    if 'weather' in trial_cats[t] and 'plants' not in trial_cats[t]:
        _ = [targ_choices[t].pop(targ_choices[t].index(x)) for x in plants]
    if 'plants' in trial_cats[t] and 'weather' not in trial_cats[t]:
        _ = [targ_choices[t].pop(targ_choices[t].index(x)) for x in weather]

targ_words = [(rng.choice(tlist, total_targs[tnum], False)).tolist()
              for tnum, tlist in enumerate(targ_choices)]
# replace stream words with target words
for tnum, trial in enumerate(trial_words):
    for snum, stream in enumerate(trial):
        for wnum, _ in enumerate(stream):
            if trial_codes[tnum][snum][wnum] == 1:
                trial_words[tnum][snum][wnum] = targ_words[tnum].pop(0)

del body, fooddrink, furniture, plants, animals, weather, clothing

# # # # # # # # # # # # # # #
# set up control condition  #
# # # # # # # # # # # # # # #
# create all possible combos, pick one at random, pick remaining ones to
# maximize diversity of category appearance
ctrl_attn_streams = ttr(conditions, ctrl_trials)
ctrl_combos = list(set(combinations(controls, streams)))
ctrl_blocks = [ctrl_combos[rng.choice(len(ctrl_combos))]]
for _ in xrange(streams - 1):
    similarity = np.sum(np.array([[len(set(b) & set(c)) for c in ctrl_combos]
                                  for b in ctrl_blocks]), axis=0).tolist()
    max_diff = [x == min(similarity) for x in similarity]
    choices = [ctrl_combos[x] for x in np.where(max_diff)[0]]
    chosen = choices[rng.choice(len(choices))]
    ctrl_blocks.append(chosen)
ctrl_cats = list(chain.from_iterable([[x] * ctrl_trials_per_block for x in
                 ctrl_blocks]))
ctrl_words = np.tile(np.array(ctrl_cats).T, (waves, 1, 1)).T
ctrl_codes = np.zeros_like(ctrl_words, dtype=int)
for tnum, trial in enumerate(ctrl_codes):
    wave_indices = rng.permutation(np.where(targ_schemes[tnum])[0])
    targ_indices = wave_indices[:num_targs[tnum]]
    dist_indices = wave_indices[num_targs[tnum]:]
    # which streams get targs / dists?
    targ_streams = np.where(ctrl_attn_streams[tnum])[0]
    dist_streams = np.where(np.logical_not(ctrl_attn_streams[tnum]))[0]
    targ_streams = ttr(targ_streams, len(targ_indices))
    dist_streams = ttr(dist_streams, len(dist_indices))
    # assign target locations
    for ii, _ in enumerate(targ_indices):
        ctrl_codes[tnum][targ_streams[ii]][targ_indices[ii]] = 1
    for ii, _ in enumerate(dist_indices):
        ctrl_codes[tnum][dist_streams[ii]][dist_indices[ii]] = 1
del wave_indices, dist_indices, targ_indices, dist_streams, targ_streams
# replace control words with target words
ctrl_targ_words = list(chain.from_iterable([rng.choice(ctrl_targs, np.sum(x),
                                                       False)
                                           for x in ctrl_codes]))
ctrl_words[np.where(ctrl_codes)] = ctrl_targ_words
ctrl_words = ctrl_words.tolist()

# # # # # # # # # # # # # # # # # # # # # # # #
# Binaural room impulse responses (anechoic)  #
# # # # # # # # # # # # # # # # # # # # # # # #
print 'loading BRIRs'
""" fimp_l and fimp_r are 32767 x 3 x 3 x 7, corresponding to the time domain
axis, the repeated measurement number, the distance (0.15, 0.40, 1 m) and the
direction (0:15:90 degrees). fs = 44100 for all BRIR files."""
brir_anechoic = mat(op.join(root, 'hrtf/anechRev.mat'))
brir_fs = 44100
# sample x measurement_num x dist x angle
plus15_l = brir_anechoic['fimp_l'][:, 0, -1, 1].astype(np.float64)
plus60_l = brir_anechoic['fimp_l'][:, 0, -1, 4].astype(np.float64)
plus15_r = brir_anechoic['fimp_r'][:, 0, -1, 1].astype(np.float64)
plus60_r = brir_anechoic['fimp_r'][:, 0, -1, 4].astype(np.float64)
brir = [plus15_l, plus60_l, plus15_r, plus60_r]
# do we need to resample BRIRs?
if not np.allclose(brir_fs, output_fs, rtol=0, atol=0.5):
    print ('resampling BRIRs: {} to {} Hz'.format(brir_fs, output_fs))
    blen = [int(round(len(x) * output_fs / brir_fs)) for x in brir]
    brir = [signal.resample(b, n, window='boxcar') for b, n in zip(brir, blen)]
    del blen
spatial = [(brir[3], brir[1]), (brir[2], brir[0]),
           (brir[0], brir[2]), (brir[1], brir[3])]
del brir_anechoic, brir, plus15_l, plus15_r, plus60_l, plus60_r

# # # # # # # #
# raw stimuli #
# # # # # # # #
print 'loading raw WAVs'
stim_files = glob(op.join(stim_dir, '*.wav'))
word_dict = {}
fs_dict = {}
for f in stim_files:
    name = op.splitext(op.basename(f))[0]
    fs, data = wav.read(op.join(stim_dir, f))
    word_dict[name] = data
    fs_dict[name] = fs
# do we need to resample raw stimuli?
printed = False
for key in word_dict.keys():
    fs = fs_dict[key]
    wav_file = word_dict[key]
    if not np.allclose(fs, output_fs, rtol=0, atol=0.5):
        if not printed:
            print 'resampling raw WAVs: {} to {} Hz'.format(fs, output_fs)
            printed = True
        n_samp = int(round(len(wav_file) * output_fs / fs))
        word_dict[key] = signal.resample(wav_file, n_samp, window='boxcar')
max_word_dur = np.max([len(x) / float(output_fs) for x in word_dict.values()])
stream_len = int((max_word_dur + streams * isi * waves) * output_fs)
brir_len = np.max([len(x) for x in list(chain.from_iterable(spatial))])
# RMS normalize raw stims
desired_rms = 0.01
print 'normalizing to {} RMS'.format(desired_rms)
word_dict = {key: val * desired_rms / rms(val) for key, val in
             word_dict.iteritems()}
del f, fs, data, key, wav_file, printed, max_word_dur

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# run the rest twice, once for ctrl and once for test condition #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
for cnum, condition in enumerate([ctrl_words, trial_words]):
    if cnum == 0:  # ctrl words
        prefix = 'ctrl'
        cats = ctrl_cats
        codes = ctrl_codes
        tr_per_bl = ctrl_trials_per_block
        n_trials = ctrl_trials
        attn_stream = np.array(ctrl_attn_streams).astype(bool)
    else:
        prefix = 'test'
        cats = trial_cats
        codes = trial_codes
        tr_per_bl = trials_per_block
        n_trials = trials
        attn_stream = np.array(attn_streams).astype(bool)
    # # # # # # # # # # # # # # # # # #
    # construct streams and convolve  #
    # # # # # # # # # # # # # # # # # #
    print 'convolving trial stimuli with BRIRs'
    stream_rms = np.empty((n_trials, streams))
    convolved_rms = np.empty((n_trials, streams))
    convolved_len = stream_len + brir_len - 1
    convolved_trials = np.empty((n_trials, streams, 2, convolved_len))
    onset_times = np.empty((n_trials, streams, waves))
    for t, trial in enumerate(condition):
        order = np.zeros((waves, streams), dtype=int)
        db_jitter = rng.uniform(db_low, db_high, np.size(order))
        db_jitter = np.reshape(db_jitter, np.shape(order))
        # ensure end of one wave & start of next are not in same stream
        for w in xrange(waves):
            order[w, :] = rng.choice(range(streams), streams, False)
            if w > 0:
                penult = np.where(order[w - 1] == streams - 2)[0][0]
                last = np.where(order[w - 1] == streams - 1)[0][0]
                first = np.where(order[w] == 0)[0][0]
                second = np.where(order[w] == 1)[0][0]
                while last in (first, second) or penult == first:
                    order[w, :] = rng.choice(range(streams), streams, False)
                    first = np.where(order[w] == 0)[0][0]
                    second = np.where(order[w] == 1)[0][0]
        onsets = np.array([isi * (od + streams * o)
                           for o, od in enumerate(order)])
        onset_samp = (onsets * output_fs).round().astype(int)
        onset_times[t] = onsets.T
        for s, stream in enumerate(trial):
            this_stream = np.zeros((stream_len))
            for w, (word, onset) in enumerate(zip(stream, onset_samp[:, s])):
                this_stream[onset:onset + len(word_dict[word])] += \
                    db_jitter[w, s] * word_dict[word]
            stream_rms[t, s] = rms(this_stream)
            this_left = signal.fftconvolve(this_stream, spatial[s][0], 'full')
            this_right = signal.fftconvolve(this_stream, spatial[s][1], 'full')
            convolved_trials[t, s, 0] = this_left
            convolved_trials[t, s, 1] = this_right
            convolved_rms[t, s] = np.mean([rms(this_left), rms(this_right)])
    del (order, onsets, onset_samp, this_left, this_right, this_stream, s,
         stream, word, onset, t, trial, o, od)
    # re-normalize after convolution
    print 're-normalizing to {} RMS'.format(desired_rms)
    mplier = stream_rms / convolved_rms
    normed_streams = np.ascontiguousarray((convolved_trials.T * mplier.T).T)
    del stream_rms, mplier, convolved_rms  # , mean_stream_rms
    # combine streams
    final_wavs = np.empty((n_trials, normed_streams.shape[-1], 2))
    for t, trial in enumerate(normed_streams):
        trial_wav = np.zeros_like(trial[0].T)
        for stream in trial:
            trial_wav += np.ascontiguousarray(stream.T)
        final_wavs[t] = trial_wav
    # write WAVs
    if write_wavs:
        print 'writing trial WAV files'
        for t, trial_wav in enumerate(final_wavs):
            wav.write(op.join(out_dir, '{}{}.wav'.format(prefix, t)),
                      output_fs, fltoin(trial_wav))
    # # # # # # #
    # training  #
    # # # # # # #
    print 'generating training stimuli'
    train_single = {}
    train_select = {}
    train_full = {}
    train_dual = {}
    train_divide = {}
    train_times = {}
    train_codes = {}
    train_locs = {}
    train_cats = {}
    for r, row in enumerate(attn_stream):
        attn = list(chain.from_iterable(np.where(row)))
        dist = list(chain.from_iterable(np.where(np.logical_not(row))))
        mask = np.ones_like(onset_times[0], dtype=bool)
        mask[attn] = False
        targ_names = np.array(cats)[r, attn].tolist()

        # full & divided phases
        name = '{1}-{2}-{3}-{4}-{0}'.format(r, *[x[:4] for x in cats[r]])
        train_locs[name] = attn
        train_times[name] = onset_times[r]
        train_codes[name] = codes[r]
        train_cats[name] = ['food & drink' if x == 'fooddrink' else x
                            for x in cats[r]]
        if sum(row) == 2:  # divided
            train_divide[name] = final_wavs[r]
        else:  # full
            train_full[name] = final_wavs[r]
        if write_wavs:
            wav.write(op.join(train_dir, '{}.wav'.format(name)), output_fs,
                      fltoin(final_wavs[r]))

        # dual & selective phases
        if sum(row) == 2:  # dual
            targ_names = np.array(cats)[r, attn].tolist()
            name = '{1}-{2}-{0}'.format(r, *targ_names)
            # attended stream nums
            train_locs[name] = attn
            # onset times
            train_times[name] = np.ma.masked_array(onset_times[r], mask,
                                                   fill_value=-1.)
            # codes
            train_codes[name] = np.ma.masked_array(codes[r], mask,
                                                   fill_value=-1)
            # categories
            train_cats[name] = ['food & drink' if x == 'fooddrink' else x
                                for x in [cats[r][n] if n in attn else u'•'
                                          for n in xrange(streams)]]
            # audio streams
            train_dual[name] = (normed_streams[r, attn[0]] +
                                normed_streams[r, attn[1]]).T
            if write_wavs:
                wav.write(op.join(train_dir, '{}.wav'.format(name)), output_fs,
                          fltoin(train_dual[name]))
        else:  # selective
            targ_name = cats[r][attn[0]]
            for d in dist:
                if sum(codes[r, d]) == 1:
                    dist_name = cats[r][d]
                    name = '{1}-{2}-{0}'.format(r, targ_name, dist_name)
                    # attended stream nums
                    train_locs[name] = attn
                    # onset times
                    train_times[name] = np.ma.masked_array(onset_times[r],
                                                           mask,
                                                           fill_value=-1.)
                    # codes
                    train_codes[name] = np.ma.masked_array(codes[r], mask,
                                                           fill_value=-1)
                    # categories
                    train_cats[name] = ['food & drink' if x == 'fooddrink' else
                                        x for x in [cats[r][n] if n in
                                                    (attn[0], d) else u'•'
                                                    for n in xrange(streams)]]
                    # audio streams
                    train_select[name] = (normed_streams[r, attn[0]] +
                                          normed_streams[r, d]).T
                    if write_wavs:
                        wav.write(op.join(train_dir, '{}.wav'.format(name)),
                                  output_fs, fltoin(train_select[name]))

        # single-stream phase
        stream = list(chain.from_iterable(np.where(row)))
        for s in stream:
            if np.sum(codes[r, s]) > 1:
                cat = cats[r][s]
                name = '{}-{}'.format(cat, r)
                train_locs[name] = s
                # mask
                mask = np.ones_like(onset_times[0], dtype=bool)
                mask[s] = False
                # onset times
                train_times[name] = np.ma.masked_array(onset_times[r], mask,
                                                       fill_value=-1.,
                                                       copy=True)
                # codes
                train_codes[name] = np.ma.masked_array(codes[r], mask,
                                                       fill_value=-1)
                # categories
                train_cats[name] = ['food & drink' if y == 'fooddrink' else y
                                    for y in [cat if x == s else u'•'
                                              for x in xrange(streams)]]
                # audio streams
                train_single[name] = normed_streams[r, s].T
                if write_wavs:
                    wav.write(op.join(train_dir, '{}.wav'.format(name)),
                              output_fs, fltoin(train_single[name]))
    # # # # # # # # # #
    # save variables  #
    # # # # # # # # # #
    if cnum == 0:
        ctrl_times = onset_times.copy()
    else:
        test_times = onset_times.copy()
    # trial stimuli by block
    for bl in xrange(blocks):
        f = tr_per_bl * bl
        l = tr_per_bl * (bl + 1)
        savemat(op.join(vars_dir, '{}_block{}.mat'.format(prefix, bl + 1)),
                dict(stims=final_wavs[f:l]), oned_as='row')
    # training stimulus dicts
    savemat(op.join(vars_dir, '{}_single.mat'.format(prefix)), train_single,
            oned_as='row')
    savemat(op.join(vars_dir, '{}_select.mat'.format(prefix)), train_select,
            oned_as='row')
    savemat(op.join(vars_dir, '{}_full.mat'.format(prefix)), train_full,
            oned_as='row')
    savemat(op.join(vars_dir, '{}_dual.mat'.format(prefix)), train_dual,
            oned_as='row')
    savemat(op.join(vars_dir, '{}_divide.mat'.format(prefix)), train_divide,
            oned_as='row')
    # training variable dicts
    savemat(op.join(vars_dir, '{}_cats.mat'.format(prefix)), train_cats,
            oned_as='row')
    savemat(op.join(vars_dir, '{}_locs.mat'.format(prefix)), train_locs,
            oned_as='row')
    savemat(op.join(vars_dir, '{}_times.mat'.format(prefix)), train_times,
            oned_as='row')
    savemat(op.join(vars_dir, '{}_codes.mat'.format(prefix)), train_codes,
            oned_as='row')

# # # # # # # # # # # # # # #
# write randomization data  #
# # # # # # # # # # # # # # #
print 'saving variables'
output_vars = dict(trials=trials, blocks=blocks, streams=streams, isi=isi,
                   fs=output_fs, tpb=trials_per_block, ctrl_trials=ctrl_trials,
                   ctpb=ctrl_trials_per_block)  # waves=waves,
with open(op.join(vars_dir, 'divAttnSemantic.json'), 'w') as f:
    json.dump(output_vars, f)

savemat(op.join(vars_dir, 'divAttnSemantic.mat'),
        dict(attn=attn_streams,
             codes=trial_codes,
             words=trial_words,
             times=test_times,
             cats=trial_cats,
             ctrl_attn=ctrl_attn_streams,
             ctrl_codes=ctrl_codes,
             ctrl_words=ctrl_words,
             ctrl_times=ctrl_times,
             ctrl_cats=ctrl_cats), oned_as='row')
