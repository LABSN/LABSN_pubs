# -*- coding: utf-8 -*-
"""
==================================
Script 'Reanalyze divAttnSem data'
==================================

This script reorganizes the divAttnSemantic data in preparation for plotting
alongside the dasCogLoad data.
"""
# Author: Dan McCloy <drmccloy@uw.edu>
#
# License: BSD (3-clause)

import json
import os.path as op
import numpy as np
import pandas as pd
from glob import glob
from ast import literal_eval
from datetime import datetime as dt
pd.set_option('display.width', 160)
np.set_printoptions(linewidth=160)


def nonan(x):
    x = np.array(x).ravel()
    return x[np.logical_not(np.isnan(x))]


def nonan_list(x):
    x = np.array(x).ravel()
    return str(x[np.logical_not(np.isnan(x))].tolist())


def assign_subj_num(x):
    return subj_nums[x]


# reconstruction of subject number based on raw data file timestamps:
rawdatafiles = glob('../divAttnSemantic/rawData/*.tab')
rawdatafiles = [op.split(x)[-1] for x in rawdatafiles]
dates = [x[3:19] for x in rawdatafiles]
dates = [dt.strptime(x, '%Y_%b_%d_%H%M').strftime('%Y-%m-%d-%H:%M')
         for x in dates]
subjs = [x[:2] for x in rawdatafiles]
subjs = np.array(subjs)[np.argsort(dates)].tolist()
subj_nums = {x: i for i, x in enumerate(subjs)}

indir = '../divAttnSemantic/analysis'
outfile = op.join('processedData', 'divAttnSemData.tsv')
bdata = pd.read_csv(op.join(indir, 'alldata.tab'), sep='\t')
cols_to_keep = ['subj', 'trial', 'test', 'div', 'adj', 'cats', 'attn',
                'words', 'targ_words', 'dist_words',
                'times', 'target_times', 'distractor_times', 'press_times',
                'num_targs', 'num_dists', 'hits', 'miss']
bdata = bdata[cols_to_keep]
bdata.columns = ['subj', 'trial', 'sem', 'div', 'adj', 'catg', 'attn',
                 'words', 'targ_words', 'foil_words',
                 'times', 'targ_times', 'foil_times', 'press_times',
                 'targs', 'foils', 'hit', 'miss']
# parse string representations
#bdata['targ_loc'] = bdata['targ_loc'].apply(literal_eval)
bdata['catg'] = bdata['catg'].apply(literal_eval)
bdata['attn'] = bdata['attn'].apply(literal_eval)
bdata['words'] = bdata['words'].apply(literal_eval)
bdata['targ_words'] = bdata['targ_words'].apply(literal_eval)
bdata['foil_words'] = bdata['foil_words'].apply(literal_eval)
bdata['times'] = bdata['times'].apply(literal_eval)
bdata['targ_times'] = bdata['targ_times'].apply(literal_eval)
bdata['foil_times'] = bdata['foil_times'].apply(literal_eval)
bdata['press_times'] = bdata['press_times'].apply(literal_eval)
# convert to arrays
#bdata['targ_loc'] = bdata['targ_loc'].apply(np.array)
bdata['words'] = bdata['words'].apply(np.array)
bdata['targ_words'] = bdata['targ_words'].apply(np.array)
bdata['foil_words'] = bdata['foil_words'].apply(np.array)
bdata['times'] = bdata['times'].apply(np.array)
bdata['targ_times'] = bdata['targ_times'].apply(np.array)
bdata['foil_times'] = bdata['foil_times'].apply(np.array)
bdata['press_times'] = bdata['press_times'].apply(np.array)
# add subject numbers, fix trial numbers
bdata['snum'] = bdata['subj'].apply(assign_subj_num)
bdata['trial'][np.logical_and(bdata['snum'] % 2 == 0, bdata['sem'])] = \
    bdata['trial'][np.logical_and(bdata['snum'] % 2 == 0, bdata['sem'])] + 60
bdata['trial'][np.logical_and(bdata['snum'] % 2 != 0, np.logical_not(bdata['sem']))] = \
    bdata['trial'][np.logical_and(bdata['snum'] % 2 != 0, np.logical_not(bdata['sem']))] + 120

# recalculate divAttnSemantic RTs to reject RTs outside 250 < x < 1250 ms
bdata['targ_loc'] = bdata['words'].apply(np.zeros_like, dtype=bool)
bdata['foil_loc'] = bdata['words'].apply(np.zeros_like, dtype=bool)
minRT = 0.25
maxRT = 1.25
tl = None
fl = None
for row in bdata.index:
    targ_loc = np.in1d(bdata.ix[row, 'words'].ravel(),
                       bdata.ix[row, 'targ_words'].ravel()
                       ).reshape(bdata.ix[row, 'words'].shape)
    foil_loc = np.in1d(bdata.ix[row, 'words'].ravel(),
                       bdata.ix[row, 'foil_words'].ravel()
                       ).reshape(bdata.ix[row, 'words'].shape)
    tl = targ_loc[np.newaxis] if tl is None else \
        np.concatenate((tl, targ_loc[np.newaxis]))
    fl = foil_loc[np.newaxis] if fl is None else \
        np.concatenate((fl, foil_loc[np.newaxis]))
bdata['targ_loc'] = tl.tolist()
bdata['foil_loc'] = fl.tolist()
bdata['targ_loc'] = bdata['targ_loc'].apply(np.array)
bdata['foil_loc'] = bdata['foil_loc'].apply(np.array)
bdata['rawrt'] = bdata['words'].apply(np.zeros_like, dtype=float) * np.nan
bdata['rt_hit'] = bdata['words'].apply(np.zeros_like, dtype=float) * np.nan
bdata['rt_fht'] = bdata['words'].apply(np.zeros_like, dtype=float) * np.nan
bdata['rt'] = bdata['targ_words'].apply(np.zeros_like, dtype=float) * np.nan

for row in bdata.index:
    press_times = bdata.ix[row, 'press_times']
    times = bdata.ix[row, 'times']
    targ = bdata.ix[row, 'targ_loc']
    foil = bdata.ix[row, 'foil_loc']
    hit = np.zeros_like(targ)  # targ hits
    fht = np.zeros_like(foil)  # foil hits
    sty = np.zeros_like(foil)  # stray presses
    rt = np.zeros_like(times) * np.nan
    for p in press_times:
        rawrt = p - times
        # exclude out-of-window response times
        rawrt[(rawrt < minRT) | (rawrt > maxRT)] = np.nan
        # exclude time slots that already have a press attributed
        # (handles two presses in rapid succession):
        if np.any(np.logical_and(np.logical_not(np.isnan(rawrt)),
                                 np.logical_not(np.isnan(rt)))):
            rawrt[np.logical_not(np.isnan(rt))] = np.nan
        # if press too late to attribute to anything, attribute to final word
        if np.sum(np.isnan(rawrt)) == rawrt.size:
            # last word of trial
            last_idx = np.where(times == np.max(times))
            rawrt[last_idx] = p - times[last_idx]
            # check again now that super-late press has been attributed:
            if np.any(np.logical_and(np.logical_not(np.isnan(rawrt)),
                                     np.logical_not(np.isnan(rt)))):
                # attribute to second-to-last word of trial instead
                rawrt[last_idx] = np.nan
                mask = times < np.max(times)
                last_idx = np.where(times == np.max(times[mask]))
                rawrt[last_idx] = p - times[last_idx]
            sty[last_idx] = True
        # else if this press can be attributed to a targ, do so
        elif not np.all(np.isnan(rawrt[targ])):
            rawrt[np.logical_not(targ)] = np.nan
            hit[np.where(rawrt > 0)] = True
        # else if this press can be attributed to a foil, do so
        elif not np.all(np.isnan(rawrt[foil])):
            rawrt[np.logical_not(foil)] = np.nan
            fht[np.where(rawrt > 0)] = True
        # else arbitrarily attribute to earliest possible non-targ non-foil
        else:
            ix = np.logical_not(np.isnan(rawrt))
            min_idx = np.where(np.logical_and(ix, rawrt == np.min(rawrt[ix])))
            mask = np.ones_like(rawrt, dtype=bool)
            mask[min_idx] = False
            rawrt[mask] = np.nan
            sty[min_idx] = True
        # aggregate RTs
        mask = np.logical_not(np.isnan(rawrt))
        rt[mask] = rawrt[mask]
    # make sure every press was attributed once
    mask = np.logical_not(np.isnan(rt))
    bdata.ix[row, 'rawrt'][mask] = rt[mask]
    rt_hit = rt.copy()
    rt_hit[np.logical_not(hit)] = np.nan
    rt_fht = rt.copy()
    rt_fht[np.logical_not(fht)] = np.nan
    mask = np.logical_not(np.isnan(rt_hit))
    bdata.ix[row, 'rt_hit'][mask] = rt_hit[mask]
    mask = np.logical_not(np.isnan(rt_fht))
    bdata.ix[row, 'rt_fht'][mask] = rt_fht[mask]
    assert press_times.size == nonan(rt).size
bdata['rt'] = bdata['rawrt'].apply(nonan)
bdata['rt_hit'] = bdata['rt_hit'].apply(nonan)
bdata['rt_fht'] = bdata['rt_fht'].apply(nonan)
bdata['presses'] = bdata['press_times'].apply(len)
bdata['fht'] = bdata['rt_fht'].apply(len)
bdata['sty'] = bdata['presses'] - bdata['hit'] - bdata['fht']
bdata['fal'] = bdata['sty'] + bdata['fht']
bdata['crj'] = bdata['words'].apply(np.size) - bdata['targs'] - bdata['fal']
bdata['frj'] = bdata['foils'] - bdata['fht']
assert(np.all(bdata['presses'] == bdata['hit'] + bdata['fht'] + bdata['sty']))
# convert to string representations prior to saveout
bdata['rt'] = bdata['rt'].apply(list)
bdata['rt_hit'] = bdata['rt_hit'].apply(list)
bdata['rt_fht'] = bdata['rt_fht'].apply(list)

bdata.to_csv(outfile, sep='\t', index=False)


#%% # # # # # # # # # # # # # # # # # # # # # # # # # #
# dump word-level data for mixed model analysis in R  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
streams = 4
waves = 12
stims = streams * waves
trialdata = None

indir = '../divAttnSemantic/analysis'
outfile = op.join('processedData', 'divAttnSemWordLevelData.tsv')

with open(op.join(indir, 'wordDurations.json')) as jd:
    word_durs = json.load(jd)
minRT, maxRT = 0.25, 1.25
for row in bdata.index:
    #date = np.tile(bdata.ix[row, 'datestring'], stims)
    subj = np.tile(bdata.ix[row, 'subj'], stims)
    subn = np.tile(bdata.ix[row, 'snum'], stims)
    trial = np.tile(bdata.ix[row, 'trial'], stims)
    onset = np.array(bdata.ix[row, 'times']).ravel()
    onset_og = np.tile(str(bdata.ix[row, 'times']), stims)  # -> to list later
    srt = np.argsort(onset)
    onset = onset[srt]
    stream = np.repeat(np.arange(streams), waves)[srt]
    attn = np.repeat(np.array(bdata.ix[row, 'attn']), waves).astype(bool)[srt]
    targ = np.array(bdata.ix[row, 'targ_loc']).ravel().astype(bool)[srt]
    foil = np.array(bdata.ix[row, 'foil_loc']).ravel().astype(bool)[srt]
    tloc = np.tile(str(bdata.ix[row, 'targ_loc']), stims)  # conv to list later
    floc = np.tile(str(bdata.ix[row, 'foil_loc']), stims)  # conv to list later
    odbl = targ + foil
    catg = np.repeat(bdata.ix[row, 'catg'], waves)[srt]
    catg_og = np.tile(str(bdata.ix[row, 'catg']), stims)  # conv to list later
    word = np.array(bdata.ix[row, 'words']).ravel()[srt]
    word_og = np.tile(str(bdata.ix[row, 'words']), stims)  # conv to list later
    dur = np.array([word_durs[x] for x in word])[srt]
    #cond = np.tile(bdata.ix[row, 'cond'], stims)[srt]
    #code = np.tile(bdata.ix[row, 'code'], stims)[srt]
    #cond_code = np.tile(bdata.ix[row, 'cond_code'], stims)[srt]
    div = np.tile(bdata.ix[row, 'div'], stims)[srt]
    adj = np.tile(bdata.ix[row, 'adj'], stims)[srt]
    sem = np.tile(bdata.ix[row, 'sem'], stims)[srt]
    presses = bdata.ix[row, 'press_times']
    rawrt = np.zeros_like(onset) * np.nan
    rt = rawrt
    hit = np.zeros_like(targ)  # targ hits
    fht = np.zeros_like(foil)  # foil hits
    sty = np.zeros_like(foil)  # stray presses
    fal = np.zeros_like(foil)  # union of fht and sty

    # assign button presses to stimulus words
    for p in presses:
        rawrt = p - onset
        rawrt[(rawrt < minRT) | (rawrt > maxRT)] = np.nan
        # if the press is too late to attribute to anything,
        # attribute to the final word
        if np.sum(np.isnan(rawrt)) == len(rawrt):
            rawrt[-1] = p - onset[-1]
            sty[-1] = True
        # else if this press can be attributed to a targ, do so
        elif not np.all(np.isnan(rawrt[targ])):
            rawrt[np.logical_not(targ)] = np.nan
            hit[np.where(rawrt > 0)[0]] = True
        # else if this press can be attributed to a foil, do so
        elif not np.all(np.isnan(rawrt[foil])):
            rawrt[np.logical_not(foil)] = np.nan
            fht[np.where(rawrt > 0)[0]] = True
        # else arbitrarily attribute to earliest possible non-targ non-foil
        else:
            min_idx = np.min(np.where(np.logical_not(np.isnan(rawrt)))[0])
            min_stray = rawrt[min_idx]
            rawrt[rawrt < min_stray] = np.nan
            sty[np.where(rawrt > 0)[0]] = True
        # make sure every press was attributed once
        assert sum(np.isnan(rawrt)) == len(rawrt) - 1
        mask = np.logical_not(np.isnan(rawrt))
        rt[mask] = rawrt[mask]
    # collect this subject's data into dataframe
    td = pd.DataFrame(dict(subn=subn, subj=subj, trial=trial, rt=rt,
                           onset=onset, onset_og=onset_og,
                           stream=stream, attn=attn, targ=targ, foil=foil,
                           odbl=odbl, tloc=tloc, floc=floc, catg=catg,
                           catg_og=catg_og, word=word, word_og=word_og,
                           dur=dur, sem=sem, div=div, adj=adj, hit=hit,
                           fht=fht, sty=sty, fal=fal), index=None)
    # merge all subjects' data into one dataframe
    if trialdata is None:
        trialdata = td
    else:
        trialdata = pd.concat((trialdata, td), ignore_index=True)

# divide reaction times into hits and foil responses
trialdata['fal'] = np.logical_or(trialdata['fht'], trialdata['sty'])
trialdata['rt_hit'] = trialdata['rt'][trialdata['hit']]
trialdata['rt_fht'] = trialdata['rt'][trialdata['fht']]
trialdata['rtch'] = trialdata['rt']
trialdata['rtch_hit'] = trialdata['rt_hit']
trialdata['rtch_fht'] = trialdata['rt_fht']
# reorder columns
column_order = ['subn', 'subj', 'trial', 'sem', 'div', 'adj', 'stream', 'attn',
                'catg', 'word', 'targ', 'foil', 'odbl', 'tloc', 'floc',
                'onset', 'onset_og', 'word_og', 'catg_og', 'dur', 'rt',
                'rt_hit', 'rt_fht', 'rtch', 'rtch_hit', 'rtch_fht', 'hit',
                'fht', 'sty', 'fal']
trialdata = trialdata[column_order].sort(['subj', 'trial', 'onset'])
# inclusion criterion: hitrate greater than 0.5 in selective attn condition
criterion = trialdata[(np.logical_not(trialdata['div']))].groupby('subj')
criterion = criterion.aggregate(np.sum)[['targ', 'hit']]
criterion = criterion[(criterion.hit / criterion.targ > 0.5)].index.tolist()
trialdata = trialdata[np.in1d(trialdata['subj'].values, criterion)]
trialdata = trialdata.reindex()
# needed for dprime
trialdata['miss'] = np.logical_and(trialdata['targ'],
                                   np.logical_not(trialdata['hit']))
trialdata['crj'] = np.logical_and(np.logical_not(trialdata['fal']),
                                  np.logical_not(trialdata['targ']))
trialdata['notg'] = np.logical_not(trialdata['targ'])
# write to file
trialdata.to_csv(outfile, sep='\t', index=False)
