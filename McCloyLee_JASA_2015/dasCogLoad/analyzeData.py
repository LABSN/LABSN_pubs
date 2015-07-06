# -*- coding: utf-8 -*-
"""
==================================
Script 'Analyze DAS-cog-load data'
==================================

This script analyses stuff.
"""
# Author: Dan McCloy <drmccloy@uw.edu>
#
# License: BSD (3-clause)

import json
import os.path as op
import numpy as np
import pandas as pd
import expyfun.analyze as efa
from ast import literal_eval


outdir = 'processedData'
infile = op.join(outdir, 'processedData.tsv')

data = []
with open(infile, 'r') as f:
    colnames = f.readline().strip().split('\t')
    """ colnames =
    ['trial_id', 'subj_num', 'subj', 'datestring', 'target_times',
     'foil_times', 'press_times', 'false_alarm_times', 'target_RTs',
     'foil_RTs', 'foil_loc', 'cats', 'words', 'targ_loc', 'attn', 'onsets']
    """
    for line in f:
        l = line.strip().split('\t')
        l = [int(l[0]),  # trial_id
             int(l[1]),  # subj_num
             str(l[2]),  # subj
             str(l[3]),  # datestring
             [float(x) for x in l[4].strip('[]').split()],       # target_times
             [float(x) for x in l[5].strip('[]').split()],       # foil_times

             [float(x.strip()) for x in l[6].strip('[]').split(',')
              if x.strip() != ''],                               # press_times

             [float(x.strip()) for x in l[7].strip('[]').split(',')
              if x.strip() != ''],                          # false_alarm_times

             [float(x.strip()) for x in l[8].strip('[]').split(',')
              if x.strip() != ''],                               # target_RTs

             [float(x.strip()) for x in l[9].strip('[]').split(',')
              if x.strip() != ''],                               # foil_RTs

             [[int(y) for y in x.split(' ')] for x in
              l[10].strip('[]').split('] [')],                   # foil_loc

             l[11].strip('[]').replace('\'', '').split(),        # cats

             [x.replace('\'', '').split() for x in
              l[12].strip('[]').split('] [')],                   # words

             [[int(y) for y in x.split()] for x in
              l[13].strip('[]').split('] [')],                   # targ_loc

             [int(x) for x in l[14].strip('[]').split(' ')],     # attn

             [[float(y) for y in x.strip(' ').split()] for x in
              l[15].strip('[]').split('] [')]                    # onsets
             ]
        data.append(l)

df = pd.DataFrame(data, columns=colnames)
df['ntarg'] = df['target_times'].apply(len)
df['nfoil'] = df['foil_times'].apply(len)
df['nfals'] = df['false_alarm_times'].apply(len)
df['nhits'] = [np.sum(np.array(x) > 0) for x in df['target_RTs'].tolist()]
df['nfhit'] = [np.sum(np.array(x) > 0) for x in df['foil_RTs'].tolist()]
df['nmiss'] = df['ntarg'] - df['nhits']
df['nfmis'] = df['nfoil'] - df['nfhit']

# condition codes
cond_codes = {'1000': 'sel', '0100': 'sel', '0010': 'sel', '0001': 'sel',
              '1100': 'adj', '0110': 'adj', '0011': 'adj',
              '1010': 'sep', '1001': 'sep', '0101': 'sep',
              '2200': 'ida', '0220': 'ida', '0022': 'ida',
              '2020': 'ids', '2002': 'ids', '0202': 'ids'}
df['cond'] = [''.join([str(x) for x in y]) for y in df['attn']]
df['ident'] = np.array([x.count('2') for x in df['cond']], dtype=bool)
df['divided'] = [np.sum(x) > 1 for x in df['attn']]
df['code'] = [cond_codes[x] for x in df['cond']]
# size codes
size_codes = dict(fish='thr', fruits='thr', birds='thr', drinks='thr',
                  food='six', colors='six', weather='six', furniture='six')
df['size'] = [size_codes[x[0]] for x in df['cats']]
df['cond_code'] = ['_'.join(x) for x in zip(df['code'], df['size'])]


#%% # # # # # # # # # # # # # # # #
# ASSESS MULTI-STREAM HIT ABILITY #
# # # # # # # # # # # # # # # # # #
def sum_bools(arr, axis=-1, usenan=False):
    result = np.sum(arr, axis=axis).astype(float)
    if usenan:
        result[result == 0] = np.nan
    return result

foo = df
foo['six'] = foo['size'] == 'six'
foo['adj'] = ['11' in x or '22' in x for x in foo['cond']]
foo['targ_onsets'] = [[[z if z in y else np.nan for z in w] for w in x]
                      for x, y in zip(foo['onsets'], foo['target_times'])]
#foo['rawrts'] = [[[v - z if v - z < 1.25 and v - z > 0.25 else np.nan
#                   for v in y for z in w] for w in x]
#                 for x, y in zip(foo['targ_onsets'], foo['press_times'])]
foo['hit_loc'] = [[[True if v - z < 1.25 and v - z > 0.25 else False
                    for v in y for z in w] for w in x]
                  for x, y in zip(foo['targ_onsets'], foo['press_times'])]
foo = foo[['subj', 'divided', 'adj', 'ident', 'six', 'attn', 'targ_loc',
          'hit_loc']]
foo['targ_per_stream'] = foo['targ_loc'].apply(sum_bools, usenan=True)
foo['hits_tmp'] = foo['hit_loc'].apply(sum_bools)
foo['miss_per_stream'] = foo['targ_per_stream'] - foo['hits_tmp']
foo['hit_per_stream'] = foo['targ_per_stream'] - foo['miss_per_stream']
foo['hit_in_stream'] = foo['hit_per_stream'].apply(lambda x: x > 0)
foo['targ_in_stream'] = foo['targ_per_stream'].apply(lambda x: x > 0)
foo['hit_multistream'] = foo['hit_in_stream'].apply(lambda x: sum(x) > 1)
foo['targ_multistream'] = foo['targ_in_stream'].apply(lambda x: sum(x) > 1)
# mh = multihit, or at least 1 hit in each attended stream
'''
mh_div = foo[foo['div']][['hit_multistream', 'targ_multistream']].sum().values
mh_adj = foo[(foo['div'] &  foo['adj'])][['hit_multistream', 'targ_multistream']].sum().values
mh_sep = foo[(foo['div'] & ~foo['adj'])][['hit_multistream', 'targ_multistream']].sum().values
mh_thr = foo[(foo['div'] & ~foo['six'])][['hit_multistream', 'targ_multistream']].sum().values
mh_six = foo[(foo['div'] &  foo['six'])][['hit_multistream', 'targ_multistream']].sum().values
'''
mh_tha = foo[(foo['divided'] & ~foo['six'] &  foo['adj'])][['hit_multistream', 'targ_multistream']].sum().values
mh_ths = foo[(foo['divided'] & ~foo['six'] & ~foo['adj'])][['hit_multistream', 'targ_multistream']].sum().values
mh_sxa = foo[(foo['divided'] &  foo['six'] &  foo['adj'])][['hit_multistream', 'targ_multistream']].sum().values
mh_sxs = foo[(foo['divided'] &  foo['six'] & ~foo['adj'])][['hit_multistream', 'targ_multistream']].sum().values
'''
mh_tham = foo[(foo['divided'] & ~foo['six'] &  foo['adj'] &  foo['ident'])][['hit_multistream', 'targ_multistream']].sum().values
mh_thad = foo[(foo['divided'] & ~foo['six'] &  foo['adj'] & ~foo['ident'])][['hit_multistream', 'targ_multistream']].sum().values
mh_thsm = foo[(foo['divided'] & ~foo['six'] & ~foo['adj'] &  foo['ident'])][['hit_multistream', 'targ_multistream']].sum().values
mh_thsd = foo[(foo['divided'] & ~foo['six'] & ~foo['adj'] & ~foo['ident'])][['hit_multistream', 'targ_multistream']].sum().values
mh_sxam = foo[(foo['divided'] &  foo['six'] &  foo['adj'] &  foo['ident'])][['hit_multistream', 'targ_multistream']].sum().values
mh_sxad = foo[(foo['divided'] &  foo['six'] &  foo['adj'] & ~foo['ident'])][['hit_multistream', 'targ_multistream']].sum().values
mh_sxsm = foo[(foo['divided'] &  foo['six'] & ~foo['adj'] &  foo['ident'])][['hit_multistream', 'targ_multistream']].sum().values
mh_sxsd = foo[(foo['divided'] &  foo['six'] & ~foo['adj'] & ~foo['ident'])][['hit_multistream', 'targ_multistream']].sum().values
'''

#%% # # # # # # # # # # # # # # # # # # # # # # # # # #
# dump word-level data for mixed model analysis in R  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
streams = 4
waves = 12
stims = streams * waves
trialdata = None
with open(op.join(outdir, 'wordDurations.json')) as jd:
    word_durs = json.load(jd)
# set reaction time window ([0.1, 1.25] was hard-coded in runDASCogLoad.py)
minRT = 0.25
maxRT = 1.25
for row in df.index:
    date = np.tile(df.ix[row, 'datestring'], stims)
    subj = np.tile(df.ix[row, 'subj'], stims)
    subn = np.tile(df.ix[row, 'subj_num'], stims)
    trial = np.tile(df.ix[row, 'trial_id'], stims)
    onset = np.array(df.ix[row, 'onsets']).ravel()
    onset_og = np.tile(str(df.ix[row, 'onsets']), stims)  # conv to list later
    srt = np.argsort(onset)
    onset = onset[srt]
    stream = np.repeat(np.arange(streams), waves)[srt]
    attn = np.repeat(np.array(df.ix[row, 'attn']), waves).astype(bool)[srt]
    targ = np.array(df.ix[row, 'targ_loc']).ravel().astype(bool)[srt]
    foil = np.array(df.ix[row, 'foil_loc']).ravel().astype(bool)[srt]
    tloc = np.tile(str(df.ix[row, 'targ_loc']), stims)  # conv to list later
    floc = np.tile(str(df.ix[row, 'foil_loc']), stims)  # conv to list later
    odbl = targ + foil
    catg = np.repeat(df.ix[row, 'cats'], waves)[srt]
    catg_og = np.tile(str(df.ix[row, 'cats']), stims)  # conv to list later
    word = np.array(df.ix[row, 'words']).ravel()[srt]
    word_og = np.tile(str(df.ix[row, 'words']), stims)  # conv to list later
    dur = np.array([word_durs[x] for x in word])[srt]
    cond = np.tile(df.ix[row, 'cond'], stims)[srt]
    code = np.tile(df.ix[row, 'code'], stims)[srt]
    cond_code = np.tile(df.ix[row, 'cond_code'], stims)[srt]
    div = np.tile(df.ix[row, 'divided'], stims)[srt]
    adj = np.array(['11' in x or '22' in x for x in cond])[srt]
    idn = np.tile(df.ix[row, 'ident'], stims)[srt]
    num = np.tile(df.ix[row, 'size'], stims)[srt]
    presses = df.ix[row, 'press_times']
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
    td = pd.DataFrame(dict(date=date, subj=subj, subn=subn, trial=trial, rt=rt,
                           onset=onset, onset_og=onset_og,
                           stream=stream, attn=attn, targ=targ, foil=foil,
                           odbl=odbl, tloc=tloc, floc=floc, catg=catg,
                           catg_og=catg_og, word=word, word_og=word_og,
                           dur=dur, cond=cond, code=code, cond_code=cond_code,
                           div=div, idn=idn, num=num, adj=adj, hit=hit,
                           fht=fht, sty=sty, fal=fal), index=None)
    # merge all subjects' data into one dataframe
    if trialdata is None:
        trialdata = td
    else:
        trialdata = pd.concat((trialdata, td), ignore_index=True)

# divide reaction times into hits and foil responses
trialdata['fal'] = np.logical_or(trialdata['fht'], trialdata['sty'])
trialdata['rt_hit'] = trialdata['rt'][trialdata['hit']]  # will aggr as list
trialdata['rt_fht'] = trialdata['rt'][trialdata['fht']]
trialdata['rtch'] = trialdata['rt']                      # will aggr as chisq
trialdata['rtch_hit'] = trialdata['rt_hit']
trialdata['rtch_fht'] = trialdata['rt_fht']
# sandwich vs leftovers
sand_a = np.logical_and(trialdata['stream'] == 1, np.in1d(trialdata['cond'],
                        ['1010', '1001', '2020', '2002']))
sand_b = np.logical_and(trialdata['stream'] == 2, np.in1d(trialdata['cond'],
                        ['0101', '1001', '0202', '2002']))
sandwich = np.logical_or(sand_a, sand_b)
trialdata['sandwich'] = np.logical_and(sandwich, trialdata['foil'])
trialdata['leftover'] = np.logical_and(np.logical_not(sandwich),
                                       trialdata['foil'])
trialdata['snd'] = np.logical_and(trialdata['fht'], trialdata['sandwich'])
trialdata['lft'] = np.logical_and(trialdata['fht'], trialdata['leftover'])
trialdata['rt_snd'] = trialdata['rt'][trialdata['sandwich']]
trialdata['rt_lft'] = trialdata['rt'][trialdata['leftover']]
trialdata['rtch_snd'] = trialdata['rt_snd']
trialdata['rtch_lft'] = trialdata['rt_lft']
# reorder columns
column_order = ['subn', 'subj', 'trial', 'div', 'adj', 'idn', 'num', 'cond',
                'code', 'cond_code', 'stream', 'attn', 'catg', 'word', 'targ',
                'foil', 'odbl', 'tloc', 'floc', 'onset', 'onset_og', 'word_og',
                'catg_og', 'dur', 'rt', 'rt_hit', 'rt_fht', 'rt_snd', 'rt_lft',
                'rtch', 'rtch_hit', 'rtch_fht', 'rtch_snd', 'rtch_lft', 'hit',
                'fht', 'sty', 'fal', 'snd', 'lft', 'sandwich', 'leftover']
trialdata = trialdata[column_order].sort(['subn', 'trial', 'onset'])
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
trialdata.to_csv(op.join(outdir, 'wordLevelData.tsv'), sep='\t', index=False)


#%% # # # # # # # # # # # # # # # # # # #
# AGGREGATE MEASURES AT VARIOUS LEVELS  #
# # # # # # # # # # # # # # # # # # # # #
def uniq(x):
    # (list of) unique value(s)
    y = list(set(x))
    if len(y) == 1:
        y = y[0]
    return y


def uniq_rt(x):
    # unique value(s) excluding NaN
    y = np.array(x)
    y = y[np.logical_not(np.isnan(y))].tolist()
    return y


def chsq_rt(x):
    # peak of chi squared distribution fit to data
    y = np.array(x)
    y = y[np.logical_not(np.isnan(y))]
    if y.size == 0:
        return np.nan
    else:
        return efa.rt_chisq(y)


col_order_out = ['subj', 'trial', 'div', 'adj', 'idn', 'num',  # 'catg', 'word'
                 'cond', 'code', 'cond_code', 'targ', 'notg', 'foil', 'hit',
                 'miss', 'fht', 'sty', 'fal', 'crj', 'snd', 'lft', 'dprime',
                 'rt', 'rt_hit', 'rt_fht', 'rt_snd', 'rt_lft',
                 'rtch', 'rtch_hit', 'rtch_fht', 'rtch_snd', 'rtch_lft',
                 'hrate', 'lrate', 'frate', 'sandwich', 'leftover']

aggregation_dict = dict(targ=sum, foil=sum, notg=sum, hit=sum, miss=sum,
                        fht=sum, sty=sum, fal=sum, crj=sum, snd=sum, lft=sum,
                        div=uniq, adj=uniq, idn=uniq, num=uniq, cond=uniq,
                        code=uniq, cond_code=uniq, subj=uniq, trial=uniq,
                        # catg=uniq, word=uniq,
                        rtch=chsq_rt, rtch_hit=chsq_rt, rtch_fht=chsq_rt,
                        rtch_snd=chsq_rt, rtch_lft=chsq_rt, rt=uniq_rt,
                        rt_hit=uniq_rt, rt_fht=uniq_rt, rt_snd=uniq_rt,
                        rt_lft=uniq_rt, sandwich=sum, leftover=sum)

# AGGREGATE BY TRIAL
trial_col_order = col_order_out + ['catg', 'word', 'tloc', 'floc', 'onset_og',
                                   'word_og', 'catg_og']
trial_aggr_dict = aggregation_dict
trial_aggr_dict.update(dict(catg=uniq, word=uniq, tloc=uniq, floc=uniq,
                            onset_og=uniq, word_og=uniq, catg_og=uniq))
grby = ['subj', 'trial']
by_trial = trialdata.groupby(grby)
by_trial = by_trial.aggregate(trial_aggr_dict)
by_trial['tloc'] = by_trial['tloc'].apply(literal_eval)
by_trial['floc'] = by_trial['floc'].apply(literal_eval)
by_trial['word_og'] = by_trial['word_og'].apply(literal_eval)
by_trial['catg_og'] = by_trial['catg_og'].apply(literal_eval)
by_trial['onset_og'] = by_trial['onset_og'].apply(literal_eval)
by_trial['hrate'] = by_trial['hit'].astype(float) / (by_trial['targ'])
by_trial['frate'] = by_trial['fht'].astype(float) / (by_trial['foil'])
by_trial['mrate'] = by_trial['fal'].astype(float) / (by_trial['notg'])
by_trial['srate'] = by_trial['snd'].astype(float) / (by_trial['sandwich'])
by_trial['lrate'] = by_trial['lft'].astype(float) / (by_trial['leftover'])
by_trial['dprime'] = efa.dprime(by_trial[['hit', 'miss', 'fal', 'crj']].values)
by_trial = by_trial[trial_col_order].sort(grby)
by_trial.to_csv(op.join(outdir, 'trialLevelData.tsv'), sep='\t', index=False)

# AGGREGATE BY CONDITION
grby = ['subj', 'div', 'adj', 'idn', 'num']
by_cond = trialdata.groupby(grby)
by_cond = by_cond.aggregate(aggregation_dict)
by_cond['hrate'] = by_cond['hit'].astype(float) / (by_cond['targ'])
by_cond['frate'] = by_cond['fht'].astype(float) / (by_cond['foil'])
by_cond['mrate'] = by_cond['fal'].astype(float) / (by_cond['notg'])
by_cond['srate'] = by_cond['snd'].astype(float) / (by_cond['sandwich'])
by_cond['lrate'] = by_cond['lft'].astype(float) / (by_cond['leftover'])
by_cond['dprime'] = efa.dprime(by_cond[['hit', 'miss', 'fal', 'crj']].values)
by_cond = by_cond[col_order_out].sort(grby)
#by_cond.to_csv(op.join(outdir, 'condLevelData.tsv'), sep='\t', index=False)

# AGGREGATE BY CONDITION IGNORING ATTN SAME/DIFF (idn)
grby = ['subj', 'div', 'adj', 'num']
by_cadj = trialdata.groupby(grby)
by_cadj = by_cadj.aggregate(aggregation_dict)
by_cadj['hrate'] = by_cadj['hit'].astype(float) / (by_cadj['targ'])
by_cadj['frate'] = by_cadj['fht'].astype(float) / (by_cadj['foil'])
by_cadj['mrate'] = by_cadj['fal'].astype(float) / (by_cadj['notg'])
by_cadj['srate'] = by_cadj['snd'].astype(float) / (by_cadj['sandwich'])
by_cadj['lrate'] = by_cadj['lft'].astype(float) / (by_cadj['leftover'])
by_cadj['dprime'] = efa.dprime(by_cadj[['hit', 'miss', 'fal', 'crj']].values)
by_cadj['code'][np.logical_and(by_cadj['div'], by_cadj['adj'])] = 'ajj'
by_cadj['code'][np.logical_and(by_cadj['div'],
                               np.logical_not(by_cadj['adj']))] = 'spp'
by_cadj['cond_code'] = ['_'.join(x) for x in zip(by_cadj['code'],
                                                 by_cadj['num'])]
by_cadj = by_cadj[col_order_out].sort(grby)
by_cadj = by_cadj[by_cadj['div']]  # don't count selective conds twice
#by_cadj.to_csv(op.join(outdir, 'condWithoutIdn.tsv'), sep='\t', index=False)

# AGGREGATE BY CONDITION IGNORING ADJ/SEP
grby = ['subj', 'div', 'idn', 'num']
by_cidn = trialdata.groupby(grby)
by_cidn = by_cidn.aggregate(aggregation_dict)
by_cidn['hrate'] = by_cidn['hit'].astype(float) / (by_cidn['targ'])
by_cidn['frate'] = by_cidn['fht'].astype(float) / (by_cidn['foil'])
by_cidn['mrate'] = by_cidn['fal'].astype(float) / (by_cidn['notg'])
by_cidn['srate'] = by_cidn['snd'].astype(float) / (by_cidn['sandwich'])
by_cidn['lrate'] = by_cidn['lft'].astype(float) / (by_cidn['leftover'])
by_cidn['dprime'] = efa.dprime(by_cidn[['hit', 'miss', 'fal', 'crj']].values)
by_cidn['code'][np.logical_and(by_cidn['div'], by_cidn['idn'])] = 'idn'
by_cidn['code'][np.logical_and(by_cidn['div'],
                               np.logical_not(by_cidn['idn']))] = 'dif'
by_cidn['cond_code'] = ['_'.join(x) for x in zip(by_cidn['code'],
                                                 by_cidn['num'])]
by_cidn = by_cidn[col_order_out].sort(grby)
by_cidn = by_cidn[by_cidn['div']]  # don't count selective conds twice
#by_cidn.to_csv(op.join(outdir, 'condWithoutAdj.tsv'), sep='\t', index=False)

# MERGE
by_all = pd.concat([by_cond.reset_index(drop=True),
                    by_cadj.reset_index(drop=True),
                    by_cidn.reset_index(drop=True)])
by_all = by_all.sort(['subj', 'div', 'code', 'num'])

#%% # # # # # # # # # # # # # # # # # #
# REPEAT ABOVE BUT IGNORING SIZE/NUM  #
# # # # # # # # # # # # # # # # # # # #
# AGGREGATE BY CONDITION
grby = ['subj', 'div', 'adj', 'idn']
by_cond = trialdata.groupby(grby)
by_cond = by_cond.aggregate(aggregation_dict)
by_cond['num'] = 'bth'
by_cond['hrate'] = by_cond['hit'].astype(float) / (by_cond['targ'])
by_cond['frate'] = by_cond['fht'].astype(float) / (by_cond['foil'])
by_cond['mrate'] = by_cond['fal'].astype(float) / (by_cond['notg'])
by_cond['srate'] = by_cond['snd'].astype(float) / (by_cond['sandwich'])
by_cond['lrate'] = by_cond['lft'].astype(float) / (by_cond['leftover'])
by_cond['dprime'] = efa.dprime(by_cond[['hit', 'miss', 'fal', 'crj']].values)
by_cond = by_cond[col_order_out].sort(grby)

# AGGREGATE BY CONDITION IGNORING ATTN SAME/DIFF (idn)
grby = ['subj', 'div', 'adj']
by_cadj = trialdata.groupby(grby)
by_cadj = by_cadj.aggregate(aggregation_dict)
by_cadj['num'] = 'bth'
by_cadj['hrate'] = by_cadj['hit'].astype(float) / (by_cadj['targ'])
by_cadj['frate'] = by_cadj['fht'].astype(float) / (by_cadj['foil'])
by_cadj['mrate'] = by_cadj['fal'].astype(float) / (by_cadj['notg'])
by_cadj['srate'] = by_cadj['snd'].astype(float) / (by_cadj['sandwich'])
by_cadj['lrate'] = by_cadj['lft'].astype(float) / (by_cadj['leftover'])
by_cadj['dprime'] = efa.dprime(by_cadj[['hit', 'miss', 'fal', 'crj']].values)
by_cadj['code'][np.logical_and(by_cadj['div'], by_cadj['adj'])] = 'ajj'
by_cadj['code'][np.logical_and(by_cadj['div'],
                               np.logical_not(by_cadj['adj']))] = 'spp'
by_cadj['cond_code'] = ['_'.join(x) for x in zip(by_cadj['code'],
                                                 by_cadj['num'])]
by_cadj = by_cadj[col_order_out].sort(grby)
by_cadj = by_cadj[by_cadj['div']]  # don't count selective conds twice

# AGGREGATE BY CONDITION IGNORING ADJ/SEP
grby = ['subj', 'div', 'idn']
by_cidn = trialdata.groupby(grby)
by_cidn = by_cidn.aggregate(aggregation_dict)
by_cidn['num'] = 'bth'
by_cidn['hrate'] = by_cidn['hit'].astype(float) / (by_cidn['targ'])
by_cidn['frate'] = by_cidn['fht'].astype(float) / (by_cidn['foil'])
by_cidn['mrate'] = by_cidn['fal'].astype(float) / (by_cidn['notg'])
by_cidn['srate'] = by_cidn['snd'].astype(float) / (by_cidn['sandwich'])
by_cidn['lrate'] = by_cidn['lft'].astype(float) / (by_cidn['leftover'])
by_cidn['dprime'] = efa.dprime(by_cidn[['hit', 'miss', 'fal', 'crj']].values)
by_cidn['code'][np.logical_and(by_cidn['div'], by_cidn['idn'])] = 'idn'
by_cidn['code'][np.logical_and(by_cidn['div'],
                               np.logical_not(by_cidn['idn']))] = 'dif'
by_cidn['cond_code'] = ['_'.join(x) for x in zip(by_cidn['code'],
                                                 by_cidn['num'])]
by_cidn = by_cidn[col_order_out].sort(grby)
by_cidn = by_cidn[by_cidn['div']]  # don't count selective conds twice

# MERGE
by_num = pd.concat([by_cond.reset_index(drop=True),
                    by_cadj.reset_index(drop=True),
                    by_cidn.reset_index(drop=True)])
by_num = by_num.sort(['subj', 'div', 'code'])

#%% # # # # # # # # # # # # # # # # # # # # # # # # #
# REPEAT ABOVE BUT IGNORING EVERYTHING BUT SIZE/NUM #
# # # # # # # # # # # # # # # # # # # # # # # # # # #
# AGGREGATE BY CONDITION
grby = ['subj', 'div', 'num']
by_cond = trialdata.groupby(grby)
by_cond = by_cond.aggregate(aggregation_dict)
by_cond['code'][by_cond['div']] = 'all'
by_cond['hrate'] = by_cond['hit'].astype(float) / (by_cond['targ'])
by_cond['frate'] = by_cond['fht'].astype(float) / (by_cond['foil'])
by_cond['mrate'] = by_cond['fal'].astype(float) / (by_cond['notg'])
by_cond['srate'] = by_cond['snd'].astype(float) / (by_cond['sandwich'])
by_cond['lrate'] = by_cond['lft'].astype(float) / (by_cond['leftover'])
by_cond['dprime'] = efa.dprime(by_cond[['hit', 'miss', 'fal', 'crj']].values)
by_cond = by_cond[col_order_out].sort(grby)
by_cond = by_cond[by_cond['div']]  # don't count selective conds twice

#%% FINAL MERGE
final = pd.concat([by_all, by_num, by_cond])
final = final.sort(['subj', 'div', 'code', 'num'])
final.to_csv(op.join(outdir, 'AggregatedFinal.tsv'), sep='\t', index=False)
