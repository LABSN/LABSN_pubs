# -*- coding: utf-8 -*-
"""
=====================================
Script 'Clean divAttnSemantic output'
=====================================

This script combines the experiment output (subject responses) with the MAT
files used to generate the experiment. Data are output in both a clean
tab-delimited format and a pandas-friendly CSV.
"""
# Author: Dan McCloy <drmccloy@uw.edu>
#
# License: BSD (3-clause)

import csv
import ast
import numpy as np
import pandas as pd
import os.path as op
from glob import glob
from itertools import chain
from scipy import io as sio


def clean_mat(md, flatten=False):
    """Remove unwanted dict entries in MAT files & optionally flatten values
    """
    for key in ('__globals__', '__header__', '__version__'):
        if key in md:
            del md[key]
    if flatten:
        md = {k: list(chain.from_iterable(v)) for k, v in md.iteritems()}
    return md


# path stuff
indir = 'rawData'
outdir = 'processedData'
vardir = 'variables'
andir = 'processedData/pandas'
infiles = glob(op.join(indir, '*.tab'))

# import MAT files
mat_dict = clean_mat(sio.loadmat(op.join(vardir, 'divAttnSemantic.mat')))
codes = mat_dict['codes'].tolist()
times = mat_dict['times'].tolist()
attn = mat_dict['attn'].tolist()
words = np.char.strip(np.char.encode(mat_dict['words'], 'ascii')).tolist()
cats = np.char.strip(np.char.encode(mat_dict['cats'], 'ascii')).tolist()

ctrl_codes = mat_dict['ctrl_codes'].tolist()
ctrl_times = mat_dict['ctrl_times'].tolist()
ctrl_attn = mat_dict['ctrl_attn'].tolist()
ctrl_words = np.char.strip(np.char.encode(mat_dict['ctrl_words'],
                                          'ascii')).tolist()
ctrl_cats = np.char.strip(np.char.encode(mat_dict['ctrl_cats'],
                                         'ascii')).tolist()
ctrl_targs = np.ma.masked_array(ctrl_words, np.logical_not(ctrl_codes),
                                fill_value='')
ctrl_targs = [x.compressed().tolist() for x in ctrl_targs]

targ_mask = np.logical_not(codes)
for m, a in zip(targ_mask, attn):
    m[np.where(np.logical_not(a))[0].tolist()] = True
dist_mask = np.logical_not(codes)
for m, a in zip(dist_mask, attn):
    m[np.where(a)[0].tolist()] = True
targ_words = np.ma.masked_array(words, targ_mask, fill_value='')
dist_words = np.ma.masked_array(words, dist_mask, fill_value='')
targ_times = np.ma.masked_array(times, targ_mask, fill_value=-1)
dist_times = np.ma.masked_array(times, dist_mask, fill_value=-1)
#targ_words_flat = [tw[np.argsort(tt.compressed())].compressed().tolist()
#                   for tw, tt in zip(targ_words, targ_times)]
#dist_words_flat = [dw[np.argsort(dt.compressed())].compressed().tolist()
#                   for dw, dt in zip(dist_words, dist_times)]
targ_words_flat = [tw.T.compressed().tolist() for tw in targ_words]
dist_words_flat = [dw.T.compressed().tolist() for dw in dist_words]

ctrl_targ_mask = np.logical_not(ctrl_codes)
for m, a in zip(ctrl_targ_mask, ctrl_attn):
    m[np.where(np.logical_not(a))[0].tolist()] = True
ctrl_dist_mask = np.logical_not(ctrl_codes)
for m, a in zip(ctrl_dist_mask, ctrl_attn):
    m[np.where(a)[0].tolist()] = True
ctrl_targ_words = np.ma.masked_array(ctrl_words, ctrl_targ_mask, fill_value='')
ctrl_dist_words = np.ma.masked_array(ctrl_words, ctrl_dist_mask, fill_value='')
ctrl_targ_times = np.ma.masked_array(ctrl_times, ctrl_targ_mask, fill_value=-1)
ctrl_dist_times = np.ma.masked_array(ctrl_times, ctrl_dist_mask, fill_value=-1)
ctrl_targ_words_flat = [tw.T.compressed().tolist() for tw in ctrl_targ_words]
ctrl_dist_words_flat = [dw.T.compressed().tolist() for dw in ctrl_dist_words]
#ctrl_targ_words_flat = [tw[np.argsort(tt.compressed())].compressed().tolist()
#                        for tw, tt in zip(ctrl_targ_words, ctrl_targ_times)]
#ctrl_dist_words_flat = [dw[np.argsort(dt.compressed())].compressed().tolist()
#                        for dw, dt in zip(ctrl_dist_words, ctrl_dist_times)]


#del mat_dict

# what is the main grouping indicator in the output file?
group_by = 'trial'
# which lines between grouping indicators should be retained?
lines_to_keep = ['target_times', 'distractor_times', 'press_times',
                 'false_alarm_times', 'target_RTs', 'distractor_RTs']

# extra columns to add from MAT files
extra_vars = {'codes': codes, 'words': words, 'times': times, 'cats': cats,
              'attn': attn, 'targ_words': targ_words_flat,
              'dist_words': dist_words_flat}
ctrl_vars = {'codes': ctrl_codes, 'words': ctrl_words, 'times': ctrl_times,
             'cats': ctrl_cats, 'attn': ctrl_attn,
             'targ_words': ctrl_targ_words_flat,
             'dist_words': ctrl_dist_words_flat}

# column headings
header = [group_by] + lines_to_keep + extra_vars.keys()
phase = ''

for infile in infiles:
    trial = []
    targ_times = []
    dist_times = []
    press_times = []
    false_times = []
    targ_rt = []
    dist_rt = []
    ctrl_trial = []
    ctrl_targ_times = []
    ctrl_dist_times = []
    ctrl_press_times = []
    ctrl_false_times = []
    ctrl_targ_rt = []
    ctrl_dist_rt = []
    test_to_keep = {'trial': trial,
                    'target_times': targ_times,
                    'distractor_times': dist_times,
                    'press_times': press_times,
                    'false_alarm_times': false_times,
                    'target_RTs': targ_rt,
                    'distractor_RTs': dist_rt}
    ctrl_to_keep = {'trial': ctrl_trial,
                    'target_times': ctrl_targ_times,
                    'distractor_times': ctrl_dist_times,
                    'press_times': ctrl_press_times,
                    'false_alarm_times': ctrl_false_times,
                    'target_RTs': ctrl_targ_rt,
                    'distractor_RTs': ctrl_dist_rt}
    outfile = op.join(outdir, op.basename(infile))
    with open(outfile, 'w') as g:
        g.write('\t'.join(header) + '\n')
        with open(infile, 'r') as f:
            csvr = csv.reader(f, delimiter='\t')
            out_line = [''] * len(header)
            for row in csvr:
                if 'phase' in row:
                    phase = row[-1]
                for cell in row:
                    if cell == group_by:  # new grouping marker
                        current_trial = row[-1]
                        if len(set(out_line)) > 1:  # prev group line finished
                            g.write('\t'.join(out_line) + '\n')
                        out_line = [''] * len(header)
                        out_line[0] = current_trial
                        if 'ctrl_trials' in phase:
                            ctrl_trial.append(current_trial)
                            var_dict = ctrl_vars
                        else:
                            trial.append(current_trial)
                            var_dict = extra_vars
                        for key, var in var_dict.iteritems():
                            out_line[header.index(key)] = \
                                str(var[int(row[-1])])

                    elif cell in lines_to_keep:
                        # parse string representation of python object
                        value = ast.literal_eval(row[-1])
                        if 'ctrl_trials' in phase:
                            ctrl_to_keep[cell].append(value)
                        else:
                            test_to_keep[cell].append(value)
                        if isinstance(value, list):
                            value = [str(x) for x in value]
                        out_line[header.index(cell)] = ','.join(value)
            g.write('\t'.join(out_line) + '\n')

    # assemble into dataframe
    col_order = ['phase', 'trial', 'target_times', 'distractor_times',
                 'press_times', 'false_alarm_times', 'target_RTs',
                 'distractor_RTs', 'targ_words', 'dist_words', 'words',
                 'times', 'attn', 'cats', 'codes']
    df_test = pd.DataFrame(dict(extra_vars.items() + test_to_keep.items()))
    df_ctrl = pd.DataFrame(dict(ctrl_vars.items() + ctrl_to_keep.items()))
    df_test['phase'] = 'test'
    df_ctrl['phase'] = 'ctrl'
    df = pd.concat((df_test, df_ctrl))
    df = df[col_order]
    df.to_csv(op.join(andir, op.basename(infile)), sep='\t', index=False)
    df.to_pickle(op.join(andir, op.splitext(op.basename(infile))[0] + '.pickle'))
