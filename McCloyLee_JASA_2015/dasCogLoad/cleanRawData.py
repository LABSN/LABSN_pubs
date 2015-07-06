# -*- coding: utf-8 -*-
"""
=============================
Script 'Clean expyfun output'
=============================

This script combines the experiment output (subject responses) with the trial
data used to generate the experiment (imported from an NPZ file). Data are
output as tab-delimited text.
"""
# Author: Dan McCloy <drmccloy@uw.edu>
#
# License: BSD (3-clause)

import csv
import ast
import numpy as np
import os.path as op
from glob import glob


indir = 'rawData'
outdir = 'processedData'
outfile = op.join(outdir, 'processedData.tsv')
varsfile = op.join(outdir, 'expVariables.npz')
infiles = glob(op.join(indir, '*.tab'))

# READ IN VARIABLES
var_dict = np.load(varsfile)
"""
SCALARS:    trials, waves, streams, isi
LISTS:      angles, targs_per_trial, foils_per_trial, total_per_trial
ARRAYS:     targs, foils (120,)
            cats, attn (120, 4)
            words, onset_sec, onset_samp, targ_loc, foil_loc (120, 4, 12)
TRAINING:   tn_one_attn     tn_one_words        tn_one_targ_loc
            tn_two_attn     tn_two_words        tn_two_targ_loc
            tn_four_a_attn  tn_four_a_words     tn_four_a_targ_loc
            tn_four_aa_attn tn_four_aa_words    tn_four_aa_targ_loc
            tn_four_ab_attn tn_four_ab_words    tn_four_ab_targ_loc
"""

# what is the main grouping indicator in the output file?
group_by = 'trial_id'

# which lines between grouping indicators should be retained?
lines_to_keep = ['target_times', 'foil_times', 'press_times',
                 'false_alarm_times', 'target_RTs', 'foil_RTs']

# variables to pull in from varsfile
attn = var_dict['tr_attn']
cats = var_dict['tr_cats']
words = var_dict['tr_words']
targ_loc = var_dict['tr_targ_loc']
foil_loc = var_dict['tr_foil_loc']
onsets = var_dict['tr_onset_sec']
additional_vars = dict(attn=attn, cats=cats, words=words, targ_loc=targ_loc,
                       foil_loc=foil_loc, onsets=onsets)

# any desired metadata from infile header
meta = dict(subj_num='', subj='', datestring='')

# outfile header
header = [group_by] + meta.keys() + lines_to_keep + additional_vars.keys()

with open(outfile, 'w') as g:
    g.write('\t'.join(header) + '\n')
    for infile in infiles:
        with open(infile, 'r') as f:
            outline = [''] * len(header)
            csvr = csv.reader(f, delimiter='\t')
            for row in csvr:
                if row[0][0] == '#':
                    metadata = ast.literal_eval(row[0][2:])
                    meta['subj'] = metadata['participant']
                    meta['subj_num'] = metadata['session']
                    meta['datestring'] = metadata['date']
                elif row[1] == group_by:  # start of new trial
                    try:
                        current_trial = int(row[-1], base=2)  # parse binary
                    except ValueError:
                        current_trial = None  # training
                    if len(set(outline)) > 1:  # old trial data needs writing
                        g.write('\t'.join(outline) + '\n')
                    if current_trial is not None:
                        outline = [''] * len(header)
                        outline[0] = str(current_trial)
                        if int(meta['subj_num']) % 2 == 1:
                            if current_trial > 59:
                                current_trial -= 60
                            else:
                                current_trial += 60
                        for k, v in additional_vars.items():
                            value = str(v[current_trial]).replace('\n', '')
                            outline[header.index(k)] = value
                        for field in meta:
                            value = meta[field].replace('\n', '')
                            outline[header.index(field)] = value
                elif row[1] in lines_to_keep:
                    try:
                        value = str(ast.literal_eval(row[-1]))
                    except SyntaxError:
                        value = row[-1]
                    outline[header.index(row[1])] = value
            g.write('\t'.join(outline) + '\n')
