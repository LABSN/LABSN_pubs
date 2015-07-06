# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'Calculate chance and ceiling for divAttnSem and dasCogLoad'
===============================================================================

This script calculates d-prime values for perfect performance and various
definitions of chance performance, for two related experiments.
"""
# Author: Dan McCloy <drmccloy@uw.edu>
# License: BSD (3-clause)

import os.path as op
import numpy as np
import expyfun as ef
import pandas as pd
pd.set_option('display.width', 150)
np.set_printoptions(linewidth=150)

indir = 'processedData'

# # # # # # # # # # # # # # # #
# which data set to analyze?  #
# # # # # # # # # # # # # # # #
das = True    # dasCogLoad
#das = False  # divAttnSem

# dasCogLoad data
infile = op.join(indir, 'processedData.tsv')
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
        l = [str(l[2]),  # subj
             [float(x) for x in l[4].strip('[]').split()],  # target_times
             [float(x) for x in l[5].strip('[]').split()]   # foil_times
             ]
        data.append(l)

cnames = [colnames[x] for x in (2, 4, 5)]
df = pd.DataFrame(data, columns=cnames)
df['targs'] = df['target_times'].apply(len)
df['foils'] = df['foil_times'].apply(len)
df = df[['subj', 'targs', 'foils']]

# divAttnSemantic data
bdata = pd.read_csv(op.join(indir, 'divAttnSemData.tsv'), sep='\t')


# # # # # # # # # # # # # # # # #
# reduce to one subjects worth  #
# # # # # # # # # # # # # # # # #
foo = df if das else bdata[['subj', 'targs', 'foils']]
foo = foo[foo['subj'] == 'DU']

perfect = foo.copy()
perfect['hit'] = perfect['targs']
perfect['miss'] = 0
perfect['fal'] = 0
perfect['crj'] = 12 - perfect['targs']
perfect_hmfc = perfect[['hit', 'miss', 'fal', 'crj']]
perfect_dprime = ef.analyze.dprime(perfect_hmfc.sum().values)

allodd = foo.copy()
allodd['hit'] = allodd['targs']
allodd['miss'] = 0
allodd['fal'] = allodd['foils']
allodd['crj'] = 12 - allodd['targs'] - allodd['foils']
allodd_hmfc = allodd[['hit', 'miss', 'fal', 'crj']]
allodd_dprime = ef.analyze.dprime(allodd_hmfc.sum().values)

attnonly = perfect_hmfc.sum()
presses = attnonly['hit']
rate = presses / float(attnonly.sum())
attnonly['hit'] = int(round(attnonly['hit'] * rate))
attnonly['miss'] = presses - attnonly['hit']
attnonly['fal'] = presses - attnonly['hit']
attnonly['crj'] = attnonly['crj'] - attnonly['fal']
attnonly_dprime = ef.analyze.dprime(attnonly.values)

chance = perfect_hmfc.sum()
presses = chance['hit']
rate = 0.5
chance['hit'] = int(round(chance['hit'] * rate))
chance['miss'] = presses - chance['hit']
chance['fal'] = presses - chance['hit']
chance['crj'] = chance['crj'] - chance['fal']
chance_dprime = ef.analyze.dprime(chance.values)

chance2 = perfect_hmfc.sum()
presses = chance2['hit']
crjs = chance2['crj']
rate = 0.5
chance2['hit'] = int(round(chance2['hit'] * rate))
chance2['miss'] = presses - chance2['hit']
chance2['crj'] = int(round(chance2['crj'] * rate))
chance2['fal'] = presses - chance2['hit'] + crjs - chance2['crj']
chance2_dprime = ef.analyze.dprime(chance2.values)

chanceall = perfect_hmfc.sum()
chanceall['fal'] = chanceall['crj']
chanceall['crj'] = 0
chanceall_dprime = ef.analyze.dprime(chanceall.values)

print('dprime summary')
print('{0:+.2f} (perfect)'.format(perfect_dprime))
print('{0:+.2f} (presses for all targs and all foils)'.format(allodd_dprime))
print('{0:+.2f} (hits 1/2 targs, with equal # of FAs)'.format(chance_dprime))
print('{0:+.2f} (press rate = targ rate)'.format(attnonly_dprime))
print('{0:+.2f} (press rate = 0.5 in each 1-sec frame)'.format(chance2_dprime))
print('{0:+.2f} (1 press in each 1-sec frame)'.format(chanceall_dprime))
