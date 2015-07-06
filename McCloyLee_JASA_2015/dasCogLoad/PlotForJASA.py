# -*- coding: utf-8 -*-
"""
================================
Script 'Plot DAS-cog-load: JASA'
================================

This script plots all the figures for the JASA paper.
"""
# Author: Dan McCloy <drmccloy@uw.edu>
#
# License: BSD (3-clause)

import json
import os.path as op
import numpy as np
import pandas as pd
import scipy.stats as ss
import expyfun.analyze as efa
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import svgutils.transform as svgt
from matplotlib import rcParams, font_manager
from ast import literal_eval
from itertools import chain as ch
pd.set_option('display.width', 160)
np.set_printoptions(linewidth=160)


def bonferroni(pvalues):
    return pvalues * np.array(pvalues).ravel().shape[0]


def holm(pvalues):
    new_pvals = np.ones_like(pvalues)
    vals = [(pv, i) for i, pv in enumerate(pvalues)]
    vals.sort()
    for rank, val in enumerate(vals):
        pval, i = val
        new_pvals[i] = (pvalues.shape[0] - rank) * pval
        if new_pvals[i] > 0.05:
            break
    return new_pvals


def uniq(x):
    # return (list of) unique value(s)
    y = list(set(x))
    if len(y) == 1:
        y = y[0]
    return y


def flat(x):
    # return list of all values
    y = list(ch.from_iterable(x.tolist()))
    #y = np.array([y, []], dtype=object)  # hack to get shape == (2,)
    #y = np.atleast_2d(y[:-1])            # hack to get shape == (1, 1)
    return y


def chsq_rt(x):
    # peak of chi squared distribution fit to data
    y = nonan(x)
    if y.size == 0:
        return np.nan
    else:
        return efa.rt_chisq(y)


def nonan(x):
    x = np.array(x).ravel()
    return x[np.logical_not(np.isnan(x))]


def parse_cond(ss):
    ss = str(ss).zfill(4)
    return [int(x) for x in list(ss)]


def parse_strings(ws):
    return literal_eval(ws)


def parse_rawrt(st, return_bool=False):
    arr = [x.replace('[', '').replace(']', '').split() for x in st.split('\n')]
    arr = np.array(arr, dtype=float)
    if return_bool:
        arr = np.logical_not(np.isnan(arr))
    return arr


def parse_loc(st):
    arr = [x.replace('[', '').replace(']', '').split() for x in st.split('\n')]
    arr = [[False if y == 'False' else True for y in x] for x in arr]
    return np.array(arr, dtype=bool)


def sum_bools(arr, axis=-1, usenan=False):
    result = np.sum(arr, axis=axis).astype(float)
    if usenan:
        result[result == 0] = np.nan
    return result


def find_min_pval(coefs, table):
    table['pv'] = [efa.format_pval(x, latex=False) for x in table['pval']]
    pvdict = {k: v for k, v in zip(table['predictor'], table['pval'])}
    relevant_pvals = table[np.in1d(table['predictor'], coefs)]
    minpv_ix = np.where(relevant_pvals['pval'] == relevant_pvals['pval'].min())
    minpv_coef = relevant_pvals['predictor'].values[minpv_ix[0][0]]
    return [pvdict[minpv_coef]]


figdir = op.join('figures', 'jasa')
indir = 'processedData'
outext = '.svg'
#outext = '.pdf'

# divAttnSemantic data
bdata = pd.read_csv(op.join(indir, 'divAttnSemData.tsv'), sep='\t')

# dasCogLoad data
adata = pd.read_csv(op.join(indir, 'AggregatedFinal.tsv'), sep='\t')

# CHANCE AND CEILING
ceilchance_das = (6.5, 1.13)
ceilchance_dcl = (6.26, 1.12)

# MIXED MODEL COEFFICIENT P-VALUES
pvals_das = pd.read_csv('documents/stats/divAttnSem_pvals.tsv', sep='\t')
pvals_dcl = pd.read_csv('documents/stats/dasCogLoad_pvals.tsv', sep='\t')

#%% # # # # # # # # # # #
# BASIC PLOTTING SETUP  #
# # # # # # # # # # # # #
rcp = {'font.sans-serif': ['Source Sans Pro'], 'font.style': 'normal',
       'font.size': 14, 'font.variant': 'normal', 'font.weight': 'medium',
       'pdf.fonttype': 42}

# plot parameters
lkw = dict(color='k', alpha=0.3)
ekw = dict(capsize=5, linewidth=1.5, capthick=1.5)
fkw = dict(size=16, xytext=(0, -4))  # size and offset of asterisks
ylim = (0., 7.)      # axis range for dprime
fylim = (0, 0.3)     # axis range for foil response rates
fstep = 0.1           # tick size for foil response rates
rtylim = (0.25, 1.)  # axis range for reaction times
rtstep = 0.25        # tick size for reaction times
pv_scheme = 'stars'
latex = False
fourway = np.array([(0, 1), (2, 3), (0, 2), (1, 3)])  # bracket specification

# COLOR SPEC
wht = ['#FFFFFF', '#777777']  # fill, line
lgr = ['#AAAAAA', '#444444']  # fill, line
dgr = ['#777777', '#111111']  # fill, line
bgr = '#DDDDDD'  # background gray
gry = ['#AAAAAA', '#777777']  # ['WhiteSmoke', 'Silver']
red = ['#DD7788', '#AA4455']  # ['Crimson', 'LightCoral']
grn = ['#88CCAA', '#44AA77']  # ['ForestGreen', 'LightGreen']
blu = ['#77AADD', '#4477AA']  # ['DodgerBlue', 'LightSkyBlue']

#%% # # # # # # # # # # #
# EXAMPLE TRIAL DIAGRAM #
# # # # # # # # # # # # #
# reset things that have changed
plt.rcdefaults()
rcParams.update(rcp)
# load data
tdata = pd.read_csv(op.join(indir, 'trialLevelData.tsv'), sep='\t')
tdata['cond'] = tdata['cond'].apply(parse_cond)
tdata['word_og'] = tdata['word_og'].apply(parse_strings)
tdata['catg_og'] = tdata['catg_og'].apply(parse_strings)
tdata['onset_og'] = tdata['onset_og'].apply(parse_strings)
tdata['tloc'] = tdata['tloc'].apply(parse_strings)
tdata['floc'] = tdata['floc'].apply(parse_strings)
# setup
trialnum = 47  # adj: 14, 33, 35, 41, 49 sep: 38, 45, 47
rcParams['font.size'] = 10
fp_md = font_manager.FontProperties(weight='medium')
fp_small = font_manager.FontProperties(size=8)
fp_tiny = font_manager.FontProperties(size=6)
fp_tinybold = font_manager.FontProperties(size=6, weight='bold')
#fp = font_manager.FontProperties(font.size=11)

# initialize figure
fig, trial_ax = plt.subplots(1, 1, figsize=(7, 1.75))  # (18, 3)
# lines & category names
theta = np.array([60, 15, -15, -60]) * np.pi / 180.
catnames = [x.upper() for x in tdata['catg_og'][trialnum]]
xcen = -3.25
ycen = 2.5
radii = [1.6, 1.25, 1.25, 1.6]
voffsets = ['bottom', 'bottom', 'top', 'top']
atn = np.array(tdata['cond'][trialnum], dtype=bool)
colors = [grn[0] if x else '#777777' for x in atn]
for th, cn, rd, vo, cl in zip(theta, catnames, radii, voffsets, colors):
    x = np.array([xcen, xcen + rd * np.cos(th)])
    y = np.array([ycen, ycen + rd * np.sin(th)])
    p = trial_ax.plot(x, y, color='k')
    q = trial_ax.annotate(cn, (x[1], y[1]), xytext=(4, 0), ha='left',
                          va=vo, textcoords='offset points',
                          fontproperties=fp_tinybold, color=cl)
# TRIAL TIME COURSE
with open(op.join(indir, 'wordDurations.json')) as jd:
    word_durs = json.load(jd)
xlim = (-3.75, 12.5)
# highlight attended streams
xb = (0, 12.5)  # (-2.125, 12.5)
for ix, stream in enumerate(atn):
    if stream:
        _ = trial_ax.fill_between(xb, 4.5 - ix, 3.5 - ix, where=None,
                                  facecolor='White', color='none', alpha=1)
    else:
        _ = trial_ax.fill_between(xb, 4.5 - ix, 3.5 - ix, where=None,
                                  facecolor='#E6E6E6', color='none', alpha=1)
# grid
trial_ax.spines['left'].set_position('zero')
trial_ax.spines['left'].set_linewidth(0.5)
#trial_ax.spines['bottom'].set_linewidth(0.5)
trial_ax.spines['bottom'].set_color('none')
plt.hlines(0.5, 0, 12.5, color='k', linewidth=0.5, zorder=4)
trial_ax.spines['right'].set_color('none')
trial_ax.spines['top'].set_color('none')
#trial_ax.grid(True, axis='y', which='minor', color='0.7', linestyle='-')
plt.hlines(np.arange(0.5, 5, 1), 0, 12.5, colors='#CCCCCC', linewidth=0.25,
           zorder=1)
plt.vlines(np.arange(0, 12.75, 0.25), 0.5, 4.5, colors='#CCCCCC',
           linewidth=0.25, zorder=1)
#plt.vlines(-2.5, 0.5, 4.5, colors='0.75', linewidth=0.25)  # prime onset
trial_ax.tick_params(axis='both', which='both', bottom='off', top='off',
                     left='off', right='off')  # tick marks off
trial_ax.tick_params(axis='x', which='both', bottom='on', top='off')
                     #direction='out')
trial_ax.set_axisbelow(True)

# word boxes
w = np.array(tdata['word_og'][trialnum])
t = w[np.array(tdata['tloc'][trialnum], dtype=bool)]
f = w[np.array(tdata['floc'][trialnum], dtype=bool)]
d = np.array([[word_durs[u] for u in v] for v in w]).ravel()
x = np.array(tdata['onset_og'][trialnum]).ravel()
y = np.tile(np.arange(4, 0, -1), (12, 1)).T.ravel()
w = w.ravel()
for word, xy, dur in zip(w, zip(x, y), d):
    if word in t:
        c = ['#FFFFFF', grn[1]]  # grn[::-1]  # 'ForestGreen'
    elif word in f:
        c = ['#FFFFFF', red[1]]  # red[::-1]  # 'Red'
    else:
        c = ['#777777', '#CCCCCC']  # ['DimGray', 'LightGray']
    _ = trial_ax.fill_between([xy[0], xy[0] + dur], xy[1] - 0.25, xy[1] + 0.25,
                              where=None, facecolor=c[1], edgecolor='none',
                              zorder=2)  # linewidth=0.25)
    _ = trial_ax.annotate(word, xy, textcoords='offset points', xytext=(2, 0),
                          va='center', ha='left', color=c[0],
                          fontproperties=fp_small, zorder=3)
# garnish
trial_ax.tick_params(length=0)
_ = trial_ax.set_xticks(range(13))
#_ = trial_ax.set_xticks([-2.5] + [z / 4.0 for z in range(49)], minor=True)
_ = trial_ax.set_yticks([z + 0.5 for z in range(4)], minor=True)
_ = trial_ax.set_yticks(range(1, 5, 1))
angle_text = [u'60°', u'15°', u'−15°', u'−60°']  # works with F5, not F9 ?!
_ = trial_ax.set_yticklabels(angle_text)
_ = trial_ax.set_xlabel('time (s)')
_ = trial_ax.xaxis.set_label_coords(6.25, -0.25,
                                    transform=trial_ax.transData)
_ = trial_ax.set_xbound(xlim)
_ = trial_ax.set_ybound(0.5, 4.5)
# intermediate finish
plt.tight_layout(pad=0.75)
plt.draw()
plt.savefig(op.join(figdir, 'trial-timecourse-tmp.svg'))
# add head and screenshot: create new SVG figure; load subfigures
fig = svgt.SVGFigure('7in', '1.75in')
fig1 = svgt.fromfile(op.join('figures', 'jasa', 'trial-timecourse-tmp.svg'))
fig2 = svgt.fromfile(op.join('figures', 'dan_head.svg'))
fig3 = svgt.fromfile(op.join('figures', 'DASCogLoadScreenshot.svg'))
# add head and screenshot: get the plot objects
plot1 = fig1.getroot()
plot2 = fig2.getroot()
plot3 = fig3.getroot()
plot1.moveto(0, 0, scale=1.25)
plot2.moveto(10, 41, scale=0.10625)
plot3.moveto(16, 112, scale=0.0425)
# add head and screenshot: add text labels
txt1 = svgt.TextElement(4, 15, 'a)', size=14, font='Source Code Pro')
txt2 = svgt.TextElement(96, 15, 'b)', size=14, font='Source Code Pro')
# add head and screenshot: append plots and labels to figure
fig.append([plot1, plot2, plot3, txt1, txt2])
fig.save(op.join('figures', 'jasa', 'trial-timecourse.svg'))

#%% # # # # # # # # # # #
# divAttnSem data setup #
# # # # # # # # # # # # #
# gather the data we want to plot
das = bdata[['subj', 'sem', 'div', 'adj', 'hit', 'miss', 'fal', 'crj', 'fht',
             'targs', 'foils']]
das = das.reset_index(drop=True)
# phonetic, semantic
das_phosem = das.groupby(['subj', 'sem']).aggregate(sum)
das_phosem['dprime'] = efa.dprime(das_phosem[['hit', 'miss', 'fal', 'crj']
                                             ].values.astype(int))
das_phosem_pt = pd.pivot_table(das_phosem.reset_index(),
                               'dprime', 'subj', 'sem')
# selective, separated, adjacent
das_slajsp = das.groupby(['subj', 'div', 'adj']).aggregate(sum)
das_slajsp['dprime'] = efa.dprime(das_slajsp[['hit', 'miss', 'fal', 'crj']
                                             ].values.astype(int))
das_slajsp_pt = pd.pivot_table(das_slajsp.reset_index(),
                               'dprime', 'subj', ['div', 'adj'])
# pho/sem X sel/sep/adj
das_inter = das.groupby(['subj', 'sem', 'div', 'adj']).aggregate(sum)
das_inter['dprime'] = efa.dprime(das_inter[['hit', 'miss', 'fal', 'crj']
                                           ].values.astype(int))
das_inter_pt = pd.pivot_table(das_inter.reset_index(),
                              'dprime', 'subj', ['div', 'sem', 'adj'])
# pho/sem X sep/adj; foil rate
das_foil = das.groupby(['subj', 'sem', 'div', 'adj']).aggregate(sum)
das_foil['frate'] = das_foil['fht'] / das_foil['foils'].astype(float)
das_foil_pt = pd.pivot_table(das_foil.reset_index(),
                             'frate', 'subj', ['div', 'sem', 'adj'])

#%% # # # # # # # # # # # #
# divAttnSem Main Effects #
# # # # # # # # # # # # # #
""" pho vs sem  |  sel vs (adj vs sep) """
# reset things that have changed
plt.rcdefaults()
rcParams.update(rcp)
# INITIALIZE FIGURE
fig, (sub_a, sub_b) = plt.subplots(1, 2, figsize=(7, 3.5))

# pho vs sem stats
relevant_coefs = ('semantic', 'target:semantic', 'foil:semantic')
corr_pval = find_min_pval(relevant_coefs, table=pvals_das)
form_pval = efa.format_pval(corr_pval, scheme=pv_scheme, latex=latex)
signif = [pv < 0.05 for pv in corr_pval]
# pho vs sem plot
rkw = dict(color=[lgr[0]] + [dgr[0]], linewidth=1,
           edgecolor=[lgr[1]] + [dgr[1]])
ax_a, bar_a = efa.barplot(das_phosem_pt, axis=0, err_bars='se',
                          #groups=[(0, 1), (2, 3)],
                          brackets=np.array([(0, 1)])[np.where(signif)],
                          bracket_text=form_pval[np.where(signif)],
                          bracket_inline=True,
                          bar_names=['phonetic', 'semantic'],
                          #group_names=['selective', 'divided'],
                          bar_kwargs=rkw, err_kwargs=ekw, pval_kwargs=fkw,
                          ax=sub_a, ylim=ylim)
_ = ax_a.yaxis.set_label_text('d-prime')
_ = ax_a.yaxis.set_tick_params('major', labelsize=12)
# add the d-prime chance & ceiling as grey dashed lines
_ = ax_a.axhspan(0, ceilchance_das[1], color=bgr, zorder=-1)
_ = ax_a.axhspan(ceilchance_das[0], sub_a.get_ybound()[1], color=bgr,
                 zorder=-1)

# sel vs div stats
relevant_coefs = ('selective', 'target:selective', 'foil:selective')
corr_pval = find_min_pval(relevant_coefs, table=pvals_das)
# adj vs sep stats
relevant_coefs = ('separated', 'target:separated', 'foil:separated')
corr_pval = corr_pval + find_min_pval(relevant_coefs,
                                      table=pvals_das)  # list concat, not add
form_pval = efa.format_pval(corr_pval, scheme=pv_scheme, latex=latex)
signif = [pv < 0.05 for pv in corr_pval]
# sel vs div, adj vs sep plot
rkw = dict(color=wht[0], linewidth=1, edgecolor=wht[1])
brackets = np.array([([0], [1, 2]), (1, 2)])[np.where(signif)[0]][::-1]
br_text = form_pval[np.where(signif)[0]][::-1]  # reverse order: better drawing
ax_b, bar_b = efa.barplot(das_slajsp_pt, axis=0, err_bars='se',
                          groups=[[0], [1, 2]],
                          eq_group_widths=True,
                          brackets=brackets,
                          bracket_text=br_text,
                          bracket_inline=True,
                          bracket_group_lines=True,
                          bar_names=['', 'sep.', 'adj.'],
                          group_names=['selective', 'divided'],
                          bar_kwargs=rkw, err_kwargs=ekw, pval_kwargs=fkw,
                          ax=sub_b, ylim=ylim)
_ = ax_b.yaxis.set_ticklabels([])
for b, h in zip(bar_b, ['\\', '', '//']):
    b.set_hatch(h)
# add the d-prime chance & ceiling as grey dashed lines
_ = ax_b.axhspan(0, ceilchance_das[1], color=bgr, zorder=-1)
_ = ax_b.axhspan(ceilchance_das[0], sub_b.get_ybound()[1], color=bgr,
                 zorder=-1)
# finish plot & save
plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
plt.subplots_adjust(bottom=0.2)
_ = ax_a.text(-0.4, ylim[-1], 'a)')
_ = ax_b.text(-0.3, ylim[-1], 'b)')
plt.savefig(op.join(figdir, 'das-main-effects' + outext))

#%% # # # # # # # # # # # #
# divAttnSem Interactions #
# # # # # # # # # # # # # #
""" (pho vs sem) vs (sel vs (adj vs sep)) """
# reset things that have changed
plt.rcdefaults()
rcParams.update(rcp)
# INITIALIZE FIGURE
fig, sub_a = plt.subplots(1, 1, figsize=(7, 3.5))

# pho/sem X sel stats
relevant_coefs = ('semantic:selective', 'target:semantic:selective',
                  'foil:semantic:selective')
#pvals_das[np.in1d(pvals_das['predictor'], relevant_coefs)]
corr_pval = find_min_pval(relevant_coefs, table=pvals_das)
# pho/sem X adj/sep stats
relevant_coefs = ('semantic:separated', 'target:semantic:separated',
                  'foil:semantic:separated')
corr_pval = corr_pval + find_min_pval(relevant_coefs,
                                      table=pvals_das)  # list concat, not add
form_pval = efa.format_pval(corr_pval, scheme=pv_scheme, latex=latex)
signif = [pv < 0.05 for pv in corr_pval]
# pho vs sem plot
colseq = [lgr[0]] + [dgr[0]] + 2 * [lgr[0]] + 2 * [dgr[0]]
linseq = [lgr[1]] + [dgr[1]] + 2 * [lgr[1]] + 2 * [dgr[1]]
rkw = dict(color=colseq, linewidth=1, edgecolor=linseq)
ax_a, bar_a = efa.barplot(das_inter_pt, axis=0, err_bars='se',
                          groups=[[0, 1], [2, 3, 4, 5]],
                          eq_group_widths=True,
                          #brackets=[brackets],
                          #bracket_text=br_text,
                          #bracket_inline=True,
                          #bracket_group_lines=True,
                          bar_names=['phonetic', 'semantic', 'sep.', 'adj.',
                                     'sep.', 'adj.'],
                          group_names=['selective',
                                       'phonetic             semantic'],
                          bar_kwargs=rkw, err_kwargs=ekw, pval_kwargs=fkw,
                          ax=sub_a, ylim=ylim)
_ = ax_a.yaxis.set_label_text('d-prime')
_ = ax_a.yaxis.set_tick_params('major', labelsize=12)
for b, h in zip(bar_a, ['\\', '\\', '', '//', '', '//']):
    b.set_hatch(h)
# custom brackets
pval_kwargs = dict(ha='center', va='center', xytext=(0, -3),
                   annotation_clip=False, textcoords='offset points')
# first bracket
xll = bar_a.patches[2].get_x() + 0.5 * bar_a.patches[2].get_width()
xlr = bar_a.patches[3].get_x() + 0.5 * bar_a.patches[3].get_width()
xrl = bar_a.patches[4].get_x() + 0.5 * bar_a.patches[4].get_width()
xrr = bar_a.patches[5].get_x() + 0.5 * bar_a.patches[5].get_width()
xlm = (xll + xlr) / 2.
xrm = (xrl + xrr) / 2.
xm = (xlm + xrm) / 2.
adj = 0.5
yl = adj + max([bar_a.patches[num].get_height() for num in [2, 3]])
yr = adj + max([bar_a.patches[num].get_height() for num in [4, 5]])
ym = 0.8 * adj + max([yl, yr])
_ = ax_a.plot((xll, xlr), (yl, yl), color='0.3')  # left  horz
_ = ax_a.plot((xrl, xrr), (yr, yr), color='0.3')  # right horz
_ = ax_a.plot((xlm, xlm), (yl, ym), color='0.3')  # left  vert
_ = ax_a.plot((xrm, xrm), (yr, ym), color='0.3')  # right vert
txt = ax_a.annotate(form_pval[1], (xm, ym), **pval_kwargs)
txt.set_bbox(dict(facecolor='w', alpha=0, boxstyle='round, pad=0.25'))
plt.draw()
bb = txt.get_bbox_patch().get_window_extent()
txth = np.diff(ax_a.transData.inverted().transform(bb), axis=0).ravel()[-1]
txtw = np.diff(ax_a.transData.inverted().transform(bb), axis=0).ravel()[0]
xml = np.mean([xlm, xrm]) - txtw / 2.
xmr = np.mean([xlm, xrm]) + txtw / 2.
_ = ax_a.plot((xlm, xml), (ym, ym), color='0.3')  # top   horz
_ = ax_a.plot((xrm, xmr), (ym, ym), color='0.3')  # top   horz
'''
# second bracket (removed b/c only coef. is bias)
xll = bar_a.patches[0].get_x() + 0.5 * bar_a.patches[0].get_width()
xlr = bar_a.patches[1].get_x() + 0.5 * bar_a.patches[1].get_width()
xrl = xlm  # from prev bracket
xrr = xrm  # from prev bracket
xlm = (xll + xlr) / 2.
xrm = (xrl + xrr) / 2.
xm = (xlm + xrm) / 2.
adj = 0.5
yl = adj + max([bar_a.patches[num].get_height() for num in [0, 1]])
#yr = adj + max([bar_a.patches[num].get_height() for num in [2, 3, 4, 5]])
yr = adj + max([yl, yr])
ym = adj + max([yl, yr])
_ = ax_a.plot((xll, xlr), (yl, yl), color='0.3')  # left  horz
_ = ax_a.plot((xrl, xrr), (yr, yr), color='0.3')  # right horz
_ = ax_a.plot((xlm, xlm), (yl, ym), color='0.3')  # left  vert
_ = ax_a.plot((xrm, xrm), (yr, ym), color='0.3')  # right vert
txt = ax_a.annotate(form_pval[0], (xm, ym), **pval_kwargs)
txt.set_bbox(dict(facecolor='w', alpha=0, boxstyle='round, pad=0.25'))
plt.draw()
bb = txt.get_bbox_patch().get_window_extent()
txth = np.diff(ax_a.transData.inverted().transform(bb), axis=0).ravel()[-1]
txtw = np.diff(ax_a.transData.inverted().transform(bb), axis=0).ravel()[0]
xml = np.mean([xlm, xrm]) - txtw / 2.
xmr = np.mean([xlm, xrm]) + txtw / 2.
_ = ax_a.plot((xlm, xml), (ym, ym), color='0.3')  # top   horz
_ = ax_a.plot((xrm, xmr), (ym, ym), color='0.3')  # top   horz
'''

# add the d-prime chance & ceiling as grey dashed lines
_ = ax_a.axhspan(0, ceilchance_das[1], color=bgr, zorder=-1)
_ = ax_a.axhspan(ceilchance_das[0], sub_a.get_ybound()[1], color=bgr,
                 zorder=-1)
# finish plot & save
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.savefig(op.join(figdir, 'das-interactions' + outext))

#%% # # # # # # # # # # #
# divAttnSem foil rate  #
# # # # # # # # # # # # #
""" (pho vs sem) vs (sel vs (adj vs sep)) """
# reset things that have changed
plt.rcdefaults()
rcParams.update(rcp)
# INITIALIZE FIGURE
fig, sub_a = plt.subplots(1, 1, figsize=(3.4, 3.5))

# pho/sem X adj/sep foil rate stats
relevant_coefs = ('foil:semantic:separated')
corr_pval = find_min_pval(relevant_coefs, table=pvals_das)
form_pval = efa.format_pval(corr_pval, scheme=pv_scheme, latex=latex)
signif = [pv < 0.05 for pv in corr_pval]
# pho/sem X adj/sep foil rate plot
colseq = 2 * [lgr[0]] + 2 * [dgr[0]]
linseq = 2 * [lgr[1]] + 2 * [dgr[1]]
rkw = dict(color=colseq, linewidth=1, edgecolor=linseq)
#brackets = [([0, 1], [2, 3])]
#br_text = form_pval[np.where(signif)[0]]
ax_a, bar_a = efa.barplot(das_foil_pt[True], axis=0, err_bars='se',
                          groups=[[0, 1], [2, 3]],
                          #eq_group_widths=True,
                          #brackets=brackets,
                          #bracket_text=br_text,
                          #bracket_inline=True,
                          #bracket_group_lines=True,
                          bar_names=['sep.', 'adj.'] * 2,
                          group_names=['phonetic', 'semantic'],
                          bar_kwargs=rkw, err_kwargs=ekw, pval_kwargs=fkw,
                          ax=sub_a, ylim=fylim)
_ = ax_a.yaxis.set_label_text('foil response rate')
_ = ax_a.yaxis.set_ticks(np.arange(*(fylim + np.array([0, fstep])),
                                   step=fstep))
_ = ax_a.yaxis.set_tick_params('major', labelsize=12)
for b, h in zip(bar_a, ['', '//', '', '//']):
    b.set_hatch(h)
# finish plot & save
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.savefig(op.join(figdir, 'das-foils' + outext))

#%% # # # # # # # # # # #
# dasCogLoad data setup #
# # # # # # # # # # # # #
# gather the data we want to plot
dcl = adata[['subj', 'num', 'idn', 'div', 'adj', 'hit', 'miss', 'fal', 'crj',
             'fht', 'targ', 'foil']]
dcl = dcl.reset_index(drop=True)
# 3-word, 6-word
dcl_size = dcl.groupby(['subj', 'num']).aggregate(sum)
dcl_size['dprime'] = efa.dprime(dcl_size[['hit', 'miss', 'fal', 'crj']
                                         ].values.astype(int))
dcl_size_pt = pd.pivot_table(dcl_size.reset_index(), 'dprime', 'subj', 'num')
dcl_size_pt = dcl_size_pt[['thr', 'six']]
# categories same, diff
dcl_cong = dcl.groupby(['subj', 'idn']).aggregate(sum)
dcl_cong['dprime'] = efa.dprime(dcl_cong[['hit', 'miss', 'fal', 'crj']
                                         ].values.astype(int))
dcl_cong_pt = pd.pivot_table(dcl_cong.reset_index(), 'dprime', 'subj', 'idn')
dcl_cong_pt = dcl_cong_pt[['False', 'True']]
# selective, separated, adjacent
dcl_slajsp = dcl.groupby(['subj', 'div', 'adj']).aggregate(sum)
dcl_slajsp['dprime'] = efa.dprime(dcl_slajsp[['hit', 'miss', 'fal', 'crj']
                                             ].values.astype(int))
dcl_slajsp_pt = pd.pivot_table(dcl_slajsp.reset_index(),
                               'dprime', 'subj', ['div', 'adj'])
dcl_slajsp_pt = dcl_slajsp_pt.loc[slice(None), [(False, 'False'),
                                                (True, 'False'),
                                                (True, 'True')]]
# thr/six X sel/sep/adj
dcl_num_adj = dcl.groupby(['subj', 'num', 'div', 'adj']).aggregate(sum)
dcl_num_adj['dprime'] = efa.dprime(dcl_num_adj[['hit', 'miss', 'fal', 'crj']
                                               ].values.astype(int))
dcl_num_adj_pt = pd.pivot_table(dcl_num_adj.reset_index(),
                                'dprime', 'subj', ['div', 'num', 'adj'])
dcl_num_adj_pt = dcl_num_adj_pt.loc[slice(None), [(False, 'thr', 'False'),
                                                  (False, 'six', 'False'),
                                                  (True, 'thr', 'False'),
                                                  (True, 'thr', 'True'),
                                                  (True, 'six', 'False'),
                                                  (True, 'six', 'True')]]
# thr/six X same/diff
dcl_num_idn = dcl.groupby(['subj', 'num', 'idn']).aggregate(sum)
dcl_num_idn['dprime'] = efa.dprime(dcl_num_idn[['hit', 'miss', 'fal', 'crj']
                                               ].values.astype(int))
dcl_num_idn_pt = pd.pivot_table(dcl_num_idn.reset_index(),
                                'dprime', 'subj', ['num', 'idn'])
dcl_num_idn_pt = dcl_num_idn_pt.loc[slice(None), [('thr', 'False'),
                                                  ('thr', 'True'),
                                                  ('six', 'False'),
                                                  ('six', 'True')]]
# sel/sep/adj X same/diff
dcl_adj_idn = dcl.groupby(['subj', 'idn', 'div', 'adj']).aggregate(sum)
dcl_adj_idn['dprime'] = efa.dprime(dcl_adj_idn[['hit', 'miss', 'fal', 'crj']
                                               ].values.astype(int))
dcl_adj_idn_pt = pd.pivot_table(dcl_adj_idn.reset_index(),
                                'dprime', 'subj', ['div', 'idn', 'adj'])
dcl_adj_idn_pt = dcl_adj_idn_pt.loc[slice(None), [(False, 'False', 'False'),
                                                  (True, 'False', 'False'),
                                                  (True, 'False', 'True'),
                                                  (True, 'True', 'False'),
                                                  (True, 'True', 'True')]]
# pho/sem X sep/adj; foil rate
dcl_num_adj_f = dcl.groupby(['subj', 'num', 'div', 'adj']).aggregate(sum)
dcl_num_idn_f = dcl.groupby(['subj', 'num', 'idn']).aggregate(sum)
dcl_adj_idn_f = dcl.groupby(['subj', 'idn', 'div', 'adj']).aggregate(sum)
dcl_num_adj_f['frate'] = dcl_num_adj_f['fht'] / dcl_num_adj_f['foil'].astype(float)
dcl_num_idn_f['frate'] = dcl_num_idn_f['fht'] / dcl_num_idn_f['foil'].astype(float)
dcl_adj_idn_f['frate'] = dcl_adj_idn_f['fht'] / dcl_adj_idn_f['foil'].astype(float)
dcl_num_adj_f_pt = pd.pivot_table(dcl_num_adj_f.reset_index(),
                                  'frate', 'subj', ['div', 'num', 'adj'])
dcl_num_idn_f_pt = pd.pivot_table(dcl_num_idn_f.reset_index(),
                                  'frate', 'subj', ['num', 'idn'])
dcl_adj_idn_f_pt = pd.pivot_table(dcl_adj_idn_f.reset_index(),
                                  'frate', 'subj', ['div', 'idn', 'adj'])
dcl_num_adj_f_pt = dcl_num_adj_f_pt.loc[slice(None), [(False, 'thr', 'False'),
                                                      (False, 'six', 'False'),
                                                      (True, 'thr', 'False'),
                                                      (True, 'thr', 'True'),
                                                      (True, 'six', 'False'),
                                                      (True, 'six', 'True')]]
dcl_num_idn_f_pt = dcl_num_idn_f_pt.loc[slice(None), [('thr', 'False'),
                                                      ('thr', 'True'),
                                                      ('six', 'False'),
                                                      ('six', 'True')]]
dcl_adj_idn_f_pt = dcl_adj_idn_f_pt.loc[slice(None),
                                        [(False, 'False', 'False'),
                                         (True, 'False', 'False'),
                                         (True, 'False', 'True'),
                                         (True, 'True', 'False'),
                                         (True, 'True', 'True')]]

#%% # # # # # # # # # # # #
# dasCogLoad Main Effects #
# # # # # # # # # # # # # #
""" 3 vs 6  |  same vs diff  |  sel vs (adj vs sep) """
# reset things that have changed
plt.rcdefaults()
rcParams.update(rcp)
# INITIALIZE FIGURE
fig, (sub_a, sub_b, sub_c) = plt.subplots(1, 3, figsize=(7, 3.5))

# 3 vs 6 stats
relevant_coefs = ('thr', 'target:thr', 'foil:thr')
#pvals_dcl[np.in1d(pvals_dcl['predictor'], relevant_coefs)]
corr_pval = find_min_pval(relevant_coefs, table=pvals_dcl)
form_pval = efa.format_pval(corr_pval, scheme=pv_scheme, latex=latex)
signif = [pv < 0.05 for pv in corr_pval]
# 3 vs 6 plot
colseq = [lgr[0]] + [dgr[0]]
linseq = [lgr[1]] + [dgr[1]]
rkw = dict(color=colseq, linewidth=1, edgecolor=linseq)
ax_a, bar_a = efa.barplot(dcl_size_pt, axis=0, err_bars='se',
                          groups=[(0, 1)],
                          brackets=np.array([(0, 1)])[np.where(signif)],
                          bracket_text=form_pval[np.where(signif)],
                          bracket_inline=True,
                          bar_names=['three', 'six'],
                          group_names=['category size'],
                          bar_kwargs=rkw, err_kwargs=ekw, pval_kwargs=fkw,
                          ax=sub_a, ylim=ylim)
_ = ax_a.yaxis.set_label_text('d-prime')
_ = ax_a.yaxis.set_tick_params('major', labelsize=12)
# add the d-prime chance & ceiling as grey dashed lines
_ = ax_a.axhspan(0, ceilchance_dcl[1], color=bgr, zorder=-1)
_ = ax_a.axhspan(ceilchance_dcl[0], sub_a.get_ybound()[1], color=bgr,
                 zorder=-1)
# same vs diff stats
relevant_coefs = ('idn', 'target:idn', 'foil:idn')
corr_pval = find_min_pval(relevant_coefs, table=pvals_dcl)
form_pval = efa.format_pval(corr_pval, scheme=pv_scheme, latex=latex)
signif = [pv < 0.05 for pv in corr_pval]
# same vs diff plot
rkw = dict(color=wht[0], linewidth=1, edgecolor=wht[1])
ax_b, bar_b = efa.barplot(dcl_cong_pt, axis=0, err_bars='se',
                          groups=[(0, 1)],
                          brackets=np.array([(0, 1)])[np.where(signif)],
                          bracket_text=form_pval[np.where(signif)],
                          bracket_inline=True,
                          bar_names=['incongr.', 'congr.'],
                          group_names=['attn. catg. congruence'],
                          bar_kwargs=rkw, err_kwargs=ekw, pval_kwargs=fkw,
                          ax=sub_b, ylim=ylim)
_ = ax_b.yaxis.set_ticklabels([])
for b, h in zip(bar_b, ['', '--']):
    b.set_hatch(h)
# add the d-prime chance & ceiling as grey dashed lines
_ = ax_b.axhspan(0, ceilchance_dcl[1], color=bgr, zorder=-1)
_ = ax_b.axhspan(ceilchance_dcl[0], sub_b.get_ybound()[1], color=bgr,
                 zorder=-1)
'''
_ = ax_b.hlines(ceilchance_dcl, colors='#444444', linestyles='--',
                xmin=sub_a.get_xbound()[0],
                xmax=sub_a.get_xbound()[1])
'''

# adj vs sep stats
relevant_coefs = ('adj', 'target:adj', 'foil:adj')
corr_pval = find_min_pval(relevant_coefs, table=pvals_dcl)
form_pval = efa.format_pval(corr_pval, scheme=pv_scheme, latex=latex)
signif = [pv < 0.05 for pv in corr_pval]
# sel vs div, adj vs sep plot
rkw = dict(color=wht[0], linewidth=1, edgecolor=wht[1])
#brackets = [(0, 1)][np.where(signif)[0]]
#br_text = form_pval[np.where(signif)[0]]
ax_c, bar_c = efa.barplot(dcl_slajsp_pt[True], axis=0, err_bars='se',
                          groups=[(0, 1)],
                          eq_group_widths=True,
                          brackets=np.array([(0, 1)])[np.where(signif)],
                          bracket_text=form_pval[np.where(signif)],
                          bracket_inline=True,
                          bar_names=['sep.', 'adj.'],
                          group_names=['attn. stream adjacency'],
                          bar_kwargs=rkw, err_kwargs=ekw, pval_kwargs=fkw,
                          ax=sub_c, ylim=ylim)
_ = ax_c.yaxis.set_ticklabels([])
# add the d-prime chance & ceiling as grey dashed lines
_ = ax_c.axhspan(0, ceilchance_dcl[1], color=bgr, zorder=-1)
_ = ax_c.axhspan(ceilchance_dcl[0], sub_c.get_ybound()[1], color=bgr,
                 zorder=-1)
'''
_ = ax_c.hlines(ceilchance_dcl, colors='#444444', linestyles='--',
                xmin=sub_c.get_xbound()[0],
                xmax=sub_c.get_xbound()[1])
'''
for b, h in zip(bar_c, ['', '//']):
    b.set_hatch(h)
# finish plot & save
plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
plt.subplots_adjust(bottom=0.2)
_ = ax_a.text(-0.5, ylim[-1], 'a)')
_ = ax_b.text(-0.35, ylim[-1], 'b)')
_ = ax_c.text(-0.2, ylim[-1], 'c)')
plt.savefig(op.join(figdir, 'dcl-main-effects' + outext))

#%% # # # # # # # # # # # #
# dasCogLoad Interactions #
# # # # # # # # # # # # # #
""" (pho vs sem) vs (sel vs (adj vs sep)) """
# reset things that have changed
plt.rcdefaults()
rcParams.update(rcp)
# INITIALIZE FIGURE
fig, (sub_a, sub_b, sub_c) = plt.subplots(1, 3, figsize=(7, 3.5))

# 3/6 X same/diff stats
relevant_coefs = ('idn:thr', 'target:idn:thr', 'foil:idn:thr')
#pvals_dcl[np.in1d(pvals_dcl['predictor'], relevant_coefs)]
corr_pval = find_min_pval(relevant_coefs, table=pvals_dcl)
form_pval = efa.format_pval(corr_pval, scheme=pv_scheme, latex=latex)
signif = [pv < 0.05 for pv in corr_pval]
# 3/6 X same/diff plot
colseq = 2 * [lgr[0]] + 2 * [dgr[0]]
linseq = 2 * [lgr[1]] + 2 * [dgr[1]]
rkw = dict(color=colseq, linewidth=1, edgecolor=linseq)
ax_a, bar_a = efa.barplot(dcl_num_idn_pt, axis=0, err_bars='se',
                          groups=[[0, 1], [2, 3]],
                          eq_group_widths=True,
                          #brackets=[brackets],
                          #bracket_text=br_text,
                          #bracket_inline=True,
                          #bracket_group_lines=True,
                          bar_names=['inc.', 'cgr.', 'inc.', 'cgr.'],
                          group_names=['three', 'six'],
                          bar_kwargs=rkw, err_kwargs=ekw, pval_kwargs=fkw,
                          ax=sub_a, ylim=ylim)
_ = ax_a.yaxis.set_label_text('d-prime')
_ = ax_a.yaxis.set_tick_params('major', labelsize=12)
for b, h in zip(bar_a, ['', '--'] * 2):
    b.set_hatch(h)
# add the d-prime chance & ceiling as grey dashed lines
_ = ax_a.axhspan(0, ceilchance_dcl[1], color=bgr, zorder=-1)
_ = ax_a.axhspan(ceilchance_dcl[0], sub_a.get_ybound()[1], color=bgr,
                 zorder=-1)
'''
_ = ax_a.hlines(ceilchance_dcl, colors='#444444', linestyles='--',
                xmin=sub_a.get_xbound()[0],
                xmax=sub_a.get_xbound()[1])
'''

# 3/6 X adj/sep stats
relevant_coefs = ('adj:thr', 'target:adj:thr', 'foil:adj:thr')
corr_pval = find_min_pval(relevant_coefs, table=pvals_dcl)
form_pval = efa.format_pval(corr_pval, scheme=pv_scheme, latex=latex)
signif = [pv < 0.05 for pv in corr_pval]
# 3/6 X adj/sep plot
colseq = 2 * [lgr[0]] + 2 * [dgr[0]]
linseq = 2 * [lgr[1]] + 2 * [dgr[1]]
rkw = dict(color=colseq, linewidth=1, edgecolor=linseq)
ax_b, bar_b = efa.barplot(dcl_num_adj_pt[True], axis=0, err_bars='se',
                          groups=[[0, 1], [2, 3]],
                          eq_group_widths=True,
                          #brackets=[brackets],
                          #bracket_text=br_text,
                          #bracket_inline=True,
                          #bracket_group_lines=True,
                          bar_names=['sep.', 'adj.', 'sep.', 'adj.'],
                          group_names=['three', 'six'],
                          bar_kwargs=rkw, err_kwargs=ekw, pval_kwargs=fkw,
                          ax=sub_b, ylim=ylim)
_ = ax_b.yaxis.set_ticklabels([])
for b, h in zip(bar_b, ['', '//'] * 2):
    b.set_hatch(h)
# add the d-prime chance & ceiling as grey dashed lines
_ = ax_b.axhspan(0, ceilchance_dcl[1], color=bgr, zorder=-1)
_ = ax_b.axhspan(ceilchance_dcl[0], sub_b.get_ybound()[1], color=bgr,
                 zorder=-1)
'''
_ = ax_b.hlines(ceilchance_dcl, colors='#444444', linestyles='--',
                xmin=sub_b.get_xbound()[0],
                xmax=sub_b.get_xbound()[1])
'''

# same/diff X adj/sep stats
relevant_coefs = ('adj:idn', 'target:adj:idn', 'foil:adj:idn')
corr_pval = find_min_pval(relevant_coefs, table=pvals_dcl)
form_pval = efa.format_pval(corr_pval, scheme=pv_scheme, latex=latex)
signif = [pv < 0.05 for pv in corr_pval]
# same/diff X adj/sep plot
rkw = dict(color=wht[0], linewidth=1, edgecolor=wht[1])
ax_c, bar_c = efa.barplot(dcl_adj_idn_pt[True], axis=0, err_bars='se',
                          groups=[[0, 1], [2, 3]],
                          eq_group_widths=True,
                          brackets=[([0, 1], [2, 3])],
                          bracket_text=form_pval[np.where(signif)],
                          bracket_inline=True,
                          bracket_group_lines=True,
                          bar_names=['sep.', 'adj.', 'sep.', 'adj.'],
                          group_names=['incongr.', 'congr.'],
                          bar_kwargs=rkw, err_kwargs=ekw, pval_kwargs=fkw,
                          ax=sub_c, ylim=ylim)
_ = ax_c.yaxis.set_ticklabels([])
for b, h in zip(bar_c, ['', '//', '--', '//--']):
    b.set_hatch(h)
# add the d-prime chance & ceiling as grey dashed lines
_ = ax_c.axhspan(0, ceilchance_dcl[1], color=bgr, zorder=-1)
_ = ax_c.axhspan(ceilchance_dcl[0], sub_c.get_ybound()[1], color=bgr,
                 zorder=-1)
'''
_ = ax_c.hlines(ceilchance_dcl, colors='k', linestyles=':',
                xmin=sub_c.get_xbound()[0],
                xmax=sub_c.get_xbound()[1], zorder=-1)
'''

# finish plot & save
plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
plt.subplots_adjust(bottom=0.2)
_ = ax_a.text(-0.55, ylim[-1], 'a)')
_ = ax_b.text(-0.4, ylim[-1], 'b)')
_ = ax_c.text(-0.4, ylim[-1], 'c)')
plt.savefig(op.join(figdir, 'dcl-interactions' + outext))

#%% # # # # # # # # # # #
# dasCogLoad foil rate  #
# # # # # # # # # # # # #
# reset things that have changed
plt.rcdefaults()
rcParams.update(rcp)
# INITIALIZE FIGURE
fig, (sub_a, sub_b, sub_c) = plt.subplots(1, 3, figsize=(7, 3.5))

# 3/6 X same/diff stats
relevant_coefs = ('foil:idn:thr')
#pvals_dcl[np.in1d(pvals_dcl['predictor'], relevant_coefs)]
corr_pval = find_min_pval(relevant_coefs, table=pvals_dcl)
form_pval = efa.format_pval(corr_pval, scheme=pv_scheme, latex=latex)
signif = [pv < 0.05 for pv in corr_pval]
# 3/6 X same/diff plot
colseq = 2 * [lgr[0]] + 2 * [dgr[0]]
linseq = 2 * [lgr[1]] + 2 * [dgr[1]]
rkw = dict(color=colseq, linewidth=1, edgecolor=linseq)
ax_a, bar_a = efa.barplot(dcl_num_idn_f_pt, axis=0, err_bars='se',
                          groups=[[0, 1], [2, 3]],
                          eq_group_widths=True,
                          #brackets=[brackets],
                          #bracket_text=br_text,
                          #bracket_inline=True,
                          #bracket_group_lines=True,
                          bar_names=['inc.', 'cgr.', 'inc.', 'cgr.'],
                          group_names=['three', 'six'],
                          bar_kwargs=rkw, err_kwargs=ekw, pval_kwargs=fkw,
                          ax=sub_a, ylim=fylim)
_ = ax_a.yaxis.set_label_text('foil response rate')
_ = ax_a.yaxis.set_ticks(np.arange(*(fylim + np.array([0, fstep])),
                                   step=fstep))
_ = ax_a.yaxis.set_tick_params('major', labelsize=12)
for b, h in zip(bar_a, ['', '--'] * 2):
    b.set_hatch(h)

# 3/6 X adj/sep stats
relevant_coefs = ('foil:adj:thr')
corr_pval = find_min_pval(relevant_coefs, table=pvals_dcl)
form_pval = efa.format_pval(corr_pval, scheme=pv_scheme, latex=latex)
signif = [pv < 0.05 for pv in corr_pval]
# 3/6 X adj/sep plot
colseq = 2 * [lgr[0]] + 2 * [dgr[0]]
linseq = 2 * [lgr[1]] + 2 * [dgr[1]]
rkw = dict(color=colseq, linewidth=1, edgecolor=linseq)
ax_b, bar_b = efa.barplot(dcl_num_adj_f_pt[True], axis=0, err_bars='se',
                          groups=[[0, 1], [2, 3]],
                          eq_group_widths=True,
                          #brackets=[([0, 1], [2, 3])],
                          #bracket_text=form_pval[np.where(signif)],
                          #bracket_inline=True,
                          #bracket_group_lines=True,
                          bar_names=['sep.', 'adj.', 'sep.', 'adj.'],
                          group_names=['three', 'six'],
                          bar_kwargs=rkw, err_kwargs=ekw, pval_kwargs=fkw,
                          ax=sub_b, ylim=fylim)
_ = ax_b.yaxis.set_ticks(np.arange(*(fylim + np.array([0, fstep])),
                                   step=fstep))
_ = ax_b.yaxis.set_ticklabels([])
for b, h in zip(bar_b, ['', '//'] * 2):
    b.set_hatch(h)

# same/diff X adj/sep stats
relevant_coefs = ('foil:adj:idn')
corr_pval = find_min_pval(relevant_coefs, table=pvals_dcl)
form_pval = efa.format_pval(corr_pval, scheme=pv_scheme, latex=latex)
signif = [pv < 0.05 for pv in corr_pval]
# same/diff X adj/sep plot
rkw = dict(color=wht[0], linewidth=1, edgecolor=wht[1])
ax_c, bar_c = efa.barplot(dcl_adj_idn_f_pt[True], axis=0, err_bars='se',
                          groups=[[0, 1], [2, 3]],
                          eq_group_widths=True,
                          brackets=[([0, 1], [2, 3])],
                          bracket_text=form_pval[np.where(signif)],
                          bracket_inline=True,
                          bracket_group_lines=True,
                          bar_names=['sep.', 'adj.', 'sep.', 'adj.'],
                          group_names=['incongr.', 'congr.'],
                          bar_kwargs=rkw, err_kwargs=ekw, pval_kwargs=fkw,
                          ax=sub_c, ylim=fylim)
_ = ax_c.yaxis.set_ticks(np.arange(*(fylim + np.array([0, fstep])),
                                   step=fstep))
_ = ax_c.yaxis.set_ticklabels([])
for b, h in zip(bar_c, ['', '//', '--', '//--']):
    b.set_hatch(h)

# finish plot & save
plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
plt.subplots_adjust(bottom=0.2)
_ = ax_a.text(-0.7, fylim[-1], 'a)')
_ = ax_b.text(-0.4, fylim[-1], 'b)')
_ = ax_c.text(-0.4, fylim[-1], 'c)')
plt.savefig(op.join(figdir, 'dcl-foils' + outext))

#%% # # # # # # # # # # # # #
# Reaction Time Data Setup  #
# # # # # # # # # # # # # # #
# divAttnSem reaction time data
dasrt = bdata[['subj', 'sem', 'div', 'adj', 'rt', 'rt_hit', 'rt_fht']]
dasrt['rt'] = dasrt['rt'].apply(parse_strings)
dasrt['rt_hit'] = dasrt['rt_hit'].apply(parse_strings)
dasrt['rt_fht'] = dasrt['rt_fht'].apply(parse_strings)
dasrt = dasrt.reset_index(drop=True)

agg_dict = dict(rt=flat, rt_hit=flat, rt_fht=flat)
dasrt_agg = dasrt.groupby(['subj', 'div', 'sem']).aggregate(agg_dict)
dasrt_agg['rtch'] = dasrt_agg['rt'].apply(chsq_rt)
dasrt_agg['rtch_hit'] = dasrt_agg['rt_hit'].apply(chsq_rt)
dasrt_agg['rtch_fht'] = dasrt_agg['rt_fht'].apply(chsq_rt)
dasrt_agg_pt = pd.pivot_table(dasrt_agg.reset_index(),
                              ['rtch', 'rtch_hit', 'rtch_fht'], 'subj',
                              ['div', 'sem'])
# dasCogLoad reaction time data
dclrt = adata[['subj', 'div', 'adj', 'idn', 'num', 'rt', 'rt_hit', 'rt_fht']]
dclrt = dclrt.set_index('idn', drop=False).loc[['True', 'False']]
dclrt = dclrt.set_index('adj', drop=False).loc[['True', 'False']]
dclrt = dclrt.set_index('num', drop=False).loc[['thr', 'six']]
dclrt['rt'] = dclrt['rt'].apply(literal_eval)
dclrt['rt_hit'] = dclrt['rt_hit'].apply(literal_eval)
dclrt['rt_fht'] = dclrt['rt_fht'].apply(literal_eval)
dclrt = dclrt.reset_index(drop=True)

agg_dict = dict(rt=flat, rt_hit=flat, rt_fht=flat)
dclrt_agg = dclrt.groupby(['subj', 'div', 'num']).aggregate(agg_dict)
dclrt_agg['rtch'] = dclrt_agg['rt'].apply(chsq_rt)
dclrt_agg['rtch_hit'] = dclrt_agg['rt_hit'].apply(chsq_rt)
dclrt_agg['rtch_fht'] = dclrt_agg['rt_fht'].apply(chsq_rt)
dclrt_agg_pt = pd.pivot_table(dclrt_agg.reset_index(),
                              ['rtch', 'rtch_hit', 'rtch_fht'], 'subj',
                              ['div', 'num'])
# subset & reorder columns for plotting (get 3 before 6)
dclrt_agg_pt_plot = dclrt_agg_pt['rtch_hit']
dclrt_agg_pt_plot = dclrt_agg_pt_plot.loc[slice(None), [(False, 'thr'),
                                                        (False, 'six'),
                                                        (True, 'thr'),
                                                        (True, 'six')]]

#%% # # # # # # # # # # #
# Reaction Time Barplot #
# # # # # # # # # # # # #
# reset things that have changed
plt.rcdefaults()
rcParams.update(rcp)
# INITIALIZE FIGURE
fig, (sub_a, sub_b) = plt.subplots(1, 2, figsize=(7, 3.5))

# divAttnSem pho/sem X sel/div RT stats
cmpr_a = pd.concat((dasrt_agg_pt['rtch_hit', False, False],
                    dasrt_agg_pt['rtch_hit', True, True],
                    dasrt_agg_pt['rtch_hit', True, False],
                    dasrt_agg_pt['rtch_hit', False, True]), axis=1)
cmpr_b = pd.concat((dasrt_agg_pt['rtch_hit', False, True],
                    dasrt_agg_pt['rtch_hit', True, False],
                    dasrt_agg_pt['rtch_hit', False, False],
                    dasrt_agg_pt['rtch_hit', True, True]), axis=1)
tval, pval = ss.ttest_rel(cmpr_a, cmpr_b, axis=0)
corr_pval = bonferroni(pval)  # holm(pval)
form_pval = efa.format_pval(corr_pval, scheme=pv_scheme, latex=latex)
signif = corr_pval < 0.05

# divAttnSem pho/sem X sel/div RT plot
colseq = 2 * ([lgr[0]] + [dgr[0]])
linseq = 2 * ([lgr[1]] + [dgr[1]])
rkw = dict(color=colseq, linewidth=1, edgecolor=linseq)
ax_a, bar_a = efa.barplot(dasrt_agg_pt['rtch_hit'], axis=0, err_bars='se',
                          groups=[(0, 1), (2, 3)],
                          brackets=fourway[signif],
                          bracket_text=form_pval[signif],
                          bracket_inline=True,
                          bar_names=['phon.', 'sem.'] * 2,
                          group_names=['selective', 'divided'],
                          bar_kwargs=rkw, err_kwargs=ekw, pval_kwargs=fkw,
                          ax=sub_a, ylim=rtylim)
_ = ax_a.yaxis.set_label_text('reaction time (s)')
_ = ax_a.yaxis.set_ticks(np.arange(*(rtylim + np.array([0, rtstep])),
                                   step=rtstep))
_ = ax_a.yaxis.set_tick_params('major', labelsize=12)
for b, h in zip(bar_a, ['\\', '\\', '', '']):
    b.set_hatch(h)

# dasCogLoad three/six X sel/div RT stats
cmpr_a = pd.concat((dclrt_agg_pt_plot[False, 'six'],
                    dclrt_agg_pt_plot[True, 'thr'],
                    dclrt_agg_pt_plot[False, 'thr'],
                    dclrt_agg_pt_plot[True, 'six']), axis=1)
cmpr_b = pd.concat((dclrt_agg_pt_plot[False, 'thr'],
                    dclrt_agg_pt_plot[True, 'six'],
                    dclrt_agg_pt_plot[True, 'thr'],
                    dclrt_agg_pt_plot[False, 'six']), axis=1)
tval, pval = ss.ttest_rel(cmpr_a, cmpr_b, axis=0)
corr_pval = bonferroni(pval)  # holm(pval)
form_pval = efa.format_pval(corr_pval, scheme=pv_scheme, latex=latex)
signif = corr_pval < 0.05

# dasCogLoad three/six X sel/div RT plot
ax_b, bar_b = efa.barplot(dclrt_agg_pt_plot, axis=0, err_bars='se',
                          groups=[(0, 1), (2, 3)],
                          brackets=fourway[signif],
                          bracket_text=form_pval[signif],
                          bracket_inline=True,
                          bar_names=['three', 'six'] * 2,
                          group_names=['selective', 'divided'],
                          bar_kwargs=rkw, err_kwargs=ekw, pval_kwargs=fkw,
                          ax=sub_b, ylim=rtylim)
_ = ax_b.yaxis.set_ticklabels([])
_ = ax_b.yaxis.set_ticks(np.arange(*(rtylim + np.array([0, rtstep])),
                                   step=rtstep))
for b, h in zip(bar_b, ['\\', '\\', '', '']):
    b.set_hatch(h)

plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
plt.subplots_adjust(bottom=0.2)
_ = ax_a.text(-1.1, rtylim[-1], 'a)')
_ = ax_b.text(-0.5, rtylim[-1], 'b)')
plt.savefig(op.join(figdir, 'rt-barplot' + outext))

"""
# P-VALUES FOR REACTION TIMES BETWEEN EXPERIMENTS 1 AND 2
# (only for the subjects that were in both experiments)
combined_rts = pd.concat((dasrt_agg_pt['rtch_hit'],
                          dclrt_agg_pt_plot), axis=1).dropna()
tval, pval = ss.ttest_rel(combined_rts.ix[:, :4], combined_rts.ix[:, 4:])
"""

#%% # # # # # # # # # # # #
# REACTION TIME HISTOGRAM #
# # # # # # # # # # # # # #
bins = np.arange(0.25, 1.25, 0.025)
lsp = np.linspace(0.25, 1.25, 100)


def plot_hist_chsq(x, bins, fig, color, linsp, lcolor=None, histalpha=0.25,
                   label='', ltyp=(None, None), **kwargs):
    histcol = mplc.hex2color(color) + (histalpha,)
    if lcolor is None:
        lcolor = color
    plt.hist(x, bins, normed=True, color=histcol, histtype='stepfilled',
             **kwargs)
    df, loc, scale = ss.chi2.fit(x, floc=0)
    pdf = ss.chi2.pdf(linsp, df, scale=scale)
    l = plt.plot(linsp, pdf, color=lcolor, figure=fig, linewidth=2,
                 label=label)
    l[0].set_dashes(ltyp)
    l[0].set_dash_capstyle('round')


# gather the reaction time values we want
dasrt_agg_ri = dasrt_agg.reset_index()
dclrt_agg_ri = dclrt_agg.reset_index()
phonetic_rts = dasrt_agg_ri.loc[dasrt_agg_ri['sem'] == False, 'rt_hit']  # analysis:ignore
thr_word_rts = dclrt_agg_ri.loc[dclrt_agg_ri['num'] == 'thr', 'rt_hit']
six_word_rts = dclrt_agg_ri.loc[dclrt_agg_ri['num'] == 'six', 'rt_hit']
phonetic_rts = list(ch.from_iterable(phonetic_rts))
thr_word_rts = list(ch.from_iterable(thr_word_rts))
six_word_rts = list(ch.from_iterable(six_word_rts))

# plot
fig = plt.figure(figsize=(3.4, 3))
plot_hist_chsq(phonetic_rts, bins, fig, color=grn[1], linsp=lsp,
               edgecolor='none', label='phonetic')
plot_hist_chsq(thr_word_rts, bins, fig, color=red[1], linsp=lsp,
               edgecolor='none', label='3-word', ltyp=[8, 4])
plot_hist_chsq(six_word_rts, bins, fig, color=blu[1], linsp=lsp,
               edgecolor='none', label='6-word', ltyp=[2, 4])

ax = fig.gca()
_ = ax.set_xlim(0.25, 1.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params('both', top=False, left=False, right=False, direction='out')
_ = ax.yaxis.set_ticklabels([])
_ = ax.xaxis.set_ticks(np.arange(0.25, 1.5, 0.25))
_ = ax.set_xlabel('Response time (s)')
_ = ax.set_ylabel('Proportion of responses')
lgnd = ax.legend(loc='upper right', frameon=False, fontsize='small',
                 bbox_to_anchor=(1.15, 1.1), handlelength=2.3)
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.savefig(op.join(figdir, 'rt-hist' + outext), bbox_extra_artists=(lgnd,))

#%% # # # # # # # # # # # # # #
# ENERGETIC MASKING HISTOGRAM #
# # # # # # # # # # # # # # # #
vardict = np.load(op.join('processedData', 'masking.npz'))
db_per_ear = vardict['db_per_ear']
word_max = np.max(db_per_ear, axis=-1)
word_max.shape = word_max.shape + (1,)
better_ear = db_per_ear == np.tile(word_max, 2)
db_better_ear = db_per_ear[better_ear]
db_min = np.min(db_better_ear)
db_max = np.max(db_better_ear)
x_range = (np.floor(db_min), np.ceil(db_max))
bins = np.arange(*x_range)
lsp = np.linspace(db_min, db_max, 100)

fig = plt.figure(figsize=(3.4, 3))
_ = plt.hist(db_better_ear, bins, color=lgr[0], histtype='stepfilled',
             edgecolor='none', normed=True)
loc, scale = ss.norm.fit(db_better_ear)
q = np.percentile(db_better_ear, (25, 75))
pdf = ss.norm.pdf(lsp, loc=loc, scale=scale)
peak = ss.norm.pdf(loc, loc=loc, scale=scale)
l = plt.plot(lsp, pdf, color='k', figure=fig, linewidth=1, label='')

ax = fig.gca()
_ = ax.set_xlim(*x_range)
_ = ax.annotate('mean = {0:+.2f} dB'.format(loc), xy=(loc, peak),
                xycoords='data', xytext=(loc, 0.12), textcoords='data',
                horizontalalignment='center', verticalalignment='bottom',
                arrowprops=dict(width=1, headwidth=4, facecolor='0.3',
                                edgecolor='0.3', shrink=0.1))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params('both', top=False, right=False, direction='out')
# convert to percentage, add + signs to positive dB values
plt.draw()
tcy = ax.yaxis.get_ticklabels()
tcx = ax.xaxis.get_ticklabels()
tcp = ['{}%'.format(int(100 * float(tc.get_text()))) for tc in tcy]
tcd = ['{0:+}'.format(int(tc.get_text().replace('−', '-')))
       if tc.get_text() != '' else '' for tc in tcx]
_ = ax.yaxis.set_ticklabels(tcp)
_ = ax.xaxis.set_ticklabels(tcd)
# annotation tweaks
_ = ax.yaxis.set_tick_params('major', labelsize=12)
_ = ax.xaxis.set_tick_params('major', labelsize=12)
_ = ax.set_xlabel('Signal-to-masker ratio (dB)')
_ = ax.set_ylabel('Attended words')
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.savefig(op.join(figdir, 'masking-hist' + outext))

#%% # # # # #
# CLEAN UP  #
# # # # # # #
plt.close('all')

#%% # # # # # # # # # # # # #
# dasCogLoad foil vs. stray #
# # # # # # # # # # # # # # #
dcl_stray_thr = dcl.groupby(['div', 'num']).aggregate(sum)
dcl_stray_thr['sty'] = dcl_stray_thr['fal'] - dcl_stray_thr['fht']
dcl_stray_thr[['fht', 'sty']].loc[True].loc[['thr', 'six']]

dcl_stray_adj_idn = dcl.groupby(['div', 'adj', 'idn']).aggregate(sum)
dcl_stray_adj_idn['sty'] = dcl_stray_adj_idn['fal'] - dcl_stray_adj_idn['fht']
dcl_stray_adj_idn[['fht', 'sty']].loc[True].loc[['False', 'True']]

#%% # # # # # # # # # # # #
# Min and Max performance #
# # # # # # # # # # # # # #
das_pts = [das_phosem_pt, das_slajsp_pt, das_inter_pt]
dcl_pts = [dcl_size_pt, dcl_cong_pt, dcl_slajsp_pt, dcl_num_adj_pt,
           dcl_num_idn_pt, dcl_adj_idn_pt]
das_min = min([xx.min().min() for xx in das_pts])
das_max = max([xx.max().max() for xx in das_pts])
dcl_min = min([xx.min().min() for xx in dcl_pts])
dcl_max = max([xx.max().max() for xx in dcl_pts])

#%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ANALYSIS OF SUCCESSFUL TARGET DETECTION TO MULTIPLE STREAMS #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
foo = bdata[['subj', 'sem', 'div', 'adj', 'attn', 'rawrt', 'targ_loc']]
foo['targ_loc'] = foo['targ_loc'].apply(parse_loc)
foo['presses'] = foo['rawrt'].apply(parse_rawrt, return_bool=True)
foo['rawrt'] = foo['rawrt'].apply(parse_rawrt)
foo['attn'] = foo['attn'].apply(parse_strings)
targ_shape = (foo.shape[0],) + foo['targ_loc'][0].shape
presses = np.array([xx for xx in foo['presses'].values])
targloc = np.array([xx for xx in foo['targ_loc'].values])
bar = np.logical_and(presses, targloc)
foo['hit_loc'] = [xx for xx in bar]
foo['targ_per_stream'] = foo['targ_loc'].apply(sum_bools, usenan=True)
foo['hits_tmp'] = foo['hit_loc'].apply(sum_bools)
foo['miss_per_stream'] = foo['targ_per_stream'] - foo['hits_tmp']
foo['hit_per_stream'] = foo['targ_per_stream'] - foo['miss_per_stream']
foo['hit_in_stream'] = foo['hit_per_stream'].apply(lambda x: x > 0)
foo['targ_in_stream'] = foo['targ_per_stream'].apply(lambda x: x > 0)
foo['hit_multistream'] = foo['hit_in_stream'].apply(lambda x: sum(x) > 1)
foo['targ_multistream'] = foo['targ_in_stream'].apply(lambda x: sum(x) > 1)
# mh = multihit, or at least 1 hit in each attended stream
mh_div = foo[foo['div']][['hit_multistream', 'targ_multistream']].sum().values
mh_adj = foo[(foo['div'] & foo['adj'])][['hit_multistream', 'targ_multistream']].sum().values
mh_sep = foo[(foo['div'] & ~foo['adj'])][['hit_multistream', 'targ_multistream']].sum().values
mh_pho = foo[(foo['div'] & ~foo['sem'])][['hit_multistream', 'targ_multistream']].sum().values
mh_pha = foo[(foo['div'] & ~foo['sem'] & foo['adj'])][['hit_multistream', 'targ_multistream']].sum().values
mh_phs = foo[(foo['div'] & ~foo['sem'] & ~foo['adj'])][['hit_multistream', 'targ_multistream']].sum().values
mh_sem = foo[(foo['div'] & foo['sem'])][['hit_multistream', 'targ_multistream']].sum().values
mh_sea = foo[(foo['div'] & foo['sem'] & foo['adj'])][['hit_multistream', 'targ_multistream']].sum().values
mh_ses = foo[(foo['div'] & foo['sem'] & ~foo['adj'])][['hit_multistream', 'targ_multistream']].sum().values


#%%  sand left sand left
#      three      six
'''
column_order = ['subj', 'div', 'adj', 'idn', 'num', 'sandwich', 'leftover',
                'foil', 'snd', 'lft', 'fht']  # 'cond'
figeight = adata[column_order]
figeight = figeight.set_index('div', drop=False).loc[True]  # no sel. trials
figeight = figeight.set_index('idn', drop=False).loc[['True', 'False']]
figeight = figeight.set_index('adj', drop=False).loc['False']  # no adj. trials
figeight = figeight.set_index('num', drop=False).loc[['thr', 'six']]  # no both
figeight['num'][figeight['num'] == 'six'] = 6
figeight['num'][figeight['num'] == 'thr'] = 3
figeight['num'] = figeight['num'].astype(int)
figeight['idn'] = np.in1d(figeight['idn'], np.array('True'))  # convert to bool
figeight['adj'] = np.in1d(figeight['adj'], np.array('True'))  # convert to bool
figeight = figeight.reset_index(drop=True)

# INITIALIZE FIGURE
ylim = (0., 0.3)
step = 0.1
fig, sub_8a = plt.subplots(1, 1, figsize=(3.4, 3.5))
#% 8a: 3/6, sand/left
agg_dict = dict(sandwich=sum, leftover=sum, foil=sum, snd=sum, lft=sum,
                fht=sum)
eight_a = figeight.groupby(['subj', 'num']).aggregate(agg_dict)
max_ev_sand = eight_a['sandwich'].max()
max_ev_left = eight_a['leftover'].max()
eight_a['shr'] = eight_a['snd'] / eight_a['sandwich'].astype(float)
eight_a['lhr'] = eight_a['lft'] / eight_a['leftover'].astype(float)
eight_s = pd.pivot_table(eight_a.reset_index(), 'shr', 'subj', 'num')
eight_s['sandwich'] = True
eight_s.set_index('sandwich', append=True, inplace=True)
eight_l = pd.pivot_table(eight_a.reset_index(), 'lhr', 'subj', 'num')
eight_l['sandwich'] = False
eight_l.set_index('sandwich', append=True, inplace=True)
# reorder columns
eight_a = pd.concat((eight_s, eight_l), axis=0)
eight_a = eight_a.unstack('sandwich')
# stats
cmpr_a = efa.logit(pd.concat((eight_a[3, False], eight_a[6, True],
                              eight_a[6, False], eight_a[3, True]
                              ), axis=1), max_events)
cmpr_b = efa.logit(pd.concat((eight_a[3, True], eight_a[6, False],
                              eight_a[3, False], eight_a[6, True]
                              ), axis=1), max_events)
tval, pval = ss.ttest_rel(cmpr_a, cmpr_b, axis=0)
corr_pval = bonferroni(pval)  # holm(pval)
form_pval = efa.format_pval(corr_pval, scheme=pv_scheme, latex=latex)
signif = corr_pval < 0.05
# figure
rkw = dict(color=grn, linewidth=1)  # , edgecolor=(0.25, 0.25, 0.25))
ax_8a, bar_8a = efa.barplot(eight_a, axis=0, err_bars='se',
                            groups=[(0, 1), (2, 3)],
                            brackets=fourway[signif],
                            bracket_text=form_pval[signif],
                            bracket_inline=True,
                            bar_names=['left.', 'sand.'] * 2,
                            group_names=['three', 'six'],
                            bar_kwargs=rkw, err_kwargs=ekw, pval_kwargs=fkw,
                            ax=sub_8a, ylim=ylim)

_ = ax_8a.yaxis.set_label_text('foil response rate')
_ = ax_8a.yaxis.set_ticks(np.arange(*(ylim + np.array([0, step])), step=step))
ax_8a.yaxis.set_tick_params('major', labelsize=12)
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.savefig(op.join(figdir, 'fig9' + outext))
'''

#%% WORD-LEVEL ANALYSIS
'''
wdata = pd.read_csv(op.join(indir, 'wordLevelData.tsv'), sep='\t')
column_order = ['subj', 'div', 'adj', 'idn', 'num', 'sandwich', 'leftover',
                'foil', 'snd', 'lft', 'fht', 'rt_snd', 'rt_lft', 'rt_fht',
                'cond']
figeight = wdata[column_order]
# only look at trials that had both sandwich and leftovers
figeight = figeight[np.in1d(figeight['cond'], [101, 202, 1010, 2020])]
figeight = figeight.set_index('num', drop=False).loc[['thr', 'six']]  # no both
figeight['num'][figeight['num'] == 'six'] = 6
figeight['num'][figeight['num'] == 'thr'] = 3
figeight['num'] = figeight['num'].astype(int)
figeight = figeight.reset_index(drop=True)
aggregation_dict = dict(subj=uniq, div=uniq, adj=uniq, idn=uniq, num=uniq,
                        cond=uniq, sandwich=sum, leftover=sum, foil=sum,
                        snd=sum, lft=sum, fht=sum,
                        rt_snd=chsq_rt, rt_lft=chsq_rt, rt_fht=chsq_rt)
# grouping
by_sin = figeight.groupby(['subj', 'idn', 'num'])
by_sn = figeight.groupby(['subj', 'num'])
by_si = figeight.groupby(['subj', 'idn'])
by_in = figeight.groupby(['idn', 'num'])
by_s = figeight.groupby(['subj'])
by_i = figeight.groupby(['idn'])
by_n = figeight.groupby(['num'])
column_order.remove('subj')
by_s = by_s.aggregate(aggregation_dict)[column_order]
column_order.remove('idn')
by_i = by_i.aggregate(aggregation_dict)[column_order]
by_si = by_si.aggregate(aggregation_dict)[column_order]
column_order.remove('num')
by_in = by_in.aggregate(aggregation_dict)[column_order]
column_order = column_order[:2] + ['idn'] + column_order[2:]
by_n = by_n.aggregate(aggregation_dict)[column_order]
by_sn = by_sn.aggregate(aggregation_dict)[column_order]
column_order.remove('idn')
by_sin = by_sin.aggregate(aggregation_dict)[column_order]
'''
