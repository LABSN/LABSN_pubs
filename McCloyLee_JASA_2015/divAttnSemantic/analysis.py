# -*- coding: utf-8 -*-
"""
=====================================
Script 'Analyze divAttnSemantic data'
=====================================

This script analyses stuff.
"""
# Author: Dan McCloy <drmccloy@uw.edu>
#
# License: BSD (3-clause)

import json
import numpy as np
import pandas as pd
import os.path as op
from glob import glob
from itertools import chain
import scipy.stats as ss
import matplotlib.pyplot as plt
from matplotlib import lines, patches, rcParams


def box_off(ax):
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='y', direction='out')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')


def logit(pct, max_events=None):
    """Convert percentage (expressed in the range [0, 1]) to logit.

    Parameters
    ----------
    pct : float | array-like
        the occurrence proportion.
    max_events : int | None
        the number of events used to calculate ``pct``.  Used in a correction
        factor for cases when ``pct`` is 0 or 1, to prevent returning ``inf``.
        If ``None``, no correction is done, and ``inf`` or ``-inf`` may result.
    """
    if max_events is not None:
        # add equivalent of half an event to 0s, and subtract same from 1s
        corr_factor = 0.5 / max_events
        for loc in zip(*np.where(pct == 0)):
            pct.iloc[loc] = corr_factor
        for loc in zip(*np.where(pct == 1)):
            pct.iloc[loc] = 1 - corr_factor
    return np.log(pct / (np.ones_like(pct) - pct))


def rtwr(df):
    """Reaction time wrapper function specific to this experiment.
    """
    rt = [x for x in chain.from_iterable(df['target_RTs'].tolist()) if x > 0]
    return(np.mean(rt))


def format_pval(pval, latex=True):
    """Format a p-value
    """
    pv = []
    for p in pval:
        if p > 0.05:
            if latex:
                pv.append('$n.s.$')
            else:
                pv.append('n.s.')
        elif p > 0.01:
            if latex:
                pv.append('$p < 0.05$')
            else:
                pv.append('p < 0.05')
        elif np.log10(p) > -3:
            prc = int(np.log10(p))
            if latex:
                pv.append('$p < {}$'.format(np.round(10**prc, np.abs(prc))))
            else:
                pv.append('p < {}'.format(np.round(10**prc, np.abs(prc))))
        elif latex:
            pv.append('$p < 10^{{{}}}$'.format(int(np.log10(p))))
        else:
            pv.append('p < 10^{}'.format(int(np.log10(p))))
    return(pv)


def dprime(h, m, fa, cr, zero_correction=True):
    if zero_correction:
        a = 0.5
    else:
        a = 0
    return ss.norm.ppf((h + a) / (h + m + 2 * a)) - \
        ss.norm.ppf((fa + a) / (fa + cr + 2 * a))


def dpwr(df):
    """D-prime wrapper function specific to this experiment.
    """
    return dprime(sum(df['hits']), sum(df['miss']),
                  sum(df['f_alarm']) + sum(df['d_alarm']),
                  sum(df['corr_rej']))


def dpwr2(df):
    """D-prime wrapper function specific to this experiment.
    """
    return dprime(sum(df['hits']), sum(df['miss']),
                  sum(df['f_alarm']) + sum(df['d_alarm']),
                  sum(df['corr_rej2']))


def dprime_barplot(df, grouping=None, fix_bar_width=True, xlab=None, gap=None,
                   group_names=None, lines=False, err_bars=None, filename=None,
                   bar_kwargs=None, err_kwargs=None, line_kwargs=None,
                   ylim=None, ylab='d-prime', gn2=None, gr2=None,
                   brackets=None, bracket_text=None, bracket_kwargs=None,
                   figsize=None):
    """Generates optionally grouped barplots with connected line overlays.
    Parameters
    ----------
    df : pandas.DataFrame
        Data to be plotted. If not already a ``DataFrame``, will be coerced to
        one. Passing a ``numpy.ndarray`` as ``df`` should work transparently,
        with sequential integers assigned as column names.
    groups : None | list
        List of lists containing the integers in ``range(len(df.columns))``,
        with sub-lists indicating the desired grouping. For example, if your
        DataFrame has four columns and you want the first bar isolated and the
        remaining three grouped, then specify ``grouping=[[0], [1, 2, 3]]``.
    fix_bar_width : bool
        Should all bars be same width, or all groups be same width?
    xlab : list | None
        Labels for each bar to place along the x-axis. If ``None``, defaults
        to the column names of ``df``.
    group_names : list | None
        Additional labels to go below the individual bar labels on the x-axis.
    lines : bool
        Should lines be plotted over the bars? Values are drawn from the rows
        of ``df``.
    err_bars : str | None
        Type of error bars to be added to the barplot. Possible values are
        ``'sd'`` for sample standard deviation, ``'se'`` for standard error of
        the mean, or ``'ci'`` for 95% confidence interval. If ``None``, no
        error bars will be plotted.
    filename : str
        Full path (absolute or relative) of the output file. At present only
        PDF format implemented.
    bar_kwargs : dict
        arguments passed to ``pyplot.bar()`` (e.g., color, linewidth).
    err_kwargs : dict
        arguments passed to ``pyplot.bar(error_kw)`` (e.g., ecolor, capsize).
    line_kwargs : dict
        arguments passed to ``pyplot.plot()`` (e.g., color, marker, linestyle).
    ylim : array-like | None
        force y limits
    ylab : str | None
        y axis label
    brackets : list | None
        Where to put significance brackets.  Scheme is similar to ``grouping``;
        if you want a bracket that is between groups instead of between bars,
        specify as [[0, 1], [2, 3]].
    bracket_text : str | list
        text to use on brackets.
    bracket_kwargs : dict
        arguments passed to ``pyplot.plot()`` (e.g., color, marker, linestyle).
    figsize : tuple | None
        dimensions of finished figure.
    gn2 : list
        third-tier group names.
    gr2 : list
        third-tier group locations.

    Returns
    -------
    p : pyplot figure subplot instance
    """
    if bracket_kwargs is None:
        bracket_kwargs = dict(color='k')
    if bar_kwargs is None:
        bar_kwargs = dict()
    if err_kwargs is None:
        err_kwargs = dict()
    if line_kwargs is None:
        line_kwargs = dict()
    if err_bars is not None:
        if not isinstance(err_bars, basestring):
            raise TypeError()
        if err_bars not in ['sd', 'se', 'ci']:
            raise ValueError('err_bars must be one of "sd", "se", or "ci".')
    if bar_kwargs is None:
        bar_kwargs = dict()
    if err_kwargs is None:
        err_kwargs = dict()
    if line_kwargs is None:
        line_kwargs = dict()
    if not isinstance(df, pd.core.frame.DataFrame):
        df = pd.DataFrame(df)
    err = None
    wid = 1.0
    if gap is None:
        gap = 0.2
    gap = gap * wid
    if grouping is None:
        grouping = [range(len(df.columns))]
        fix_bar_width = True
    gr_flat = list(chain.from_iterable(grouping))
    num_bars = len(gr_flat)
    if xlab is None:
        xlab = np.array(df.columns)[gr_flat].tolist()
    num_groups = len(grouping)
    group_sizes = [len(x) for x in grouping]
    grp_width = [wid * num_bars / float(num_groups) / float(siz)
                 for siz in group_sizes]
    indices = [grouping.index(grp) for grp in grouping
               for bar in xrange(num_bars) if bar in grp]
    if len(set(indices)) == 1:
        offsets = gap * (np.array(indices) + 2)
    else:
        offsets = gap * (np.array(indices) + 1)
    if fix_bar_width:
        bar_width = [wid for _ in xrange(num_bars)]
        bar_xpos = offsets + np.arange(num_bars)
    else:
        bar_width = [grp_width[x] for x in indices]
        bar_xpos = offsets + np.arange(num_bars) * ([0] + bar_width)[:-1]
    bar_ypos = df.mean()[gr_flat]
    pts_xpos = bar_xpos + 0.5 * np.array(bar_width)
    # basic plot setup
    plt.figure(figsize=figsize)
    p = plt.subplot(1, 1, 1)
    # barplot & line overlays
    if err_bars is not None:
        if err_bars == 'sd':
            err = df.std()[gr_flat]
        elif err_bars == 'se':
            err = df.std()[gr_flat] / np.sqrt(len(df.index))
        else:  # 95% conf int
            err = 1.96 * df.std()[gr_flat] / np.sqrt(len(df.index))
        bar_kwargs['yerr'] = err
    else:
        err = [0] * num_bars  # needs to be defined, for significance brackets
    p.bar(bar_xpos, bar_ypos, bar_width, error_kw=err_kwargs, **bar_kwargs)
    if lines:
        for idx in df.index.tolist():
            pts = [df[col][idx]
                   for col in np.array(df.columns)[gr_flat].tolist()]
            p.plot(pts_xpos, pts, **line_kwargs)
        max_pts = [df[col].max() for col in
                   np.array(df.columns)[gr_flat].tolist()]
    # annotation
    box_off(p)
    p.tick_params(axis='x', length=0, pad=16)
    plt.ylabel(ylab)
    plt.xticks(pts_xpos, xlab, va='baseline')
    # axis limits
    if ylim is not None:
        plt.ylim(ylim)
    p.set_xlim(0, bar_xpos[-1] + bar_width[-1] + gap)
    # significance brackets
    if brackets is not None:
        offset = np.diff(p.get_ylim()) * 0.025
        brack_h = np.diff(p.get_ylim()) * 0.05 + offset
        for b, t in zip(brackets, bracket_text):
            if not hasattr(b[0], 'append'):
                if lines:
                    ymax = np.array(max_pts)[b]
                else:
                    ymax = np.array(bar_ypos)[b] + np.array(err)[b]
                h = np.max(ymax) + brack_h
                # first vertical
                x0 = (pts_xpos[b[0]],) * 2
                y0 = (ymax[0] + offset, h)
                # second vertical
                x1 = (pts_xpos[b[1]],) * 2
                y1 = (ymax[1] + offset, h)
                # horizontal
                xh = (x0[0], x1[0])
                yh = (h, h)
            else:
                txt_offset = 0.1
                bb = list(chain.from_iterable(b))
                h = np.max(np.array(bar_ypos)[bb] + np.array(err)[bb]) \
                    + 2 * brack_h + txt_offset
                x0 = (np.mean(np.array(pts_xpos)[b[0]]),) * 2
                x1 = (np.mean(np.array(pts_xpos)[b[1]]),) * 2
                y0 = (np.max(np.array(bar_ypos)[b[0]] + np.array(err)[b[0]]) +
                      brack_h + txt_offset, h)
                y1 = (np.max(np.array(bar_ypos)[b[1]] + np.array(err)[b[1]]) +
                      brack_h + txt_offset, h)
                xh = (x0[0], x1[0])
                yh = (h, h)
            for x, y in zip([x0, x1, xh], [y0, y1, yh]):
                p.plot(x, y, **bracket_kwargs)
            p.text(np.mean(xh), h + offset/2., t, ha='center', va='bottom')
    # group names
    fontsize = rcParams['font.size']
    yoffset = -2.1 * fontsize
    if group_names is not None:
        gs = np.r_[0, np.cumsum(group_sizes)]
        group_name_pos = [np.mean(pts_xpos[a:b])
                          for a, b in zip(gs[:-1], gs[1:])]
        for gnp, gn in zip(group_name_pos, group_names):
            p.annotate(gn, xy=(gnp, 0), xytext=(0, yoffset),
                       textcoords='offset points', ha='center', va='baseline')
    if gn2 is not None:
        if group_names is not None:
            yoffset = -3.2 * fontsize
        gs2 = np.r_[0, np.cumsum([len(x) for x in gr2])]
        grp2 = [np.mean(pts_xpos[a:b]) for a, b in zip(gs2[:-1], gs2[1:])]
        for gn, gp in zip(gn2, grp2):
            p.annotate(gn, xy=(gp, 0), xytext=(0, yoffset),
                       textcoords='offset points', ha='center', va='baseline')
    plt.draw()
    plt.tight_layout()
    if gn2 is None or group_names is None:
        plt.subplots_adjust(bottom=0.15)
    else:
        plt.subplots_adjust(bottom=0.2)
    # output file
    if filename is not None:
        plt.savefig(filename, format='pdf', transparent=True)
    #return p


indir = 'processedData/pandas'
andir = 'analysis'
infiles = glob(op.join(indir, '*.pickle'))

all_data = pd.DataFrame()
cats = ['body', 'plants', 'animals', 'weather', 'clothing', 'fooddrink',
        'furniture']
foil_conds = ['snd_l', 'snd_r', 'snd_c', 'lft_l', 'lft_r', 'lft_c', 'snd',
              'lft', 'ctrl_snd', 'test_snd', 'ctrl_lft', 'test_lft']
conditions = ['ctrl', 'test', 'sel', 'div', 'adj', 'sep',
              'ctrl_sel', 'ctrl_div', 'ctrl_sep', 'ctrl_adj',
              'test_sel', 'test_div', 'test_sep', 'test_adj',
              'ctrl_adj_c', 'ctrl_adj_r', 'ctrl_adj_l', 'ctrl_adj_s',
              'ctrl_sep_e', 'ctrl_sep_r', 'ctrl_sep_l', 'ctrl_sep_s',
              'test_adj_c', 'test_adj_r', 'test_adj_l', 'test_adj_s',
              'test_sep_e', 'test_sep_r', 'test_sep_l', 'test_sep_s'] \
    + cats + ['sel_' + x for x in cats] + ['div_' + x for x in cats] \
    + foil_conds

dp = {c: {} for c in conditions}
dp['subj'] = {}
dpfa = {c: {} for c in conditions}
dpfa['subj'] = {}
rt = {c: {} for c in conditions}
rt['subj'] = {}
dp_rand = {c: np.nan for c in conditions}

for f in infiles:
    s_data = pd.read_pickle(f)
    s_data.reset_index(inplace=True)
    subj_code = op.basename(f)[:2]
    # conditions
    s_data['subj'] = subj_code
    s_data['ctrl'] = s_data['phase'] == 'ctrl'
    s_data['test'] = s_data['phase'] == 'test'
    # individual streams
    s_data['l_e'] = [x[0] == 1 for x in s_data['attn']]
    s_data['l_c'] = [x[1] == 1 for x in s_data['attn']]
    s_data['r_c'] = [x[2] == 1 for x in s_data['attn']]
    s_data['r_e'] = [x[3] == 1 for x in s_data['attn']]
    # selective vs divided
    s_data['sel'] = [sum(x) == 1 for x in s_data['attn']]
    s_data['div'] = [sum(x) == 2 for x in s_data['attn']]
    s_data['ctrl_sel'] = np.logical_and(s_data['ctrl'], s_data['sel'])
    s_data['ctrl_div'] = np.logical_and(s_data['ctrl'], s_data['div'])
    s_data['test_sel'] = np.logical_and(s_data['test'], s_data['sel'])
    s_data['test_div'] = np.logical_and(s_data['test'], s_data['div'])
    # divided:adjacent
    s_data['adj'] = ['1, 1' in str(x) for x in s_data['attn']]
    s_data['adj_c'] = np.logical_and(s_data['r_c'], s_data['l_c'])
    s_data['adj_r'] = np.logical_and(s_data['r_c'], s_data['r_e'])
    s_data['adj_l'] = np.logical_and(s_data['l_c'], s_data['l_e'])
    s_data['adj_s'] = np.logical_or(s_data['adj_r'], s_data['adj_l'])
    # divided:separated
    s_data['sep_e'] = np.logical_and(s_data['r_e'], s_data['l_e'])
    s_data['sep_r'] = np.logical_and(s_data['r_e'], s_data['l_c'])
    s_data['sep_l'] = np.logical_and(s_data['r_c'], s_data['l_e'])
    s_data['sep_s'] = np.logical_or(s_data['sep_r'], s_data['sep_l'])
    s_data['sep'] = np.logical_or(s_data['sep_e'], s_data['sep_s'])
    # sandwiched foils
    s_data['snd_l'] = np.logical_and([np.sum(x, -1)[1] > 0 for x in
                                     s_data['codes']], s_data['sep_l'])
    s_data['snd_r'] = np.logical_and([np.sum(x, -1)[2] > 0 for x in
                                     s_data['codes']], s_data['sep_r'])
    s_data['snd_c'] = np.logical_and([np.logical_or(np.sum(x, -1)[1] > 0,
                                                    np.sum(x, -1)[2] > 0) for x
                                     in s_data['codes']], s_data['sep_e'])
    s_data['thinsnd'] = np.logical_or(s_data['snd_l'], s_data['snd_r'])
    s_data['snd'] = np.logical_or(s_data['thinsnd'], s_data['snd_c'])
    s_data['ctrl_snd'] = np.logical_and(s_data['ctrl'], s_data['snd'])
    s_data['test_snd'] = np.logical_and(s_data['test'], s_data['snd'])
    # leftover foils
    s_data['lft_l'] = np.logical_and([np.sum(x, -1)[3] > 0 for x in
                                     s_data['codes']], s_data['sep_l'])
    s_data['lft_r'] = np.logical_and([np.sum(x, -1)[0] > 0 for x in
                                     s_data['codes']], s_data['sep_r'])
    s_data['lft_c'] = np.logical_and([np.logical_or(np.sum(x, -1)[0] > 0,
                                                    np.sum(x, -1)[3] > 0) for x
                                     in s_data['codes']], s_data['adj_c'])
    s_data['thinlft'] = np.logical_or(s_data['lft_l'], s_data['lft_r'])
    s_data['lft'] = np.logical_or(s_data['thinlft'], s_data['lft_c'])
    s_data['ctrl_lft'] = np.logical_and(s_data['ctrl'], s_data['lft'])
    s_data['test_lft'] = np.logical_and(s_data['test'], s_data['lft'])
    # phonetic
    s_data['ctrl_adj'] = np.logical_and(s_data['ctrl'], s_data['adj'])
    s_data['ctrl_sep'] = np.logical_and(s_data['ctrl'], s_data['sep'])
    s_data['ctrl_adj_c'] = np.logical_and(s_data['ctrl'], s_data['adj_c'])
    s_data['ctrl_adj_r'] = np.logical_and(s_data['ctrl'], s_data['adj_r'])
    s_data['ctrl_adj_l'] = np.logical_and(s_data['ctrl'], s_data['adj_l'])
    s_data['ctrl_adj_s'] = np.logical_and(s_data['ctrl'], s_data['adj_s'])
    s_data['ctrl_sep_e'] = np.logical_and(s_data['ctrl'], s_data['sep_e'])
    s_data['ctrl_sep_r'] = np.logical_and(s_data['ctrl'], s_data['sep_r'])
    s_data['ctrl_sep_l'] = np.logical_and(s_data['ctrl'], s_data['sep_l'])
    s_data['ctrl_sep_s'] = np.logical_and(s_data['ctrl'], s_data['sep_s'])
    # semantic
    s_data['test_adj'] = np.logical_and(s_data['test'], s_data['adj'])
    s_data['test_sep'] = np.logical_and(s_data['test'], s_data['sep'])
    s_data['test_adj_c'] = np.logical_and(s_data['test'], s_data['adj_c'])
    s_data['test_adj_r'] = np.logical_and(s_data['test'], s_data['adj_r'])
    s_data['test_adj_l'] = np.logical_and(s_data['test'], s_data['adj_l'])
    s_data['test_adj_s'] = np.logical_and(s_data['test'], s_data['adj_s'])
    s_data['test_sep_e'] = np.logical_and(s_data['test'], s_data['sep_e'])
    s_data['test_sep_r'] = np.logical_and(s_data['test'], s_data['sep_r'])
    s_data['test_sep_l'] = np.logical_and(s_data['test'], s_data['sep_l'])
    s_data['test_sep_s'] = np.logical_and(s_data['test'], s_data['sep_s'])
    # performance
    s_data['num_targs'] = [len(x) for x in s_data['targ_words']]
    s_data['num_dists'] = [len(x) for x in s_data['dist_words']]
    s_data['num_norms'] = 12 - s_data['num_targs'] - s_data['num_dists']
    s_data['hits'] = [sum([x != -1 for x in y]) for y in s_data['target_RTs']]
    s_data['miss'] = [sum([x == -1 for x in y]) for y in s_data['target_RTs']]
    s_data['d_alarm'] = [sum([x != -1 for x in y])
                         for y in s_data['distractor_RTs']]
    s_data['f_alarm'] = [len(x) for x in s_data['false_alarm_times']]
    s_data['corr_rej'] = [sum([x == -1 for x in y])
                          for y in s_data['distractor_RTs']]
    s_data['corr_rej2'] = s_data['num_norms'] - s_data['f_alarm']
    s_data['h_rate'] = s_data['hits'] / s_data['num_targs'].astype(float)
    s_data['d_rate'] = s_data['d_alarm'] / s_data['num_dists'].astype(float)
    s_data['f_rate'] = s_data['f_alarm'] / (12 - s_data['num_targs'] -
                                            s_data['num_dists']).astype(float)
    # boolean for category of target streams
    s_data['targ_cats'] = np.ma.masked_array(np.array(
        s_data['cats'].values.tolist()), mask=np.logical_not(
        np.array(s_data['attn'].values.tolist()).astype(bool))).tolist()
    for cat in cats:
        s_data[cat] = False
    for idx, _ in s_data.iterrows():
        s_data['body'][idx] = 'body' in s_data['targ_cats'][idx]
        s_data['plants'][idx] = 'plants' in s_data['targ_cats'][idx]
        s_data['animals'][idx] = 'animals' in s_data['targ_cats'][idx]
        s_data['weather'][idx] = 'weather' in s_data['targ_cats'][idx]
        s_data['clothing'][idx] = 'clothing' in s_data['targ_cats'][idx]
        s_data['fooddrink'][idx] = 'fooddrink' in s_data['targ_cats'][idx]
        s_data['furniture'][idx] = 'furniture' in s_data['targ_cats'][idx]
    # selective / divided & categories
    s_data['sel_body'] = np.logical_and(s_data['sel'], s_data['body'])
    s_data['sel_plants'] = np.logical_and(s_data['sel'], s_data['plants'])
    s_data['sel_animals'] = np.logical_and(s_data['sel'], s_data['animals'])
    s_data['sel_weather'] = np.logical_and(s_data['sel'], s_data['weather'])
    s_data['sel_clothing'] = np.logical_and(s_data['sel'], s_data['clothing'])
    s_data['sel_fooddrink'] = np.logical_and(s_data['sel'], s_data['fooddrink'])
    s_data['sel_furniture'] = np.logical_and(s_data['sel'], s_data['furniture'])
    s_data['div_body'] = np.logical_and(s_data['div'], s_data['body'])
    s_data['div_plants'] = np.logical_and(s_data['div'], s_data['plants'])
    s_data['div_animals'] = np.logical_and(s_data['div'], s_data['animals'])
    s_data['div_weather'] = np.logical_and(s_data['div'], s_data['weather'])
    s_data['div_clothing'] = np.logical_and(s_data['div'], s_data['clothing'])
    s_data['div_fooddrink'] = np.logical_and(s_data['div'], s_data['fooddrink'])
    s_data['div_furniture'] = np.logical_and(s_data['div'], s_data['furniture'])
    # concatenate all subjects together
    all_data = pd.concat([all_data, s_data])
    # dprimes & reaction times
    for c in conditions:
        rt[c][subj_code] = rtwr(s_data[s_data[c]])
        dp[c][subj_code] = dpwr(s_data[s_data[c]])
        dpfa[c][subj_code] = dpwr2(s_data[s_data[c]])
    dp['subj'][subj_code] = dpwr(s_data)
    dpfa['subj'][subj_code] = dpwr2(s_data)

"""
# simulate a random observer
rand_data = s_data.copy()
rand_data['hits'] = [np.random.binomial(x, 0.5, 1)[0]
                     for x in rand_data['num_targs']]
rand_data['miss'] = rand_data['num_targs'] - rand_data['hits']
rand_data['f_alarm'] = [np.random.binomial(x, 0.5, 1)[0] for x in
                        12 - rand_data['num_targs'] - rand_data['num_dists']]
rand_data['d_alarm'] = [np.random.binomial(x, 0.5, 1)[0]
                        for x in rand_data['num_dists']]
rand_data['corr_rej'] = 12 - rand_data['num_targs'] - rand_data['f_alarm']
for c in conditions:
    dp_rand[c] = dpwr(rand_data[rand_data[c]])
dprime_rand = pd.DataFrame(dp_rand, index=[0])
"""
# dprime2 does not include non-targ non-dist non-presses as correct rejections
dprime2 = pd.DataFrame(dp)
dprimes = pd.DataFrame(dpfa)
rtimes = pd.DataFrame(rt)
col_order = ['subj'] + conditions
dprimes = dprimes[col_order]
all_data.reset_index(inplace=True, drop=True)
dprimes.to_csv(op.join(andir, 'dprimes.tab'), sep='\t')
all_data.to_csv(op.join(andir, 'alldata.tab'), sep='\t')
#sio.savemat(op.join(andir, 'alldata.mat'), dict(all_data), oned_as='row')

hrate = {c: {} for c in conditions}
drate = {c: {} for c in conditions}
frate = {c: {} for c in conditions}
for s in list(set(all_data['subj'])):
    d = all_data[all_data['subj'] == s]
    for c in conditions:
        dd = d[d[c]]
        hrate[c][s] = dd['hits'].sum() / dd['num_targs'].sum().astype(float)
        drate[c][s] = dd['d_alarm'].sum() / dd['num_dists'].sum().astype(float)
        frate[c][s] = dd['f_alarm'].sum() / (12 - dd['num_targs'] -
                                             dd['num_dists']
                                             ).sum().astype(float)
hrate = pd.DataFrame(hrate)
drate = pd.DataFrame(drate)
frate = pd.DataFrame(frate)

# # # # # # # # # # # # #
# within-subject t-test #
# # # # # # # # # # # # #
comp_a = ['ctrl', 'ctrl_div', 'ctrl_sel', 'test_div', 'ctrl_adj', 'test_adj',
          'ctrl_snd', 'test_snd']
comp_b = ['test', 'test_div', 'ctrl_div', 'test_sel', 'ctrl_sep', 'test_sep',
          'ctrl_lft', 'test_lft']
idx = ['ctrl_test', 'ctrl_test_div', 'ctrl_sel_div',
       'test_sel_div', 'ctrl_adj_sep', 'test_adj_sep',
       'ctrl_snd_lft', 'test_snd_lft']
# d-primes
a = dprimes[comp_a]
b = dprimes[comp_b]
t_vals, p_vals = ss.ttest_rel(a, b)
dp_ttests = pd.DataFrame(dict(t=t_vals, p=p_vals), index=idx)
# hit rates
a = hrate[comp_a]
b = hrate[comp_b]
max_hits = all_data['num_targs'][all_data['subj'] == all_data['subj'][0]].sum()
t_vals, p_vals = ss.ttest_rel(logit(a, max_hits), logit(b, max_hits))
hr_ttests = pd.DataFrame(dict(t=t_vals, p=p_vals), index=idx)
# foil rates
a = drate[comp_a]
b = drate[comp_b]
max_foil = all_data['num_dists'][all_data['subj'] == all_data['subj'][0]].sum()
t_vals, p_vals = ss.ttest_rel(logit(a, max_foil), logit(b, max_foil))
dr_ttests = pd.DataFrame(dict(t=t_vals, p=p_vals), index=idx)
# false alarm rates
a = frate[comp_a]
b = frate[comp_b]
max_fa = all_data['num_norms'][all_data['subj'] == all_data['subj'][0]].sum()
t_vals, p_vals = ss.ttest_rel(logit(a, max_fa), logit(b, max_fa))
fr_ttests = pd.DataFrame(dict(t=t_vals, p=p_vals), index=idx)
# reaction times
a = rtimes[comp_a]
b = rtimes[comp_b]
t_vals, p_vals = ss.ttest_rel(a, b)
rt_ttests = pd.DataFrame(dict(t=t_vals, p=p_vals), index=idx)

# save things
all_data.to_pickle(op.join(andir, 'alldata.pickle'))
dprimes.to_pickle(op.join(andir, 'dprimes.pickle'))
dp_ttests.to_pickle(op.join(andir, 'dp_t.pickle'))
hr_ttests.to_pickle(op.join(andir, 'hr_t.pickle'))
dr_ttests.to_pickle(op.join(andir, 'dr_t.pickle'))
fr_ttests.to_pickle(op.join(andir, 'fr_t.pickle'))
rt_ttests.to_pickle(op.join(andir, 'rt_t.pickle'))

# dump word-level data for mixed model analysis in R
streams = 4
waves = 12
stims = streams * waves

with open(op.join(andir, 'wordDurations.json')) as jd:
    word_durs = json.load(jd)
# set reaction time window ([0.1, 1.25] was hard-coded in divAttnSemantic.py)
minRT = 0.4
maxRT = 1.25

df = all_data
trialdata = None

for row in df.index:
    #date = np.tile(df.ix[row, 'datestring'], stims)
    subj = np.tile(df.ix[row, 'subj'], stims)
    #subn = np.tile(df.ix[row, 'subj_num'], stims)
    trial = np.tile(df.ix[row, 'trial'], stims)
    onset = np.array(df.ix[row, 'times']).ravel()
    stream = np.repeat(np.arange(streams), waves)
    attn = np.repeat(np.array(df.ix[row, 'attn']), waves).astype(bool)
    targ = np.sum([np.array(df.ix[row, 'times']) == x for x in
                   df.ix[row, 'target_times']], axis=0, dtype=bool).ravel()
    foil = np.sum([np.array(df.ix[row, 'times']) == x for x in
                   df.ix[row, 'distractor_times']], axis=0, dtype=bool).ravel()
    if len(foil) == 1:
        foil = np.zeros_like(targ)
        # the above is necessary to handle trials with no foils
    odbl = targ + foil
    catg = np.repeat(df.ix[row, 'cats'], waves)
    word = np.array(df.ix[row, 'words']).ravel()
    dur = np.array([word_durs[x] for x in word])
    #cond = np.tile(df.ix[row, 'cond'], stims)
    sem = np.tile(df.ix[row, 'test'], stims)
    div = np.tile(df.ix[row, 'div'], stims)
    adj = np.tile(df.ix[row, 'adj'], stims)
    #num = np.tile(df.ix[row, 'size'], stims)
    presses = df.ix[row, 'press_times']
    rt = np.zeros_like(onset) - 1
    for ix, ot in enumerate(onset):
        for p in presses:
            if ot + minRT < p < ot + maxRT:
                rt[ix] = p - ot
    td = pd.DataFrame(dict(subj=subj, trial=trial,
                           rawRT=rt, onset=onset, stream=stream, attn=attn,
                           targ=targ, foil=foil, odbl=odbl, catg=catg,
                           word=word, dur=dur, sem=sem, div=div, adj=adj
                           ), index=None)
    if trialdata is None:
        trialdata = td
    else:
        trialdata = pd.concat((trialdata, td), ignore_index=True)

column_order = ['subj', 'trial', 'sem', 'div', 'adj', 'stream', 'attn', 'catg',
                'word', 'targ', 'foil', 'odbl', 'onset', 'dur', 'rawRT']
trialdata = trialdata[column_order]
trialdata.to_csv(op.join(andir, 'wordLevelData.tsv'), sep='\t', index=False)

#%% END OF ANALYSIS, START OF PLOTTING %%#

# # # # # # # # # # # #
# general plot setup  #
# # # # # # # # # # # #
# useful bash command: fc-list | grep Libertine
rcParams['font.sans-serif'] = 'Linux Libertine Capitals O'
rcParams['font.serif'] = 'Linux Libertine O'
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 18
#rcParams['pdf.fonttype'] = 42
#rcParams['text.usetex'] = True
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Linux Libertine O'
rcParams['mathtext.it'] = 'Linux Libertine O:style=italic'
rcParams['mathtext.bf'] = 'Linux Libertine O:weight=bold'
rcParams['mathtext.sf'] = 'Linux Biolinum O'
rcParams['mathtext.cal'] = 'Linux Libertine O:style=italic'
"""
rcParams['mathtext.tt'] = 'Linux Libertine Mono O'
"""
#plt.ion()
ld = dict(marker='.', linestyle='-', color='k', alpha=0.3)
ed = dict(ecolor='k')
gn = ['phonetic', 'semantic']

# hcl colors
one = "#DE7429"
two = "#00AECF"
one_a = "#DE7429"
one_b = "#BC3F33"
two_a = "#00AECF"
two_b = "#0078CD"
one_ba = "#BC3F33"
one_bb = "#760000"
two_ba = "#0078CD"
two_bb = "#004DC1"

thre = "#DBA300"
four = "#00B2B7"
thre_a = "#DBA300"
thre_b = "#B36500"
four_a = "#00B2B7"
four_b = "#00806A"
thre_ba = "#B36500"
thre_bb = "#7A3500"
four_ba = "#00806A"
four_bb = "#00523D"

# # # # # # # # # # #
# semantic/phonetic #
# # # # # # # # # # #
conds = ['ctrl', 'test']
cl = [one, two, thre, four]
bd = dict(color=cl)
bk = dict(color='0.7', linestyle='-')
xl = gn
gr = [[0], [1]]  # None
fs = (4, 4)
# d-prime
fn = 'DP_PhonSem.pdf'
pv = dp_ttests['p'][['ctrl_test']].tolist()
dprime_barplot(dprimes[conds], grouping=gr, xlab=xl, group_names=None, gap=0.6,
               err_bars='se', filename=op.join(andir, fn), figsize=fs,
               bar_kwargs=bd, line_kwargs=ld, err_kwargs=ed, brackets=[[0, 1]],
               bracket_text=format_pval(pv), bracket_kwargs=bk, ylim=(0, 5.5))
# reaction times
fn = 'RT_PhonSem.pdf'
pv = rt_ttests['p'][['ctrl_test']].tolist()
dprime_barplot(rtimes[conds], grouping=gr, xlab=xl, group_names=None, gap=0.6,
               err_bars='se', filename=op.join(andir, fn), figsize=fs,
               bar_kwargs=bd, line_kwargs=ld, err_kwargs=ed, brackets=[[0, 1]],
               ylab='reaction time (s)', ylim=(0.4, 0.9),
               bracket_text=format_pval(pv), bracket_kwargs=bk)
# hit rate
fn = 'HR_PhonSem.pdf'
pv = hr_ttests['p'][['ctrl_test']].tolist()
dprime_barplot(hrate[conds], grouping=gr, xlab=xl, group_names=None, gap=0.6,
               err_bars='se', filename=op.join(andir, fn), figsize=fs,
               bar_kwargs=bd, line_kwargs=ld, err_kwargs=ed, brackets=[[0, 1]],
               ylab='Hit rate', bracket_text=format_pval(pv), ylim=(0, 1.1),
               bracket_kwargs=bk)
# foil rate
fn = 'DR_PhonSem.pdf'
pv = dr_ttests['p'][['ctrl_test']].tolist()
dprime_barplot(drate[conds], grouping=gr, xlab=xl, group_names=None, gap=0.6,
               err_bars='se', filename=op.join(andir, fn), figsize=fs,
               bar_kwargs=bd, line_kwargs=ld, err_kwargs=ed,
               ylab='False alarm rate (foils)', ylim=(0, 0.3),
               bracket_text=format_pval(pv), brackets=[[0, 1]],
               bracket_kwargs=bk)
# false alarm rate
fn = 'FR_PhonSem.pdf'
pv = fr_ttests['p'][['ctrl_test']].tolist()
dprime_barplot(frate[conds], grouping=gr, xlab=xl, group_names=None, gap=0.6,
               err_bars='se', filename=op.join(andir, fn), figsize=fs,
               bar_kwargs=dict(color=cl), line_kwargs=ld, err_kwargs=ed,
               ylab='False alarm rate (non-foils)', ylim=(0, 0.3),
               bracket_text=format_pval(pv), brackets=[[0, 1]],
               bracket_kwargs=bk)
# combined foils and false alarms
fn = 'DRFR_PhonSem.pdf'
pv = dr_ttests['p'][['ctrl_test']].tolist() + \
    fr_ttests['p'][['ctrl_test']].tolist()
data = drate[conds].join(frate[conds], how='outer', lsuffix='_foils')
gr = [[0, 1], [2, 3]]
xl = ['phon.', 'sem.'] * 2
gnn = ['Foils', 'Non-foils']
dprime_barplot(data, grouping=gr, xlab=xl, group_names=gnn, gap=0.6,
               err_bars='se', filename=op.join(andir, fn), figsize=(6, 4),
               bar_kwargs=dict(color=cl), line_kwargs=ld, err_kwargs=ed,
               ylab='False alarm rate', ylim=(0, 0.3),
               bracket_text=format_pval(pv), brackets=[[0, 1], [2, 3]],
               bracket_kwargs=bk)
plt.close('all')

# # # # # # # # # # # # # # # # # # # # #
# semantic/phonetic; selective/divided  #
# # # # # # # # # # # # # # # # # # # # #
conds = ['ctrl_sel', 'ctrl_div', 'test_sel', 'test_div']
cl = [one_a, one_b, two_a, two_b, thre_a, thre_b, four_a, four_b]
xl = ['sel.', 'div.', 'sel.', 'div.']
bd = dict(color=cl)
gr = [[0, 1], [2, 3]]
fs = (4, 4)
# d-primes
fn = 'DP_PhonSemSelDiv.pdf'
pv = dp_ttests['p'][['ctrl_sel_div', 'test_sel_div']].tolist()
dprime_barplot(dprimes[conds], grouping=gr, xlab=xl, group_names=gn,
               err_bars='se', filename=op.join(andir, fn), figsize=fs,
               bar_kwargs=bd, line_kwargs=ld, err_kwargs=ed,
               brackets=gr, bracket_text=format_pval(pv),
               bracket_kwargs=bk, ylim=(0, 5.5))

# reaction times
fn = 'RT_PhonSemSelDiv.pdf'
pv = rt_ttests['p'][['ctrl_sel_div', 'test_sel_div']].tolist()
dprime_barplot(rtimes[conds], grouping=gr, xlab=xl, group_names=None,
               err_bars='se', filename=op.join(andir, fn), figsize=fs,
               bar_kwargs=bd, line_kwargs=ld, err_kwargs=ed,
               ylab='reaction time (s)', ylim=(0.4, 0.9), bracket_kwargs=bk,
               brackets=gr, bracket_text=format_pval(pv))
# hit rate
fn = 'HR_PhonSemSelDiv.pdf'
pv = hr_ttests['p'][['ctrl_sel_div', 'test_sel_div']].tolist()
dprime_barplot(hrate[conds], grouping=gr, xlab=xl, group_names=None,
               err_bars='se', filename=op.join(andir, fn), figsize=fs,
               bar_kwargs=bd, line_kwargs=ld, err_kwargs=ed,
               ylab='Hit rate', ylim=(0, 1.1), bracket_kwargs=bk,
               brackets=gr, bracket_text=format_pval(pv))
# foil rate
fn = 'DR_PhonSemSelDiv.pdf'
pv = dr_ttests['p'][['ctrl_sel_div', 'test_sel_div']].tolist()
dprime_barplot(drate[conds], grouping=gr, xlab=xl, group_names=None,
               err_bars='se', filename=op.join(andir, fn), figsize=fs,
               bar_kwargs=bd, line_kwargs=ld, err_kwargs=ed,
               ylab='False alarm rate (foils)', ylim=(0, 0.3),
               bracket_kwargs=bk, brackets=gr, bracket_text=format_pval(pv))
# false alarm rate
fn = 'FR_PhonSemSelDiv.pdf'
pv = fr_ttests['p'][['ctrl_sel_div', 'test_sel_div']].tolist()
dprime_barplot(frate[conds], grouping=gr, xlab=xl, group_names=None,
               err_bars='se', filename=op.join(andir, fn), figsize=fs,
               bar_kwargs=bd, line_kwargs=ld, err_kwargs=ed,
               ylab='False alarm rate (non-foils)', ylim=(0, 0.3),
               bracket_kwargs=bk, brackets=gr, bracket_text=format_pval(pv))
# combined foils and false alarms
fn = 'DRFR_PhonSemSelDiv.pdf'
pv = dr_ttests['p'][['ctrl_sel_div', 'test_sel_div']].tolist() + \
    fr_ttests['p'][['ctrl_sel_div', 'test_sel_div']].tolist()
data = drate[conds].join(frate[conds], how='outer', lsuffix='_foils')
gr = [[0, 1], [2, 3], [4, 5], [6, 7]]
xl = xl * 2  # ['phon.', 'sem.'] * 2
gnn = ['phon.', 'sem.'] * 2
ggnn = ['Foils', 'Non-foils']
gr2 = [[0, 1, 2, 3], [4, 5, 6, 7]]
dprime_barplot(data, grouping=gr, xlab=xl, group_names=None, gn2=ggnn, gr2=gr2,
               err_bars='se', filename=op.join(andir, fn), figsize=(6, 4),
               bar_kwargs=dict(color=cl), line_kwargs=ld, err_kwargs=ed,
               ylab='False alarm rate', ylim=(0, 0.3),
               bracket_text=format_pval(pv), brackets=gr,
               bracket_kwargs=bk)
plt.close('all')

# # # # # # # # # # # # # # # # # # # # #
# semantic/phonetic; adjacent/separate  #
# # # # # # # # # # # # # # # # # # # # #
conds = ['ctrl_adj', 'ctrl_sep', 'test_adj', 'test_sep']
cl = [one_ba, one_bb, two_ba, two_bb, thre_ba, thre_bb, four_ba, four_bb]
xl = ['adj.', 'sep.', 'adj.', 'sep.']
bd = dict(color=cl)
gr = [[0, 1], [2, 3]]
fs = (4, 4)
# d-primes
dprime_barplot(dprimes[conds], grouping=gr, xlab=xl, group_names=gn, gap=0.4,
               err_bars='se', filename=op.join(andir, 'DP_PhonSemAdjSep.pdf'),
               bar_kwargs=bd, line_kwargs=ld, err_kwargs=ed,
               ylim=(-1, 5), figsize=(5, 6))
# reaction times
fn = 'RT_PhonSemAdjSep.pdf'
pv = rt_ttests['p'][['ctrl_adj_sep', 'test_adj_sep']].tolist()
dprime_barplot(rtimes[conds], grouping=gr, xlab=xl, group_names=None, gap=0.4,
               err_bars='se', filename=op.join(andir, fn), figsize=fs,
               bar_kwargs=bd, line_kwargs=ld, err_kwargs=ed,
               ylab='reaction time (s)', ylim=(0.4, 0.9), bracket_kwargs=bk,
               brackets=gr, bracket_text=format_pval(pv))
# hit rate
fn = 'HR_PhonSemAdjSep.pdf'
pv = hr_ttests['p'][['ctrl_adj_sep', 'test_adj_sep']].tolist()
dprime_barplot(hrate[conds], grouping=gr, xlab=xl, group_names=None, gap=0.4,
               err_bars='se', filename=op.join(andir, fn), figsize=fs,
               bar_kwargs=bd, line_kwargs=ld, err_kwargs=ed,
               ylab='Hit rate', ylim=(0, 1.1), bracket_kwargs=bk,
               brackets=gr, bracket_text=format_pval(pv))
# foil rate
fn = 'DR_PhonSemAdjSep.pdf'
pv = dr_ttests['p'][['ctrl_adj_sep', 'test_adj_sep']].tolist()
dprime_barplot(drate[conds], grouping=gr, xlab=xl, group_names=None, gap=0.4,
               err_bars='se', filename=op.join(andir, fn), figsize=fs,
               bar_kwargs=bd, line_kwargs=ld, err_kwargs=ed,
               ylab='False alarm rate (foils)', ylim=(0, 0.3),
               bracket_kwargs=bk, brackets=gr, bracket_text=format_pval(pv))
# false alarm rate
fn = 'FR_PhonSemAdjSep.pdf'
pv = fr_ttests['p'][['ctrl_adj_sep', 'test_adj_sep']].tolist()
dprime_barplot(frate[conds], grouping=gr, xlab=xl, group_names=None, gap=0.4,
               err_bars='se', filename=op.join(andir, fn), figsize=fs,
               bar_kwargs=bd, line_kwargs=ld, err_kwargs=ed,
               ylab='False alarm rate (non-foils)', ylim=(0, 0.3),
               bracket_kwargs=bk, brackets=gr, bracket_text=format_pval(pv))
# combined foils and false alarms
fn = 'DRFR_PhonSemAdjSep.pdf'
pv = dr_ttests['p'][['ctrl_adj_sep', 'test_adj_sep']].tolist() + \
    fr_ttests['p'][['ctrl_adj_sep', 'test_adj_sep']].tolist()
data = drate[conds].join(frate[conds], how='outer', lsuffix='_foils')
gr = [[0, 1], [2, 3], [4, 5], [6, 7]]
xl = xl * 2  # ['phon.', 'sem.'] * 2
gnn = ['phon.', 'sem.'] * 2
ggnn = ['Foils', 'Non-foils']
gr2 = [[0, 1, 2, 3], [4, 5, 6, 7]]
dprime_barplot(data, grouping=gr, xlab=xl, group_names=None, gn2=ggnn, gr2=gr2,
               err_bars='se', filename=op.join(andir, fn), figsize=(6, 4),
               bar_kwargs=dict(color=cl), line_kwargs=ld, err_kwargs=ed,
               ylab='False alarm rate', ylim=(0, 0.3), gap=0.4,
               bracket_text=format_pval(pv), brackets=gr, bracket_kwargs=bk)
plt.close('all')

# # # # # # # # # # # # # # # # # #
# semantic/phonetic; all adjacent #
# # # # # # # # # # # # # # # # # #
conds = ['ctrl_adj_l', 'ctrl_adj_c', 'ctrl_adj_r',
         'test_adj_l', 'test_adj_c', 'test_adj_r']
cl = [one_ba, one_bb, one_ba, two_ba, two_bb, two_ba]
xl = ['L', 'C', 'R', 'L', 'C', 'R']
bd = dict(color=cl)
gr = [[0, 1, 2], [3, 4, 5]]
fs = (4, 4)
# d-primes
dprime_barplot(dprimes[conds], grouping=gr, xlab=xl, group_names=gn,
               err_bars='se', filename=op.join(andir, 'DP_PhonSemAdjLRC.pdf'),
               bar_kwargs=bd, line_kwargs=ld, err_kwargs=ed,
               ylim=(-1, 5), figsize=(5, 6))
# reaction times
fn = 'RT_PhonSemAdjLRC.pdf'
#pv = rt_ttests['p'][['ctrl_adj_sep', 'test_adj_sep']].tolist()
dprime_barplot(rtimes[conds], grouping=gr, xlab=xl, group_names=gn,
               err_bars='se', filename=op.join(andir, fn), figsize=fs,
               bar_kwargs=bd, line_kwargs=ld, err_kwargs=ed,
               ylab='reaction time (s)', ylim=(0.4, 0.9))  #, bracket_kwargs=bk,
               #brackets=gr, bracket_text=format_pval(pv))
# hit rate
fn = 'HR_PhonSemAdjLRC.pdf'
#pv = hr_ttests['p'][['ctrl_adj_sep', 'test_adj_sep']].tolist()
dprime_barplot(hrate[conds], grouping=gr, xlab=xl, group_names=gn,
               err_bars='se', filename=op.join(andir, fn), figsize=fs,
               bar_kwargs=bd, line_kwargs=ld, err_kwargs=ed,
               ylab='Hit rate', ylim=(0, 1.1))  # , bracket_kwargs=bk,
               #brackets=gr, bracket_text=format_pval(pv))
# foil rate
fn = 'DR_PhonSemAdjLRC.pdf'
#pv = dr_ttests['p'][['ctrl_adj_sep', 'test_adj_sep']].tolist()
dprime_barplot(drate[conds], grouping=gr, xlab=xl, group_names=gn,
               err_bars='se', filename=op.join(andir, fn), figsize=fs,
               bar_kwargs=bd, line_kwargs=ld, err_kwargs=ed,
               ylab='False alarm rate (foils)', ylim=(0, 0.3))  # ,
               #bracket_kwargs=bk, brackets=gr, bracket_text=format_pval(pv))
# false alarm rate
fn = 'FR_PhonSemAdjLRC.pdf'
#pv = fr_ttests['p'][['ctrl_adj_sep', 'test_adj_sep']].tolist()
dprime_barplot(frate[conds], grouping=gr, xlab=xl, group_names=gn,
               err_bars='se', filename=op.join(andir, fn), figsize=fs,
               bar_kwargs=bd, line_kwargs=ld, err_kwargs=ed,
               ylab='False alarm rate (non-foils)', ylim=(0, 0.3))  # ,
               #bracket_kwargs=bk, brackets=gr, bracket_text=format_pval(pv))
# combined foils and false alarms
fn = 'DRFR_PhonSemAdjLRC.pdf'
#pv = dr_ttests['p'][['ctrl_adj_sep', 'test_adj_sep']].tolist() + \
#    fr_ttests['p'][['ctrl_adj_sep', 'test_adj_sep']].tolist()
data = drate[conds].join(frate[conds], how='outer', lsuffix='_foils')
gr = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
xl = xl * 2
gnn = ['phon.', 'sem.'] * 2
ggnn = ['Foils', 'Non-foils']
gr2 = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]
dprime_barplot(data, grouping=gr, xlab=xl, group_names=gnn, gn2=ggnn, gr2=gr2,
               err_bars='se', filename=op.join(andir, fn), figsize=(5, 4),
               bar_kwargs=dict(color=cl), line_kwargs=ld, err_kwargs=ed,
               ylab='False alarm rate', ylim=(0, 0.3))  # ,
               #bracket_text=format_pval(pv), brackets=gr, bracket_kwargs=bk)
plt.close('all')

# # # # # # # # # # # # # # # # # #
# semantic/phonetic; all divided  #
# # # # # # # # # # # # # # # # # #
conds = ['ctrl_sep_l', 'ctrl_sep_s', 'ctrl_sep_r',
         'test_sep_l', 'test_sep_s', 'test_sep_r']
cl = [one_ba, one_bb, one_ba, two_ba, two_bb, two_ba]
xl = ['L', 'edges', 'R', 'L', 'edges', 'R']
bd = dict(color=cl)
gr = [[0, 1, 2], [3, 4, 5]]
fs = (4, 4)
# d-primes
dprime_barplot(dprimes[conds], grouping=gr, xlab=xl, group_names=gn,
               err_bars='se', filename=op.join(andir, 'DP_PhonSemAdjLRC.pdf'),
               bar_kwargs=bd, line_kwargs=ld, err_kwargs=ed,
               ylim=(-1, 5), figsize=(5, 6))
# reaction times
fn = 'RT_PhonSemSepLRC.pdf'
#pv = rt_ttests['p'][['ctrl_adj_sep', 'test_adj_sep']].tolist()
dprime_barplot(rtimes[conds], grouping=gr, xlab=xl, group_names=gn,
               err_bars='se', filename=op.join(andir, fn), figsize=fs,
               bar_kwargs=bd, line_kwargs=ld, err_kwargs=ed,
               ylab='reaction time (s)', ylim=(0.4, 0.9))  #, bracket_kwargs=bk,
               #brackets=gr, bracket_text=format_pval(pv))
# hit rate
fn = 'HR_PhonSemSepLRC.pdf'
#pv = hr_ttests['p'][['ctrl_adj_sep', 'test_adj_sep']].tolist()
dprime_barplot(hrate[conds], grouping=gr, xlab=xl, group_names=gn,
               err_bars='se', filename=op.join(andir, fn), figsize=fs,
               bar_kwargs=bd, line_kwargs=ld, err_kwargs=ed,
               ylab='Hit rate', ylim=(0, 1.1))  # , bracket_kwargs=bk,
               #brackets=gr, bracket_text=format_pval(pv))
# foil rate
fn = 'DR_PhonSemSepLRC.pdf'
#pv = dr_ttests['p'][['ctrl_adj_sep', 'test_adj_sep']].tolist()
dprime_barplot(drate[conds], grouping=gr, xlab=xl, group_names=gn,
               err_bars='se', filename=op.join(andir, fn), figsize=fs,
               bar_kwargs=bd, line_kwargs=ld, err_kwargs=ed,
               ylab='False alarm rate (foils)', ylim=(0, 0.3))  # ,
               #bracket_kwargs=bk, brackets=gr, bracket_text=format_pval(pv))
# false alarm rate
fn = 'FR_PhonSemSepLRC.pdf'
#pv = fr_ttests['p'][['ctrl_adj_sep', 'test_adj_sep']].tolist()
dprime_barplot(frate[conds], grouping=gr, xlab=xl, group_names=gn,
               err_bars='se', filename=op.join(andir, fn), figsize=fs,
               bar_kwargs=bd, line_kwargs=ld, err_kwargs=ed,
               ylab='False alarm rate (non-foils)', ylim=(0, 0.3))  # ,
               #bracket_kwargs=bk, brackets=gr, bracket_text=format_pval(pv))
# combined foils and false alarms
fn = 'DRFR_PhonSemSepLRC.pdf'
#pv = dr_ttests['p'][['ctrl_adj_sep', 'test_adj_sep']].tolist() + \
#    fr_ttests['p'][['ctrl_adj_sep', 'test_adj_sep']].tolist()
data = drate[conds].join(frate[conds], how='outer', lsuffix='_foils')
gr = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
xl = xl * 2
gnn = ['phon.', 'sem.'] * 2
ggnn = ['Foils', 'Non-foils']
gr2 = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]
dprime_barplot(data, grouping=gr, xlab=xl, group_names=gnn, gn2=ggnn, gr2=gr2,
               err_bars='se', filename=op.join(andir, fn), figsize=(5, 4),
               bar_kwargs=dict(color=cl), line_kwargs=ld, err_kwargs=ed,
               ylab='False alarm rate', ylim=(0, 0.3))  # ,
               #bracket_text=format_pval(pv), brackets=gr, bracket_kwargs=bk)
plt.close('all')

# # # # # # # # # # # # #
# categories (grouped)  #
# # # # # # # # # # # # #
conds = ['sel_' + x for x in cats] + ['div_' + x for x in cats]
cl = ['LightGray'] * 7 + ['DarkGray'] * 7
xl = ['food & drink' if x == 'fooddrink' else 'clothes' if x == 'clothing'
      else x for x in cats] * 2
gr = [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13]]
gn = ['selective', 'divided']
dprime_barplot(dprimes[conds], grouping=gr, xlab=xl, group_names=gn,
               err_bars='se', filename=op.join(andir, 'catsGrouped.pdf'),
               bar_kwargs=dict(color=cl), lines=False, err_kwargs=ed,
               ylim=(-1, 5), figsize=(7, 6))

# # # # # # # # # # # # #
# example trial diagram #
# # # # # # # # # # # # #
rcParams['font.size'] = 16
trialnum = 6
with open(op.join(andir, 'wordDurations.json')) as jd:
    word_durs = json.load(jd)

plt.figure(figsize=(18, 3))

# head with angle lines
q = plt.subplot2grid((1, 16), (0, 0), colspan=2)
_ = q.axis('off')
ybounds = (-0.4, 1.4)
_ = q.set_ybound(ybounds)
#_ = q.set_xbound((0, 1))
_ = q.set_aspect('equal', adjustable='box')
cx = 0.25
cy = 0.5
x = [cx + 0.375, cx + 0.75, cx + 0.75, cx + 0.375]
y = [cy + np.sqrt(3) * 3 / 8, cy + 3 / (8 * np.sqrt(3)),
     cy - 3 / (8 * np.sqrt(3)), cy - np.sqrt(3) * 3 / 8]
# lines
l60 = lines.Line2D([cx, x[0]], [cy, y[0]], axes=q, clip_on=False, color='k')
l15 = lines.Line2D([cx, x[1]], [cy, y[1]], axes=q, clip_on=False, color='k')
r15 = lines.Line2D([cx, x[2]], [cy, y[2]], axes=q, clip_on=False, color='k')
r60 = lines.Line2D([cx, x[3]], [cy, y[3]], axes=q, clip_on=False, color='k')
zord = []
for lin in [l60, l15, r15, r60]:
    _ = q.add_line(lin)
    zord.append(lin.get_zorder())
zord = np.max(zord)
# categories
cats = ['food & drink' if c == 'fooddrink' else 'clothes' if c == 'clothing'
        else c for c in all_data['cats'][trialnum]]
yval = ybounds[0] + np.diff(ybounds) * np.array([1./8, 3./8, 5./8, 7./8])
for idx, (cat, xy) in enumerate(zip(cats, zip(x, y))):
    _ = q.annotate(cat, (xy[0], yval[idx]), xytext=(10, 0),
                   textcoords='offset points', va='center', ha='left',
                   color='k', clip_on=False, family='sans-serif')
# head
head = patches.Ellipse(xy=(cx, cy), width=0.5, height=0.4,
                       ec='none', fc='Gray', axes=q, zorder=zord + 1)
lear = patches.Ellipse(xy=(0.22, 0.325), width=0.2, height=0.1, angle=30,
                       ec='none', fc='Gray', axes=q, zorder=zord + 1)
rear = patches.Ellipse(xy=(0.22, 0.675), width=0.2, height=0.1, angle=-30,
                       ec='none', fc='Gray', axes=q, zorder=zord + 1)
nose = patches.Ellipse(xy=(0.44, 0.5), width=0.25, height=0.125,
                       ec='none', fc='Gray', axes=q, zorder=zord + 1)
for pat in [head, lear, rear, nose]:
    _ = q.add_patch(pat)
plt.tight_layout()

# trial time course
p = plt.subplot2grid((1, 16), (0, 3), colspan=13)
xb = (-0.125, 12.5)
p.grid(True, axis='y', which='minor', color='0.7', linestyle='-')
p.grid(True, axis='x', which='minor', color='0.75', linestyle='-')
p.tick_params(length=0)
# highlight attended streams
atn = all_data['attn'][trialnum]
for idx, stream in enumerate(atn):
    if stream == 0:
        _ = p.fill_between(xb, 0.5 + idx, 1.5 + idx, where=None,
                           facecolor='0.9', color='none', alpha=1)
    else:
        _ = p.fill_between(xb, 0.5 + idx, 1.5 + idx, where=None,
                           facecolor='White', color='none', alpha=1)
_ = p.set_xticks(range(13))
_ = p.set_xticks([z / 4.0 for z in range(49)], minor=True)
_ = p.set_yticks([z + 0.5 for z in range(4)], minor=True)
_ = p.set_yticks(range(1, 5, 1))
_ = p.set_yticklabels([u'60° right', u'15° right', u'15° left', u'60° left'])
_ = p.set_xlabel('time (sec)')

w = list(chain.from_iterable(all_data['words'][trialnum]))
t = all_data['targ_words'][trialnum]
d = all_data['dist_words'][trialnum]
x = list(chain.from_iterable(all_data['times'][trialnum]))
y = [1] * 12 + [2] * 12 + [3] * 12 + [4] * 12
for word, xy in zip(w, zip(x, y)):
    if word in t:
        c = 'Green'
    elif word in d:
        c = 'Red'
    else:
        c = 'DimGray'
    # boxes for word durations, word labels
    _ = p.fill_between([xy[0], xy[0] + word_durs[word]], xy[1] - 0.25,
                       xy[1] + 0.25, where=None, facecolor=c, edgecolor='face',
                       alpha=0.2)
    _ = p.annotate(word, xy, textcoords='offset points', xytext=(2, 0),
                   va='center', ha='left', color=c)

_ = p.set_xbound(xb)
_ = p.set_ybound(0.5, 4.5)
plt.tight_layout()
plt.draw()
plt.savefig(op.join(andir, 'trialFig.pdf'), format='pdf', transparent=True)
plt.close('all')

# comparison of two different d-prime calculation methods
"""
plt.ion()
plt.figure()
conds = ['ctrl_sel', 'ctrl_div', 'test_sel', 'test_div']
p = plt.subplot(1, 4, 1)
_ = p.bar(range(len(dprimes[conds].columns)), dprimes[conds].mean())
q = plt.subplot(1, 4, 2, sharey=p)
_ = q.bar(range(len(dprime2[conds].columns)), dprime2[conds].mean())

conds = ['ctrl_adj', 'ctrl_sep', 'test_adj', 'test_sep']
q = plt.subplot(1, 4, 3, sharey=p)
_ = q.bar(range(len(dprimes[conds].columns)), dprimes[conds].mean())
q = plt.subplot(1, 4, 4, sharey=p)
_ = q.bar(range(len(dprime2[conds].columns)), dprime2[conds].mean())
plt.draw()
plt.show()
"""

# histogram of hits, foil alarms, and false alarms
"""
plt.figure()
p = plt.subplot(1, 3, 1)
p.hist(all_data['hits'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5])
q = plt.subplot(1, 3, 2, sharey=p)
_ = q.hist(all_data['d_alarm'], bins=[-0.5, 0.5, 1.5, 2.5])
r = plt.subplot(1, 3, 3, sharey=p)
_ = r.hist(all_data['f_alarm'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5])
plt.draw()
plt.show()
"""

# boxplot of hit, foil click, and false alarm rates
"""
plt.figure()
p = plt.subplot(1, 1, 1)
data = [all_data['hits'], all_data['d_alarm'], all_data['f_alarm']]
#data = [all_data['h_rate'], all_data['d_rate'], all_data['f_rate']]
bars = [np.mean(all_data['h_rate']), np.mean(all_data['d_rate']),
        np.mean(all_data['f_rate'])]
errs = [np.std(all_data['h_rate']), np.std(all_data['d_rate']),
        np.std(all_data['f_rate'])]
_ = p.boxplot(data, notch=True, sym='+', vert=True, whis=1.5)
_ = p.set_ylim(-0.5, 4)
plt.draw()
plt.show()
"""

# other unused plots
"""
# categories (paired)
conds = list(chain.from_iterable([['sel_' + x, 'div_' + x] for x in cats]))
cl = ['LightGray', 'DarkGray']
xl = ['sel', 'div'] * 7
gr = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]]
gn = ['food & drink' if x == 'fooddrink' else x for x in cats]
dprime_barplot(dprimes[conds], grouping=gr, xlab=xl, group_names=gn,
               err_bars='se', filename=op.join(andir, 'catsPaired.pdf'),
               bar_kwargs=dict(color=cl), line_kwargs=ld, err_kwargs=ed)
"""
"""
# semantic/phonetic; edge effects with adjacent streams
conds = ['ctrl_adj_l', 'ctrl_adj_c', 'ctrl_adj_r',
         'test_adj_l', 'test_adj_c', 'test_adj_r']
cl = ['WhiteSmoke', 'LightGray', 'DarkGray',
      'WhiteSmoke', 'LightGray', 'DarkGray']
xl = ['left', 'midline', 'right', 'left', 'midline', 'right']
gr = [[0, 1, 2], [3, 4, 5]]
dprime_barplot(dprimes[conds], grouping=gr, xlab=xl, group_names=gn,
               err_bars='se', filename=op.join(andir, 'PhonSemAdjEdge.pdf'),
               bar_kwargs=dict(color=cl), line_kwargs=ld, err_kwargs=ed)

# semantic/phonetic; edge effects with separated streams
conds = ['ctrl_sep_l', 'ctrl_sep_e', 'ctrl_sep_r',
         'test_sep_l', 'test_sep_e', 'test_sep_r']
cl = ['WhiteSmoke', 'LightGray', 'DarkGray',
      'WhiteSmoke', 'LightGray', 'DarkGray']
xl = ['left', 'edges', 'right', 'left', 'edges', 'right']
gr = [[0, 1, 2], [3, 4, 5]]
dprime_barplot(dprimes[conds], grouping=gr, xlab=xl, group_names=gn,
               err_bars='se', filename=op.join(andir, 'PhonSemSepEdge.pdf'),
               bar_kwargs=dict(color=cl), line_kwargs=ld, err_kwargs=ed)
"""
