# -*- coding: utf-8 -*-
"""
====================================================
Experiment 'Divided Attention & Semantic Categories'
====================================================

This experiment plays spatially-distributed word streams and asks listeners
for semantic-class judgments about the words in specific target streams.
"""
# Author: Dan McCloy <drmccloy@uw.edu>
#
# License: BSD (3-clause)

import json
import socket
import numpy as np
#from glob import glob
from os import path as op
from os import chdir
from scipy import io as sio
#from scipy.io import wavfile as wav
from itertools import chain
from expyfun import ExperimentController
from expyfun._utils import set_log_level
set_log_level('DEBUG')


def clean_mat(md, flatten=False):
    """Remove unwanted dict entries in MAT files & optionally flatten values
    """
    for key in ('__globals__', '__header__', '__version__'):
        if key in md:
            del md[key]
    if flatten:
        md = {k: list(chain.from_iterable(v)) for k, v in md.iteritems()}
    return md


# # # # # # # # #
# configuration #
# # # # # # # # #
# random seed
rng = np.random.RandomState(0)
cont_btn = 8
cont_btn_label = 'Next'
resp_btn = 1
min_rt = 0.1
max_rt = 1.25
pretrial_wait = 1.0
feedback_dur = 1.5
isi = 0.2
std_args = ['divAttnSemantic']
std_kwargs = dict(screen_num=0, window_size=[800, 600], full_screen=True,
                  stim_db=65, noise_db=40,  # participant='foo', session='000',
                  stim_rms=0.02, check_rms=None, suppress_resamp=True,
                  stim_fs=24414.0625)  # 44100.0

# # # # # # # # # # # # # # # # # # # # # # #
# detect system, set experiment root folder #
# # # # # # # # # # # # # # # # # # # # # # #
hn = socket.gethostname()
if hn == 'moctezuma':  # dan's LABSN desktop
    root = '/home/dan/Documents/experiments/drmccloy/divAttnSemantic'
elif hn == 'boulboul':  # dan's laptop
    root = ('/home/dan/Documents/academics/research/auditoryAttention/'
            'drmccloy/divAttnSemantic')
elif hn == 'carver':  # booth laptop
    root = ('C:\Users\labsner\Documents\python\drmccloy\divAttnSemantic')

# # # # # # # # # #
# load variables  #
# # # # # # # # # #
vars_dir = op.join(root, 'variables')
chdir(vars_dir)
mat_dict = clean_mat(sio.loadmat('divAttnSemantic.mat'))
tr_times = clean_mat(sio.loadmat('test_times.mat'))
tr_codes = clean_mat(sio.loadmat('test_codes.mat'))
tr_locs = clean_mat(sio.loadmat('test_locs.mat'), flatten=True)
tr_cats = clean_mat(sio.loadmat('test_cats.mat'))
ctrl_tr_times = clean_mat(sio.loadmat('ctrl_times.mat'))
ctrl_tr_codes = clean_mat(sio.loadmat('ctrl_codes.mat'))
ctrl_tr_locs = clean_mat(sio.loadmat('ctrl_locs.mat'), flatten=True)
ctrl_tr_cats = clean_mat(sio.loadmat('ctrl_cats.mat'))
chdir(root)
codes = mat_dict['codes']
words = mat_dict['words']
times = mat_dict['times']
cats = mat_dict['cats']
attn = mat_dict['attn']
ctrl_codes = mat_dict['ctrl_codes']
ctrl_words = mat_dict['ctrl_words']
ctrl_times = mat_dict['ctrl_times']
ctrl_cats = mat_dict['ctrl_cats']
ctrl_attn = mat_dict['ctrl_attn']
# JSON dict (mostly scalars)
with open('variables/divAttnSemantic.json') as jd:
    json_dict = json.load(jd)
    isi = json_dict['isi']
    fs = json_dict['fs']
    blocks = json_dict['blocks']
    trials = json_dict['trials']
    streams = json_dict['streams']
    trials_per_block = json_dict['tpb']
    ctrl_trials_per_block = json_dict['ctpb']
del mat_dict, json_dict
tr_attn = {k: tuple(1 if x in v else 0 for x in xrange(streams))
           for k, v in tr_locs.iteritems()}
ctrl_tr_attn = {k: tuple(1 if x in v else 0 for x in xrange(streams))
                for k, v in ctrl_tr_locs.iteritems()}
display_block = -1

# # # # # # # # # # #
# organize stimuli  #
# # # # # # # # # # #
stim_dict = dict()
stim_dict['ctrl_train_single'] = 'ctrl_single.mat'
stim_dict['ctrl_train_selective'] = 'ctrl_select.mat'
stim_dict['ctrl_train_full'] = 'ctrl_full.mat'
stim_dict['ctrl_train_dual'] = 'ctrl_dual.mat'
stim_dict['ctrl_train_divide'] = 'ctrl_divide.mat'
stim_dict['ctrl_trials_one'] = 'ctrl_block1.mat'
stim_dict['ctrl_trials_two'] = 'ctrl_block2.mat'
stim_dict['ctrl_trials_three'] = 'ctrl_block3.mat'
stim_dict['ctrl_trials_four'] = 'ctrl_block4.mat'
stim_dict['train_single'] = 'test_single.mat'
stim_dict['train_selective'] = 'test_select.mat'
stim_dict['train_full'] = 'test_full.mat'
stim_dict['train_dual'] = 'test_dual.mat'
stim_dict['train_divided'] = 'test_divide.mat'
stim_dict['trials_one'] = 'test_block1.mat'
stim_dict['trials_two'] = 'test_block2.mat'
stim_dict['trials_three'] = 'test_block3.mat'
stim_dict['trials_four'] = 'test_block4.mat'

cat_dict = dict()
cat_dict['ctrl_train_single'] = ctrl_tr_cats
cat_dict['ctrl_train_selective'] = ctrl_tr_cats
cat_dict['ctrl_train_full'] = ctrl_tr_cats
cat_dict['ctrl_train_dual'] = ctrl_tr_cats
cat_dict['ctrl_train_divide'] = ctrl_tr_cats
cat_dict['ctrl_trials_one'] = ctrl_cats
cat_dict['ctrl_trials_two'] = ctrl_cats
cat_dict['ctrl_trials_three'] = ctrl_cats
cat_dict['ctrl_trials_four'] = ctrl_cats
cat_dict['train_single'] = tr_cats
cat_dict['train_selective'] = tr_cats
cat_dict['train_full'] = tr_cats
cat_dict['train_dual'] = tr_cats
cat_dict['train_divided'] = tr_cats
cat_dict['trials_one'] = cats
cat_dict['trials_two'] = cats
cat_dict['trials_three'] = cats
cat_dict['trials_four'] = cats

attn_dict = dict()
attn_dict['ctrl_train_single'] = ctrl_tr_attn
attn_dict['ctrl_train_selective'] = ctrl_tr_attn
attn_dict['ctrl_train_full'] = ctrl_tr_attn
attn_dict['ctrl_train_dual'] = ctrl_tr_attn
attn_dict['ctrl_train_divide'] = ctrl_tr_attn
attn_dict['ctrl_trials_one'] = ctrl_attn
attn_dict['ctrl_trials_two'] = ctrl_attn
attn_dict['ctrl_trials_three'] = ctrl_attn
attn_dict['ctrl_trials_four'] = ctrl_attn
attn_dict['train_single'] = tr_attn
attn_dict['train_selective'] = tr_attn
attn_dict['train_full'] = tr_attn
attn_dict['train_dual'] = tr_attn
attn_dict['train_divided'] = tr_attn
attn_dict['trials_one'] = attn
attn_dict['trials_two'] = attn
attn_dict['trials_three'] = attn
attn_dict['trials_four'] = attn

code_dict = dict()
code_dict['ctrl_train_single'] = ctrl_tr_codes
code_dict['ctrl_train_selective'] = ctrl_tr_codes
code_dict['ctrl_train_full'] = ctrl_tr_codes
code_dict['ctrl_train_dual'] = ctrl_tr_codes
code_dict['ctrl_train_divide'] = ctrl_tr_codes
code_dict['ctrl_trials_one'] = ctrl_codes
code_dict['ctrl_trials_two'] = ctrl_codes
code_dict['ctrl_trials_three'] = ctrl_codes
code_dict['ctrl_trials_four'] = ctrl_codes
code_dict['train_single'] = tr_codes
code_dict['train_selective'] = tr_codes
code_dict['train_full'] = tr_codes
code_dict['train_dual'] = tr_codes
code_dict['train_divided'] = tr_codes
code_dict['trials_one'] = codes
code_dict['trials_two'] = codes
code_dict['trials_three'] = codes
code_dict['trials_four'] = codes

time_dict = dict()
time_dict['ctrl_train_single'] = ctrl_tr_times
time_dict['ctrl_train_selective'] = ctrl_tr_times
time_dict['ctrl_train_full'] = ctrl_tr_times
time_dict['ctrl_train_dual'] = ctrl_tr_times
time_dict['ctrl_train_divide'] = ctrl_tr_times
time_dict['ctrl_trials_one'] = ctrl_times
time_dict['ctrl_trials_two'] = ctrl_times
time_dict['ctrl_trials_three'] = ctrl_times
time_dict['ctrl_trials_four'] = ctrl_times
time_dict['train_single'] = tr_times
time_dict['train_selective'] = tr_times
time_dict['train_full'] = tr_times
time_dict['train_dual'] = tr_times
time_dict['train_divided'] = tr_times
time_dict['trials_one'] = times
time_dict['trials_two'] = times
time_dict['trials_three'] = times
time_dict['trials_four'] = times

text_dict = dict()
text_dict_args = [cont_btn_label, resp_btn, streams, display_block, 2 * blocks]
text_dict['init'] = [('This experiment involves reading words on screen, '
                      'listening to words through headphones, and pushing '
                      'buttons to respond. To make the task slightly harder, '
                      'there will be background noise. Throughout the '
                      'experiment, press the button labeled "{0}" when you '
                      'understand the instructions and are ready to '
                      'continue.'),
                     ('In this part of the experiment, you will hear English '
                      'words coming from {2} different spatial directions. '
                      'There will also be {2} words on screen, in positions '
                      'that correspond to the spatial sources of the audio '
                      'streams.')]
text_dict['ctrl_train_single'] = [('Most of the words you hear will be the '
                                   'same as the corresponding words on screen.'
                                   ' Your job is to press the "{1}" button as '
                                   'quickly as you can when a word you hear '
                                   'does NOT match the word on screen. Let\'s '
                                   'practice a few times with just one stream.'
                                   '\n\nPress "{0}" to start the practice.')]
text_dict['ctrl_train_selective'] = ('Now let\'s practice ignoring streams. '
                                     'Pay attention to the stream whose '
                                     'word is colored green, and ignore '
                                     'the one whose word is grey. Push the '
                                     'response button as quickly as you can '
                                     'only when you hear a word in the '
                                     'attended stream that doesn\'t match '
                                     'the corresponding word on screen.'
                                     '\n\nPress "{0}" to start the practice.')
text_dict['ctrl_train_full'] = ('Good work. This time there will be four '
                                'streams. Remember, ignore the streams whose '
                                'words are grey, and respond as fast as you '
                                'can when a word in the attended stream '
                                'doesn\'t match the green word.'
                                '\n\nPress "{0}" to start the practice.')
text_dict['ctrl_train_dual'] = ('Well done. Sometimes you will be asked to '
                                'attend to two word streams at the same time. '
                                'Let\'s practice that now with just two '
                                'streams.\n\nPress "{0}" to start the '
                                'practice.')
text_dict['ctrl_train_divide'] = ('Now for the last step in the training: '
                                  'four streams total, attending to two '
                                  'streams while ignoring two other streams.'
                                  '\n\nPress "{0}" to start the practice.')
text_dict['ctrl_trials_one'] = ('Good job! you passed the training. From this '
                                'point on, all trials will have {2} spatial '
                                'streams. Like the training, the streams to '
                                'ignore will be grey text; the streams to '
                                'attend will be green text. Each block will '
                                'last about 4 minutes. Take a short break now '
                                'if you want, then when you\'re ready to start'
                                ' the next block of trials, press "{0}".')
text_dict['ctrl_trials_two'] = ('You finished block {3} out of {4}. Feel free '
                                'to take a break now and leave the booth if '
                                'you like.\n\nPress "{0}" when you are ready '
                                'to resume.')
text_dict['ctrl_trials_three'] = text_dict['ctrl_trials_two']
text_dict['ctrl_trials_four'] = text_dict['ctrl_trials_two']
text_dict['train_single'] = [('The words on screen will be names of categories'
                              ', and most of the words you hear in a spatial '
                              'stream will belong to the corresponding '
                              'category seen on screen (for example: types of '
                              'furniture, animal names, weather terms, etc).'),
                             ('Your job is to press the "{1}" button as '
                              'quickly as you can when you hear a word that '
                              'does NOT match the expected category. Let\'s '
                              'practice a few times with just one stream.'
                              '\n\nPress "{0}" to start the practice.')]
text_dict['train_selective'] = ('Now let\'s practice ignoring streams. Pay '
                                'attention to the stream whose category is '
                                'colored green, and ignore the one whose '
                                'category is grey. Push the response button '
                                'as quickly as you can only when there is a '
                                'word in the attended stream that doesn\'t '
                                'match the corresponding category name on '
                                'screen.'
                                '\n\nPress "{0}" to start the practice.')
text_dict['train_full'] = ('Good work. This time there will be four streams. '
                           'Remember, ignore the streams whose categories are '
                           'grey, and respond as fast as you can when a word '
                           'in the attended stream doesn\'t belong to the '
                           'category in green.'
                           '\n\nPress "{0}" to start the practice.')
text_dict['train_dual'] = ('Well done. Sometimes you will be asked to attend '
                           'to two word streams at the same time. Let\'s '
                           'practice that now with just two streams.'
                           '\n\nPress "{0}" to start the practice.')
text_dict['train_divided'] = ('Now for the last step in the training: '
                              'four streams total, attending to two '
                              'streams while ignoring two other streams.'
                              '\n\nPress "{0}" to start the practice.')
text_dict['trials_one'] = ('Good job! you passed the training. From this point'
                           ' on, all trials will have {2} spatial streams. '
                           'Like the training, the streams to ignore will be '
                           'grey text; the streams to attend to will be green '
                           'text. Each block will last about 8 minutes. Take '
                           'a short break now if you want, then when you\'re '
                           'ready to start the next block of trials, press '
                           '"{0}".')
text_dict['trials_two'] = text_dict['ctrl_trials_two']
text_dict['trials_three'] = text_dict['ctrl_trials_two']
text_dict['trials_four'] = text_dict['ctrl_trials_two']
text_dict['midpoint'] = [('You are halfway done. Please take a break now and '
                          'walk around outside the booth for a bit. \n\nPress '
                          '"{0}" when you have returned and are ready to '
                          'continue.'),
                         ('For the second half of the experiment, the task '
                          'will change slightly, so there is another round '
                          'of instructions and training to do first.'
                          '\n\nPress "{0}" to begin training.'),
                         ('In this part of the experiment, you will hear '
                          'English words coming from {2} different spatial '
                          'directions like before. Just like before, there '
                          'will also be {2} words on screen, in positions '
                          'that correspond to the spatial sources of the '
                          'audio streams.')]
text_dict['end'] = ('All done! Thanks for participating. You can leave the '
                    'sound booth now.')

ctrl_train = ('ctrl_train_single', 'ctrl_train_selective', 'ctrl_train_full',
              'ctrl_train_dual', 'ctrl_train_divide',)
ctrl_test = ('ctrl_trials_one', 'ctrl_trials_two', 'ctrl_trials_three',
             'ctrl_trials_four')
train = ('train_single', 'train_selective', 'train_full', 'train_dual',
         'train_divided')
test = ('trials_one', 'trials_two', 'trials_three', 'trials_four')
training_phases = ctrl_train + train
xpos = [-0.75, -0.25, 0.25, 0.75]
ypos = [-0.25, 0.25, 0.25, -0.25]

# # # # # # # # # #
# run experiment  #
# # # # # # # # # #
with ExperimentController(*std_args, **std_kwargs) as ec:
    # counterbalance experiment order across subjects
    if int(ec._exp_info['session']) % 2 == 0:
        order = ctrl_train + ctrl_test + train + test
    else:
        order = train + test + ctrl_train + ctrl_test

    text_dict['init'] = [x.format(*text_dict_args) for x in text_dict['init']]
    ec.screen_prompt(text_dict['init'], live_keys=[cont_btn])
    # counters for tracking trial number across blocks
    test_t = 0
    ctrl_t = 0

    for cur_block, phase in enumerate(order):
        if cur_block < 9:
            display_block = cur_block - 5
        else:
            display_block = cur_block - 10
        text_dict_args = [cont_btn_label, resp_btn, streams, display_block,
                          2 * blocks]
        if cur_block == 9:
            text_dict['midpoint'] = [x.format(*text_dict_args)
                                     for x in text_dict['midpoint']]
            ec.screen_prompt(text_dict['midpoint'], live_keys=[cont_btn])
        # load parameters for this phase
        ec.screen_text('loading...')
        stims = clean_mat(sio.loadmat(op.join(vars_dir, stim_dict[phase])))
        phase_keys = sorted(stims.keys())
        if phase in ('ctrl_train_single', 'train_single', 'train_selective',
                     'ctrl_train_selective'):
            # randomize the easiest conditions to alleviate boredom
            rng.shuffle(phase_keys)
        if phase in training_phases:
            phase_stims = [stims[x] for x in phase_keys]
            phase_cats = [cat_dict[phase][x].tolist() for x in phase_keys]
            phase_attn = [attn_dict[phase][x] for x in phase_keys]
            phase_code = [code_dict[phase][x] for x in phase_keys]
            phase_time = [time_dict[phase][x] for x in phase_keys]
        else:
            phase_stims = [y for y in [stims[x] for x in phase_keys][0]]
            phase_cats = cat_dict[phase].tolist()
            phase_attn = attn_dict[phase].tolist()
            phase_code = code_dict[phase].tolist()
            phase_time = time_dict[phase].tolist()
        phase_trials = len(phase_stims)
        phase_text = text_dict[phase]

        # get ready to run
        if phase in training_phases:
            current = False
            previous = False
        t = 0
        if isinstance(phase_text, list):
            phase_text = [x.format(*text_dict_args) for x in phase_text]
            ec.screen_prompt(phase_text, live_keys=[cont_btn])
        else:
            ec.screen_prompt(phase_text.format(*text_dict_args),
                             live_keys=[cont_btn])
        if cur_block == 0:
            ec.start_noise()
        ec.write_data_line('phase', phase)
        ec.screen_prompt('Here we go!', pretrial_wait, live_keys=[])

        while t < phase_trials:
            end_wait = ec.current_time + pretrial_wait
            if phase in training_phases:
                counter = t
            elif phase in test:
                counter = test_t
            elif phase in ctrl_test:
                counter = ctrl_t
            ec.write_data_line('trial', counter)
            # draw categories on screen
            current_cats = tuple(['food & drink' if x.strip() == 'fooddrink'
                                  else x.strip() for x in phase_cats[counter]])
            current_colors = np.where(phase_attn[counter], 'Lime',
                                      'LightGray').tolist()
            txt_obj = []
            for n, cat in enumerate(current_cats):
                txt_obj.append(ec.screen_text(cat, pos=[xpos[n], ypos[n]],
                               color=current_colors[n]))
                txt_obj[-1].autoDraw = True
            #ec.flip()
            rt = []   # reaction time to targets
            drt = []  # reaction time to distractors
            fa = []   # false alarms - pressed too late or non-targ non-dist
            current_stim = phase_stims[t]
            current_dur = len(current_stim) / float(ec.fs)
            # play stim
            ec.load_buffer(current_stim)
            ec.wait_until(end_wait)
            ec.flip_and_play()
            # handle user responses
            presses = ec.wait_for_presses(current_dur + max_rt, min_rt,
                                          [resp_btn], True)
            ec.stop()
            p_times = [x for _, x in presses]
            t_times = sorted(np.ma.masked_array(phase_time[counter], np.array(
                                                phase_code[counter]) != 1)[
                             np.where(phase_attn[counter])].compressed())
            if any([x < 0 for x in t_times]):
                raise ValueError('negative targ time = wrong stream chosen')
            d_times = sorted(np.ma.masked_array(phase_time[counter], np.array(
                                                phase_code[counter]) != 1)[
                             np.where(np.logical_not(phase_attn[counter])
                                      )].compressed())
            for t_time in t_times:
                not_early = np.where(t_time < np.array(p_times) - min_rt)[0]
                not_late = np.where(t_time > np.array(p_times) - max_rt)[0]
                viable = set(not_early).intersection(set(not_late))
                if len(viable):
                    p_index = sorted(list(viable))[0]
                    p_time = p_times.pop(p_index)
                    rt.append(p_time - t_time)
                else:
                    rt.append(-1)
            for d_time in d_times:
                not_early = np.where(d_time < np.array(p_times) - min_rt)[0]
                not_late = np.where(d_time > np.array(p_times) - max_rt)[0]
                viable = set(not_early).intersection(set(not_late))
                if len(viable):
                    p_index = sorted(list(viable))[0]
                    p_time = p_times.pop(p_index)
                    drt.append(p_time - d_time)
                else:
                    drt.append(-1)
            fa.extend(p_times)

            # clear screen
            for obj in txt_obj:
                obj.setAutoDraw(False)

            if phase in training_phases:
                # give feedback
                n_targs = len(t_times)
                correct = len([x for x in rt if x > 0])
                mean_rt = np.mean([x for x in rt if x > 0] +
                                  [x for x in drt if x > 0])
                f_alarm = len(fa) + len([x for x in drt if x > 0])
                if f_alarm > 0:
                    feedback = ('{} of {} targets correct.\n'
                                '{} presses incorrect or too slow.\n'
                                'Mean reaction time {:.2f} sec.'
                                ''.format(correct, n_targs, f_alarm, mean_rt))
                elif correct == 0:
                    feedback = ('There were {} targets but you didn\'t push '
                                'the response button.'.format(n_targs))
                else:
                    feedback = ('{} of {} targets correct.\n'
                                'Mean reaction time {:.2f} sec.'
                                ''.format(correct, n_targs, mean_rt))
                if f_alarm > 0:
                    ec.screen_prompt(feedback, feedback_dur + 0.5)
                else:
                    ec.screen_prompt(feedback, feedback_dur)
                ec.wait_secs(pretrial_wait)
                # check if trained: can miss 0-1 targs & have 0-1 false alarms
                previous = current
                current = correct > n_targs - 2 and len(fa) < 2
                #if all([previous, current]):
                if t > 1 and all([previous, current]):
                    break
            else:
                # write out data
                ec.write_data_line('target_times', t_times)
                ec.write_data_line('distractor_times', d_times)
                ec.write_data_line('press_times', [x for _, x in presses])
                ec.write_data_line('target_RTs', rt)
                ec.write_data_line('distractor_RTs', drt)
                ec.write_data_line('false_alarm_times', fa)

            # iterate
            if phase in training_phases and t == range(phase_trials)[-1]:
                t = 0
            else:
                t = t + 1
            if phase in test:
                test_t = test_t + 1
            elif phase in ctrl_test:
                ctrl_t = ctrl_t + 1
    # finished!
    ec.screen_prompt(text_dict['end'], max_wait=6.0, live_keys=[])
