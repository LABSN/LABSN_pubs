# -*- coding: utf-8 -*-
"""
===================================
Script 'Categorization training GUI'
===================================

This script makes a GUI for categorizing words.
"""
# Author: Dan McCloy <drmccloy@uw.edu>
#
# License: BSD (3-clause)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as mpp

plt.ion()

# random number generator
rng = np.random.RandomState(0)

# semantic categories
fish = ['eel', 'bass', 'cod']
birds = ['hawk', 'duck', 'goose']
fruit = ['lime', 'fig', 'grape']
drinks = ['wine', 'juice', 'tea']

allwords = fish + birds + fruit + drinks
allwords = sorted(allwords)
cats = dict(fish=fish, birds=birds, fruit=fruit, drinks=drinks)

# initialize figure
fig = plt.figure(1)
sfig = fig.add_subplot(111)
ax_range = [0, 8, -41.5, 1.5]
offset = 0.2
plt.axis(ax_range)
plt.axis('off')
fig.suptitle('Click once to select a word at left, then click on category '
             'box to move it to that category. If you make a mistake, click '
             'on a categorized word to move it back to the left.', fontsize=16)

# wordlist
word_x = {}
word_correct = {}
#word_handles = []
for idx, word in enumerate(allwords):
    wh = plt.text(0.2, 0 - idx, word, picker=True)
    #word_x[word] = (0, 0 - idx)
    #word_handles.append(wh)

# categories
rects = []
for idx, (name, cat) in enumerate(cats.iteritems()):
    if name == 'fooddrink':
        name = 'food & drink'
    plt.text(1 + idx, 1.5, name, fontsize=16)
    r = mpp.Rectangle((1 + idx - offset, -43), 0.9, 44, figure=fig,
                      color='white', fill=True, picker=True)
    rects.append(r)
    sfig.add_patch(r)

plt.plot()
redword = [None]
chosen = [None]
msg = [None]


def on_pick(event):
    picked = event.artist
    #print sorted(picked.properties().keys())
    if isinstance(picked, mpp.Rectangle):
        if redword[0] is not None:
            # put the red word here
            newx = picked.get_x()
            redword[0].set_x(newx + offset)
            word_x[redword[0].get_text()] = redword[0].get_position()[0]
            redword[0].set_color('k')
            redword[0] = None

    elif isinstance(picked, plt.Text):
        oldx, oldy = picked.get_position()
        if oldx > 0.5:  # they clicked an already categorized word
            if chosen[0] is None:
                picked.set_position((0.2, oldy))
                word_x[picked.get_text()] = 0.2
            else:  # there was a red word in the margin
                chosen[0] = None
        else:
            # they clicked a margin word
            if chosen[0] is not None:
                # there was another one already chosen
                chosen[0].set_color('k')
            chosen[0] = picked
            redword[0] = picked
            chosen[0].set_color('red')
    plt.draw()

    # check if done
    if len(word_x) == len(allwords):
        for key, val in word_x.iteritems():
            if key in fish:
                word_correct[key] = int(round(val)) == 1
            elif key in fruit:
                word_correct[key] = int(round(val)) == 2
            elif key in birds:
                word_correct[key] = int(round(val)) == 3
            elif key in drinks:
                word_correct[key] = int(round(val)) == 4
        if all(word_correct.values()):
            msg[0] = plt.text(3, -18, "success!", fontsize=60, color='blue')
            plt.draw()
        elif msg[0] is not None:
            msg[0].remove()
            msg[0] = None
            plt.draw()

pid = fig.canvas.mpl_connect('pick_event', on_pick)
