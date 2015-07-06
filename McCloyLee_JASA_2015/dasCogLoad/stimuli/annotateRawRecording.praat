#! /usr/bin/praat

# # # # # # # # # # # # # # # # # # # # # # # # #
# PRAAT SCRIPT "ANNOTATE RAW WORDLIST RECORDING"
# This script opens a text file and a TextGrid and uses the text file to fill
# in the TextGrid intervals, optionally skipping alternate intervals.
#
# VERSION 0.1 (2014 03 17)
#
# AUTHOR: Daniel McCloy (drmccloy@uw.edu)
# LICENSE: BSD 3-clause (http://opensource.org/licenses/BSD-3-Clause)
# # # # # # # # # # # # # # # # # # # # # # # # #

form Annotate raw wordlist recording
    sentence word_list ../stimSelection/finalWordList.txt
    sentence TextGrid_file ./rawRecordings/dan-recording3.TextGrid
    positive tier_number 1
    boolean skip_first 1
    boolean skip_alternate 1
    boolean reverse_order 1
    boolean save_and_clear = 1
endform

clearinfo
wl = Read Strings from raw text file: word_list$
tg = Read from file: textGrid_file$

selectObject: wl
num_words = Get number of strings

if skip_alternate
    skip_first -= 1  ; because skip_alternate already makes it start at #2
endif

for num to num_words
    if reverse_order
        wnum = num_words - num + 1
    else
        wnum = num
    endif
    selectObject: wl
    word$ = Get string: wnum
    selectObject: tg
    intv = num * (skip_alternate + 1) + skip_first
    intv_text$ = Get label of interval: tier_number, intv
    if intv_text$ = "X"
        word$ = "X_" + word$
    endif
    Set interval text: tier_number, intv, word$
endfor

if save_and_clear
    selectObject: tg
    Save as text file: textGrid_file$
    plusObject: wl
    Remove
endif
