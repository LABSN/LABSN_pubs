#! /usr/bin/praat

# # # # # # # # # # # # # # # # # # # # # # # #
# PRAAT SCRIPT "SEGMENT RAW WORDLIST RECORDING"
# This script opens a sound file and a TextGrid and excises portions of the
# sound corresponding to non-empty interval labels. Includes support for 
# excluding labels that begin with a particular substring.
#
# VERSION 0.1 (2014 03 17)
#
# AUTHOR: Daniel McCloy (drmccloy@uw.edu)
# LICENSE: BSD 3-clause (http://opensource.org/licenses/BSD-3-Clause)
# # # # # # # # # # # # # # # # # # # # # # # #

form Segment raw wordlist recording
    sentence sound_file ./rawRecordings/dan-recording3.wav
    sentence TextGrid_file ./rawRecordings/dan-recording3.TextGrid
    positive tier_number 1
    word exclusion_prefix X_
    sentence output_directory ./excisedWords
    sentence output_prefix 
    sentence output_postfix 
endform

if right$(output_directory$, 1) <> "/"
    output_directory$ = output_directory$ + "/"
endif

au = Read from file: sound_file$
tg = Read from file: textGrid_file$
num_intv = Get number of intervals: tier_number

for intv to num_intv
    selectObject: tg
    intv_text$ = Get label of interval: tier_number, intv
    if intv_text$ <> ""
        if left$(intv_text$, length(exclusion_prefix$)) <> exclusion_prefix$
            sta = Get start point: tier_number, intv
            end = Get end point: tier_number, intv
            selectObject: au
            wau = Extract part: sta, end, "rectangular", 1, "no"
            filename$ = output_directory$ + output_prefix$ + intv_text$ + output_postfix$ + ".wav"
            Save as WAV file: filename$
            Remove
        else
            appendInfoLine: "skipping " + right$(intv_text$, length(intv_text$) - length(exclusion_prefix$))
        endif
    endif
endfor
selectObject: au
plusObject: tg
Remove
