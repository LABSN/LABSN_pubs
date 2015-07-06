# # # # # # # # # # # # # # # # #
# PRAAT SCRIPT "MONOTONIZE 107Hz"
#
# VERSION 0.2 (2014 04 03)
#
# AUTHOR: Daniel McCloy (drmccloy@uw.edu)
# LICENSE: BSD 3-clause (http://opensource.org/licenses/BSD-3-Clause)
# # # # # # # # # # # # # # # # #

# COLLECT ALL THE USER INPUT
form Neutralize Prosody: Select directories & starting parameters
	sentence source_dir manipulationObjects/
	sentence output_dir monotonizedWords/
	positive desired_pitch 107 (=Hz)
endform

if right$(source_dir$, 1) <> "/"
    source_dir$ = source_dir$ + "/"
endif

if right$(output_dir$, 1) <> "/"
    output_dir$ = output_dir$ + "/"
endif

# READ IN FILES
flist = Create Strings as file list... flist 'source_dir$'*.Manipulation
n = Get number of strings

for i from 1 to n
	# READ IN EACH STIMULUS
	select Strings flist
	curFile$ = Get string... 'i'
	curManip = Read from file... 'source_dir$''curFile$'
	curName$ = selected$ ("Manipulation", 1)
	pt = do ("Extract pitch tier")
	do ("Formula...", "'desired_pitch'")
	selectObject (curManip)
	plusObject (pt)
	do ("Replace pitch tier")
	select curManip
	wav = do ("Get resynthesis (overlap-add)")
	do ("Save as WAV file...", "'output_dir$''curName$'.wav")
	selectObject (curManip)
	plusObject (pt)
	plusObject (wav)
	Remove
endfor

selectObject (flist)
Remove
