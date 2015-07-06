# # # # # # # # # # # # # # # # # # # # # # # #
# PRAAT SCRIPT "SOUND TO MANIPULATION"
# This script semi-automates the process of determining appropriate pitch settings for analysis of sound files, and optionally uses those settings to create manipulation objects for later resynthesis.  The script cycles through a directory of sound files, opens them one at a time, displays the pitch contour over a narrowband spectrogram, and prompts the user to either: (1) accept the pitch settings, (2) adjust the pitch floor/ceiling and redraw, or (3) mark the file as unmeasurable, before continuing on to the next file.  An "advancedInterface" option is available for users who want full control over all pitch parameters during the process.  Filename, duration, and pitch settings are saved to a tab-delimited file.  The main purpose of this script is as a "feeder" for other scripts that are designed to perform fully automated tasks based on pitch analysis settings read in from the output of this script.  The advantage to this approach is that there is a permanent record of the pitch settings used in later analyses, and those settings are assured to be appropriate for each file (thereby minimizing or eliminating pitch halving/doubling errors).  The optional second step for each file (controlled by the "outputType" setting) is the creation of a manipulation object based on the pitch settings, and an opportunity for hand-correction of the detected pulses.
#
# FORM INSTRUCTIONS
# "logFile" should specify the FULL PATH of the log file.  The log file will store the sequential number, filename, duration, pitch floor and ceiling settings, and notes for each file.  "startingFileNum" allows you to pick up where you left off if you're processing a lot of files: just look at your log file from last time and enter the next number in sequence from the "number" column (if you do this, be sure to click "Append" when asked if you want to overwrite the existing log file).  The setting "defaultZoomDuration" is measured in seconds; if it is zero or negative, the whole duration of the file will be displayed.  If "carryover" is unchecked, then each new file analyzed will start out with the default pitch settings.  Otherwise, each new file (after the first one) will start out with the accepted settings from the preceding file (unless the preceding file was skipped, in which case the settings will revert to default).
#
# VERSION 0.5 (2013 03 16)
#
# CHANGELOG
# VERSION 0.5: added the manipulation object editing phase.
# VERSION 0.4: added control over (and logging of) all the advanced pitch settings, and option to export pitch and manipulation objects on the fly.  Hard-coded cross-correlation as pitch algorithm on lines 129, 132, 225, and 244
# VERSION 0.3: added default zoom duration setting
# VERSION 0.2: added autoplay functionality, ability to adjust octave jump cost file-by-file, and ability to set pitch viewing range.
#
# AUTHOR: DANIEL MCCLOY: (drmccloy@uw.edu)
# LICENSED UNDER THE GNU GENERAL PUBLIC LICENSE v3.0: http://www.gnu.org/licenses/gpl.html
# DEVELOPMENT OF THIS SCRIPT WAS FUNDED BY THE NATIONAL INSTITUTES OF HEALTH, GRANT # R01DC006014 TO PAMELA SOUZA
# # # # # # # # # # # # # # # # # # # # # # # #

# COLLECT ALL THE USER INPUT
form Pitch settings tool: Select directories & starting parameters
#	comment See the script's header for explanation of the form variables.
	sentence Sound_directory excisedWords/
	sentence Pitch_directory pitchObjects/
	sentence Manip_directory manipulationObjects
	sentence logFile soundToManipulation.log
#	sentence Sound_extension .wav
	optionmenu outputType: 3
		option only log pitch settings
		option save manipulation objects
		option edit manip. objs. before save
#	boolean logOnly 1
	integer startingFileNum 1
	real defaultZoomDuration 0
	integer viewRangeMin 1
	integer viewRangeMax 200
	integer defaultMinPitch 50 (=PSOLA minimum)
	integer defaultMaxPitch 200
#	comment Interface settings
	boolean advancedInterface 0
	boolean carryover 0
	optionmenu autoplay: 1
		option off
		option on first view
		option on each redraw
#	comment Advanced settings
#	optionmenu pitchAlgorithm: 2
#		option autocorrelation
#		option cross-correlation
	real defaultTimeStep 0 (=auto)
	integer defaultMaxCandidates 15
	boolean defaultHighAccuracy 1
	real defaultSilenceThreshold 0.03
	real defaultVoicingThreshold 0.45
	real defaultOctaveCost 0.01
	real defaultJumpCost 0.35
	real defaultVoicingCost 0.14
endform

showSettingsUI = 0

# BE FORGIVING IF THE USER FORGOT TRAILING PATH SLASHES OR LEADING FILE EXTENSION DOTS
call cleanPath 'sound_directory$'
soundDir$ = "'cleanPath.out$'"
call cleanPath 'pitch_directory$'
pitchDir$ = "'cleanPath.out$'"
call cleanPath 'manip_directory$'
manipDir$ = "'cleanPath.out$'"
#call cleanExtn 'sound_extension$'
#soundExt$ = "'cleanExtn.out$'"

# INITIATE THE OUTPUT FILE
if fileReadable (logFile$)
	beginPause ("The log file already exists!")
		comment ("The log file already exists!")
		comment ("You can overwrite the existing file, or append new data to the end of it.")
	overwrite_setting = endPause ("Append", "Overwrite", 1)
	if overwrite_setting = 2
		filedelete 'logFile$'
		call initializeOutfile
	endif
else
	# THERE IS NOTHING TO OVERWRITE, SO CREATE THE HEADER ROW FOR THE NEW OUTPUT FILE
	call initializeOutfile
endif

# MAKE A LIST OF ALL SOUND FILES IN THE FOLDER
Create Strings as file list... list 'soundDir$'*.wav
fileList = selected("Strings")
fileCount = Get number of strings

# INITIALIZE SOME VARIABLES
minPitch = defaultMinPitch
maxPitch = defaultMaxPitch
timeStep = defaultTimeStep
maxCandidates = defaultMaxCandidates
highAccuracy = defaultHighAccuracy
silenceThreshold = defaultSilenceThreshold
voicingThreshold = defaultVoicingThreshold
octaveCost = defaultOctaveCost
jumpCost = defaultJumpCost
voicingCost = defaultVoicingCost

# LOOP THROUGH THE LIST OF FILES...
for curFile from startingFileNum to fileCount

	# READ IN THE SOUND...
	select Strings list
	soundname$ = Get string... curFile
#	basename$ = mid$(soundname$,7,11)
	Read from file... 'soundDir$''soundname$'
#	Rename... 'basename$'
	filename$ = selected$ ("Sound", 1)
	totalDur = Get total duration

	# SHOW THE EDITOR WINDOW
	zoomStart = 0
	if defaultZoomDuration > 0
		zoomEnd = defaultZoomDuration
	else
		zoomEnd = totalDur
	endif
	select Sound 'filename$'
	View & Edit
	editor Sound 'filename$'

		# HIDE THE SPECTROGRAM & ANALYSES TO PREVENT ANNOYING FLICKERING
		Show analyses... no no no no no 10
		Zoom... zoomStart zoomEnd

		# SET ALL THE RELEVANT SETTINGS
		Spectrogram settings... 0 2500 0.025 50
		Advanced spectrogram settings... 1000 250 Fourier Gaussian yes 100 6 0
		if carryover = 0
			Pitch settings... defaultMinPitch defaultMaxPitch Hertz cross-correlation speckles
			Advanced pitch settings... viewRangeMin viewRangeMax defaultHighAccuracy defaultMaxCandidates defaultSilenceThreshold defaultVoicingThreshold defaultOctaveCost defaultJumpCost defaultVoicingCost
		else
			Pitch settings... minPitch maxPitch Hertz cross-correlation automatic
			Advanced pitch settings... viewRangeMin viewRangeMax highAccuracy maxCandidates silenceThreshold voicingThreshold octaveCost jumpCost voicingCost
		endif

		# DISPLAY NARROWBAND SPECTROGRAM AND PITCH (MAKING SURE "MAX ANALYSIS" IS LONG ENOUGH SO THE SPECTROGRAM ACTUALLY SHOWS UP)
#		Show analyses... yes yes no no no totalDur+1
		Show analyses... yes yes no no no zoomEnd-zoomStart+1
	endeditor

	# INITIALIZE SOME VARIABLES FOR THE PAUSE U.I.
	clicked = 1
	notes$ = ""
	if autoplay > 1
		firstview = 1
	else
		firstview = 0
	endif
	if carryover = 0 or maxPitch = 0  ;  THE maxPitch=0 CONDITION PREVENTS ERRORS WHEN carryover=1 AND THE PREV. FILE WAS SKIPPED
		minPitch = defaultMinPitch
		maxPitch = defaultMaxPitch
		timeStep = defaultTimeStep
		maxCandidates = defaultMaxCandidates
		highAccuracy = defaultHighAccuracy
		silenceThreshold = defaultSilenceThreshold
		voicingThreshold = defaultVoicingThreshold
		octaveCost = defaultOctaveCost
		voicingCost = defaultVoicingCost
		jumpCost = defaultJumpCost
	endif

	# SHOW A U.I. WITH PITCH SETTINGS.  KEEP SHOWING IT UNTIL THE USER ACCEPTS OR CANCELS
	if showSettingsUI = 1
		repeat
			beginPause ("Adjust pitch analysis settings")
				comment ("File 'filename$' (file number 'curFile' of 'fileCount')")
				comment ("You can change the pitch settings if the pitch track doesn't look right.")
				integer ("newMinPitch", minPitch)
				integer ("newMaxPitch", maxPitch)
				if advancedInterface = 1
					real ("newTimeStep", timeStep)
					integer ("newMaxCandidates", maxCandidates)
					boolean ("newHighAccuracy", highAccuracy)
					real ("newSilenceThreshold", silenceThreshold)
					real ("newVoicingThreshold", voicingThreshold)
					real ("newOctaveCost", octaveCost)
					real ("newVoicingCost", voicingCost)
					real ("newJumpCost", jumpCost)
				endif
				comment ("clicking RESET will reset parameters to the default values and redraw;")
				comment ("Clicking REDRAW will redraw the pitch contour with the settings above;")
				comment ("clicking SKIP will write zeros to the log file and go to next file.")
				sentence ("Notes", notes$)
				# AUTOPLAY
				if autoplay = 3 or firstview = 1
					editor Sound 'filename$'
						Play... 0 totalDur
					endeditor
				endif
				firstview = 0
			clicked = endPause ("Play","Reset", "Redraw", "Accept", "Skip", 3)

			# STILL NEED TO PASS ALONG THE ADVANCED SETTINGS, EVEN IF USER DIDN'T CHANGE THEM
			if advancedInterface = 0
				newTimeStep = timeStep
				newMaxCandidates = maxCandidates
				newHighAccuracy = highAccuracy
				newSilenceThreshold = silenceThreshold
				newVoicingThreshold = voicingThreshold
				newOctaveCost = octaveCost
				newVoicingCost = voicingCost
				newJumpCost = jumpCost
			endif

			# IF THE USER CLICKS "PLAY"
			if clicked = 1
				editor Sound 'filename$'
					Play... 0 totalDur
				endeditor

			# IF THE USER CLICKS "RESET"
			elif clicked = 2
				minPitch = defaultMinPitch
				maxPitch = defaultMaxPitch
				timeStep = defaultTimeStep
				maxCandidates = defaultMaxCandidates
				highAccuracy = defaultHighAccuracy
				silenceThreshold = defaultSilenceThreshold
				voicingThreshold = defaultVoicingThreshold
				octaveCost = defaultOctaveCost
				voicingCost = defaultVoicingCost
				jumpCost = defaultJumpCost

				# REDRAW THE PITCH CONTOUR
				editor Sound 'filename$'
					Pitch settings... minPitch maxPitch Hertz cross-correlation speckles
					Advanced pitch settings... viewRangeMin viewRangeMax highAccuracy maxCandidates silenceThreshold voicingThreshold octaveCost jumpCost voicingCost
				endeditor

			# IF THE USER CLICKS "REDRAW"
			elif clicked = 3
				minPitch = newMinPitch
				maxPitch = newMaxPitch
				timeStep = newTimeStep
				maxCandidates = newMaxCandidates
				highAccuracy = newHighAccuracy
				silenceThreshold = newSilenceThreshold
				voicingThreshold = newVoicingThreshold
				octaveCost = newOctaveCost
				voicingCost = newVoicingCost
				jumpCost = newJumpCost

				# REDRAW THE PITCH CONTOUR
				editor Sound 'filename$'
					Pitch settings... minPitch maxPitch Hertz cross-correlation speckles
					Advanced pitch settings... viewRangeMin viewRangeMax highAccuracy maxCandidates silenceThreshold voicingThreshold octaveCost jumpCost voicingCost
				endeditor
			endif
		until clicked >3

		# IF THE USER SKIPS, WRITE OVERRIDE VALUES
		if clicked = 5
			minPitch = 0
			maxPitch = 0
			timeStep = 0
			maxCandidates = 0
			highAccuracy = 0
			silenceThreshold = 0
			voicingThreshold = 0
			octaveCost = 0
			voicingCost = 0
			jumpCost = 0
		endif
	endif

	select Sound 'filename$'
	# IF WE'RE DOING MORE THAN JUST WRITING SETTINGS LOG
	if outputType > 0 ; not logOnly
		To Pitch (cc)... timeStep minPitch 15 highAccuracy silenceThreshold voicingThreshold octaveCost jumpCost voicingCost maxPitch
		Save as text file... 'pitchDir$''filename$'.Pitch
		plus Sound 'filename$'
		To Manipulation
		Extract pulses
		Rename... ppBackup
		select Manipulation 'filename$'

		if outputType = 3 ; (EDIT MANIPULATION OBJ. BEFORE SAVING)
			editor Sound 'filename$'
			Close
			select Manipulation 'filename$'
			View & Edit
			# UI FOR PULSE CORRECTION
			repeat
				beginPause ("Correct pulses")
					comment ("File 'filename$' (file number 'curFile' of 'fileCount')")
					# MAYBE ADD A NOTES BOX HERE TOO?
				clicked = endPause ("Play","Reset", "Redraw", "Finished", 3)

				# IF THE USER CLICKS "PLAY"
				if clicked = 1
					select Sound 'filename$'
					Play

				elif clicked=2
					# undo all the changes to the pulses
					select Manipulation 'filename$'
					plus PointProcess ppBackup
					Replace pulses
					minus Manipulation 'filename$'
					To PitchTier... 1/minPitch
					select Manipulation 'filename$'
					plus PitchTier ppBackup
					Replace pitch tier
					select PitchTier ppBackup
					Remove
					select Manipulation 'filename$'

				elif clicked=3
					# redraw the PitchTier from the new PointProcess
					select Manipulation 'filename$'
					Extract pulses
					To PitchTier... 1/minPitch
					# previous line: 20ms is the longest allowable gap for PSOLA algorithm, so lowest pitch it will track is 50Hz.  May be better just to hard code that value?
					select Manipulation 'filename$'
					plus PitchTier 'filename$'
					Replace pitch tier
					select PointProcess 'filename$'
					plus PitchTier 'filename$'
					Remove
					select Manipulation 'filename$'
				endif

			until clicked>3
			select PointProcess ppBackup
			Remove
		endif

		# DON'T REALLY NEED TO SAVE THESE OUT, SINCE THEY'RE PART OF THE MANIPULATION OBJECT
#		select Manipulation 'filename$'
#		Extract pulses
#		Save as text file... 'pulseDir$''filename$'.PointProcess
#		select Manipulation 'filename$'
#		Extract pitch tier
#		Save as text file... 'pitchDir$''filename$'.PitchTier
		select Manipulation 'filename$'
		Save as binary file... 'manipDir$''filename$'.Manipulation
#		plus PointProcess 'filename$'
#		plus PitchTier 'filename$'
		plus Pitch 'filename$'
		plus Sound 'filename$'
	endif
	Remove

	# WRITE TO LOG FILE
	resultline$ = "'curFile''tab$''filename$''tab$''totalDur''tab$''minPitch''tab$''maxPitch''tab$''timeStep''tab$''maxCandidates''tab$''highAccuracy''tab$''silenceThreshold''tab$''voicingThreshold''tab$''octaveCost''tab$''jumpCost''tab$''voicingCost''tab$''notes$''newline$'"
	fileappend "'logFile$'" 'resultline$'
endfor

# REMOVE THE STRINGS LIST AND GIVE A SUCCESS MESSAGE
select Strings list
Remove
clearinfo
files_read = fileCount - startingFileNum + 1
printline Done! 'files_read' files read.'newline$'

# FUNCTIONS (A.K.A. PROCEDURES) THAT WERE CALLED EARLIER
procedure cleanPath .in$
	if not right$(.in$, 1) = "/"
		.out$ = "'.in$'" + "/"
	else
		.out$ = "'.in$'"
	endif
endproc

procedure cleanExtn .in$
	if not left$(.in$, 1) = "."
		.out$ = "." + "'.in$'"
	else
		.out$ = "'.in$'"
	endif
endproc

procedure initializeOutfile
	headerline$ = "number'tab$'filename'tab$'duration'tab$'pitch_floor'tab$'pitch_ceiling'tab$'timeStep'tab$'maxCandidates'tab$'highAccuracy'tab$'silenceThreshold'tab$'voicingThreshold'tab$'octaveCost'tab$'jumpCost'tab$'voicingCost'tab$'notes'newline$'"
	fileappend "'logFile$'" 'headerline$'
endproc
