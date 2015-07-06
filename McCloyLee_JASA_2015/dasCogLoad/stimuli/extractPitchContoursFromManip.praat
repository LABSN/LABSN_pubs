form ExtractPitchContoursFromManip
	sentence input_folder manipulationObjects/
	sentence output_folder pitchDataFromManip/
endform

Create Strings as file list... flist 'input_folder$'*.Manipulation
n = Get number of strings

for i from 1 to n
	# READ IN EACH STIMULUS
	select Strings flist
	curFile$ = Get string... 'i'
	curManip = Read from file... 'input_folder$''curFile$'
	curName$ = selected$ ("Manipulation", 1)
	curPitch = Extract pitch tier
	Save as headerless spreadsheet file... 'output_folder$''curName$'.tab
	select curManip
	plus curPitch
	Remove
endfor

select Strings flist
Remove


