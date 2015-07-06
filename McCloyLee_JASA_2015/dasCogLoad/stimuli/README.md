# Raw recordings of monosyllabic words

Pipeline: files in `rawRecordings` are hand-segmented to create TextGrids. If the word list order corresponds to the interval order in the TextGrid, interval labels may be auto-populated with the script `annotateRawRecording.praat`. Non-empty intervals are extracted as WAV files by `segmentRawRecording.praat`, converted to Manipulation objects by `soundToManipulation.praat`, and monotonized by `monotonize107Hz.praat`. 

Folder pipeline (not tracked): rawRecordings > excisedWords > manipulationObjects > monotonizedWords (> finalStims)
