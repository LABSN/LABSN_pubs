#!/bin/bash

# rt-hist.svg must be converted to EPS using Adobe Illustrator, in
# order to preserve the look of the transparent overlays.

# remove intermediate file leftover from generation time
rm trial-timecourse-tmp.svg
inkscape -f trial-timecourse.svg -A trial-timecourse.pdf

for fig in manhandled/*.svg; do
	tmp=$(basename $fig .svg)
	if [ $tmp != "rt-hist" ]; then
		inkscape -f $fig -E "$tmp.eps"
	fi
	inkscape -f $fig -e "$tmp.png" -d 300
	#inkscape -f $fig -A "$tmp.pdf"
done
