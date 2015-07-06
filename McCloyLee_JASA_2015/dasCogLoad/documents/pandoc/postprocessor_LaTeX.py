# -*- coding: utf-8 -*-
import sys

submission = True if '-s' in sys.argv else False
infile = sys.argv[-2]
outfile = sys.argv[-1]

with open(infile, 'r') as f:
	with open(outfile, 'w') as g:
		## initialize
		skipnextline = False
		appendix = False
		table = False
		xtable = False
		tablehead = False
		caption = False
		columnspec = ''
		## main loop
		for line in f:
			if skipnextline:
				skipnextline = False
				continue
			## spacing hacks
			line = line.replace(' vs. ', ' vs.\ ')
			line = line.replace('cf. ', 'cf.\ ')
			line = line.replace('St. ', 'St.\ ')
			line = line.replace('Table ', 'Table~')
			line = line.replace('Figure ', 'Figure~')
			line = line.replace('Figures ', 'Figures~')
			line = line.replace('Equation ', 'Equation~')
			line = line.replace('Experiment ', 'Experiment~')
			line = line.replace('Experiments ', 'Experiments~')
			## Table hacks
			if 'begin{longtable}' in line:
				table = True
				## replace column specs
				columnspec = line.strip().replace('\\begin{longtable}[c]', '')
				if columnspec == '{@{}lllll@{}}':
					xtable = True
					columnspec = '{@{}llllX@{}}'
				else:
					xtable =False
				if columnspec == '{@{}llrrrrc@{}}':
					columnspec = '{@{}l l S[round-mode=places,round-precision=2] S[round-mode=places,round-precision=2] S[round-mode=places,round-precision=2] S[round-mode=places,round-precision=3,table-format=>1.3e2] c@{}}'
				## make tables float
				line = '\\begin{table}[htbp]\n\\centering\n'
			## clean up at end of table
			if 'end{longtable}' in line:
				table = False
				columnspec = ''
				tabend = '\\end{tabularx}\n' if xtable else '\\end{tabular}\n'
				line = tabend + line.replace('longtable', 'table')
			## detect captions to know when to start table floats
			if table and 'caption{' in line:
				caption = True
			if caption and 'tabularnewline' in line:
				caption = False
				tabstart = '\n\\begin{tabularx}{\\textwidth}' if xtable else '\n\\begin{tabular}'
				line = line.replace('\\tabularnewline', tabstart + columnspec)
			## only replace textless in table data, not captions
			if table and not caption and 'textless' in line:
				line = line.replace('\\textless{}', '<')
			## remove pandoc's longtable line formatting, replace with booktabs
			if '\\toprule' in line:
				if tablehead:
					tablehead = False
				else:
					tablehead = True
			if tablehead:
				line = ''
			if '\\endhead' in line:
				line = ''
			## multicolumn hacks
			if '\\textbf{Baseline' in line:
				if '\\tabularnewline' not in line:
					skipnextline = True
				line = '\\multicolumn{7}{@{}l}{\\bfseries Baseline response levels}\\tabularnewline\n'
			if '\\textbf{Effect of manipulations on response bias' in line:
				if '\\tabularnewline' not in line:
					skipnextline = True
				line = '\\multicolumn{7}{@{}l}{\\bfseries Effect of manipulations on response bias}\\tabularnewline\n'
			if '\\textbf{Effect of manipulations on response to targets' in line:
				if '\\tabularnewline' not in line:
					skipnextline = True
				line = '\\multicolumn{7}{@{}l}{\\bfseries Effect of manipulations on response to targets}\\tabularnewline\n'
			if '\\textbf{Effect of manipulations on response to foils' in line:
				if '\\tabularnewline' not in line:
					skipnextline = True
				line = '\\multicolumn{7}{@{}l}{\\bfseries Effect of manipulations on response to foils}\\tabularnewline\n'
			## multirow hacks
			if table and xtable:
				if '\\texttt{attn.config}' in line or '\\texttt{trial.type}' in line:
					line = '\\midrule\n' + line
				if '\\(W_{fi}\\)' in line or '\\(C_{ai}\\)' in line:
					line = '\\cmidrule(l){3-5}\n' + line
				line = line.replace('\\texttt{word.type} &', '\\multirow{2}{*}{\\texttt{word.type}} &')
				line = line.replace('Treatment', '\\multirow{2}{*}{Treatment}')
				line = line.replace('\\texttt{attn.config} &', '\\multirow{5}{*}{\\texttt{attn.config}} &')
				line = line.replace('Helmert', '\\multirow{5}{*}{Helmert}')
				line = line.replace('\\texttt{selective}', '\\multirow{2}{*}{\\texttt{selective}}')
				line = line.replace('\\(C_{si}\\)', '\\multirow{2}{*}{\\(C_{si}\\)}')
				line = line.replace('\\texttt{adjacent}', '\\multirow{3}{*}{\\texttt{adjacent}}')
				line = line.replace('\\(C_{ai}\\)', '\\multirow{3}{*}{\\(C_{ai}\\)}')
			## push acknowledgments to separate page (JASA submission only)
			if submission:
				if 'section{Acknowledgments' in line:
					g.write('\\cleardoublepage\n')
					line = line.replace('section{Acknowledgments', 'section*{Acknowledgments')
			if 'section{Appendix' in line:
				apdxsep = '\\cleardoublepage\n' if submission else '\\FloatBarrier\n'
				g.write(apdxsep)
				line = line.replace('section{Appendix', 'section*{Appendix')
				appendix = True
			## convert to ASCII to be plain LaTeX friendly (not XeLaTeX)
			if submission:
				minus = '-' if table else '\\textminus{}'
				line = line.replace('−', minus)
				line = line.replace('°', '\\textdegree{}')
				line = line.replace('±', '\\textpm{}')
				line = line.replace('™', '\\texttrademark{}')
				line = line.replace('†', '\\ensuremath{\\dagger}')
				line = line.replace('×', '\\texttimes{}')
				line = line.replace('⅔', '$\\sfrac{2}{3}$')
				line = line.replace('⁻⅓', '$\sfrac{-1}{3}$')
				line = line.replace('ƒ₀', '\\ensuremath{\\mathit{f}_0}')
			## prevent prime collisions
			line = line.replace('d^\\prime', 'd\\thinspace^\\prime')
			## image handling
			if submission:
				## JASA standalone file must refer to images in same dir
				line = line.replace('../figures/jasa/', '')
			else:
				## web version uses PDFs instead of EPS (avoid bug in ps2pdf)
				line = line.replace('.eps', '.pdf')
			## table formatting hacks (must come at end because of greedy bracket replacement)
			for heading in ['Coef.', 'SE', '\\emph{p}', '\\emph{z}']:
				if heading in line and table:
					line = line.replace(heading, '{' + heading + '}')
			g.write(line)
