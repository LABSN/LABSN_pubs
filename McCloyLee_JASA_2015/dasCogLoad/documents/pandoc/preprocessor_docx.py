# -*- coding: utf-8 -*-
import sys
infile = sys.argv[-2]
outfile = sys.argv[-1]

formulae = [('$\Phi^{-1}(y_{ij}) = \\beta_0 + \\beta_1 W_{ti} + \\beta_2 W_{fi} + \\beta_3 T_i + \\beta_4 C_{ai} + \\beta_5 C_{si} + \\ldots + S_{0j} + \\epsilon_{ij}$',
             'Φ^−1^(_**y~ij~**_) = _**β~0~**_ + _**β~1~W~ti~**_ + _**β~2~W~fi~**_ + _**β~3~T~i~**_ + _**β~4~C~ai~**_ + _**β~5~C~si~**_ + ... + _**S~0j~**_ + _**ϵ~ij~**_'),
            ('$\Phi^{-1}(y_{ij}) = \\beta_0 + \\beta_1 W_{ti} + \\beta_2 W_{fi} + \\beta_3 Z_i + \\beta_4 A_i + \\beta_5 C_i + \ldots + S_{0j} + \epsilon_{ij}$', 
            'Φ^−1^(_**y~ij~**_) = _**β~0~**_ + _**β~1~W~ti~**_ + _**β~2~W~fi~**_ + _**β~3~Z~i~**_ + _**β~4~A~i~**_ + _**β~5~C~i~**_ + ... + _**S~0j~**_ + _**ϵ~ij~**_'),
			('$d^\\prime = \\Phi^{-1}(\\textrm{hit rate}) - \\Phi^{-1}(\\textrm{false alarm rate})$', 'Φ^−1^(hit rate) − Φ^−1^(false alarm rate)'),
            ('$Pr(Y = 1 \\mid X) = \\Phi(X^\\prime \\beta)$',                                         'Pr(_**Y**_=1 | _**X**_) = Φ(_**X′β**_)'),
			('$\\Phi^{-1}(Pr(Y = 1 \\mid X)) = X^\\prime \\beta$',                                    'Φ^−1^(Pr(_**Y**_=1 | _**X**_)) = _**X′β**_'),
			('$Pr(Y = 1 \\mid X)$', 'Pr(_**Y**_=1 | _**X**_)'),
			('$\\Phi^{-1}(k)$',     'Φ^−1^(k)'),
			('$X$',                 '_**X**_'),
			('$Y$',                 '_**Y**_'),
			('$Y=1$',               '_**Y=1**_'),
			('$\\Phi$',             'Φ'),
			('$\\Phi^{-1}$',        'Φ^−1^'), 
			('$\\beta$',            '_**β**_'),
			('$y_{ij}$',            '_**y**_~ij~'),
			('$i$',                 '_i_'),
			('$j$',                 '_j_'),
			('$\\beta_0$',          '_**β**_~0~'),
			('$\\beta$',            '_**β**_'),
			('$W_{ti}$',            '_**W**_~_ti_~'),
			('$W_{fi}$',            '_**W**_~_fi_~'),
			('$W_{ti}=1$',          '_**W**_~_ti_~=1'),
			('$W_{fi}=1$',          '_**W**_~_fi_~=1'),
			('$W_{ti}=W_{fi}=0$',   '_**W**_~_ti_~=_**W**_~_fi_~=0'),
			('$T_i$',               '_**T**_~_i_~'),
			('$C_{ai}$',            '_**C**_~_ai_~'),
			('$C_{si}$',            '_**C**_~_si_~'),
			('$W_{ti} : T_i$',          '_**W**_~_ti_~:_**T**_~_i_~'),
			('$W_{fi} : T_i$',          '_**W**_~_fi_~:_**T**_~_i_~'),
			('$W_{ti} : C_{ai}$',       '_**W**_~_ti_~:_**C**_~_ai_~'),
			('$W_{fi} : C_{ai}$',       '_**W**_~_fi_~:_**C**_~_ai_~'),
			('$W_{ti} : C_{si}$',       '_**W**_~_ti_~:_**C**_~_si_~'),
			('$W_{fi} : C_{si}$',       '_**W**_~_fi_~:_**C**_~_si_~'),
			('$T_i : C_{ai}$',          '_**T**_~_i_~:_**C**_~_ai_~'),
			('$T_i : C_{si}$',          '_**T**_~_i_~:_**C**_~_si_~'),
			('$W_{ti} : T_i : C_{ai}$',    '_**W**_~_ti_~:_**T**_~_i_~:_**C**_~_ai_~'), 
			('$W_{fi} : T_i : C_{ai}$',    '_**W**_~_fi_~:_**T**_~_i_~:_**C**_~_ai_~'),
			('$W_{ti} : T_i : C_{si}$',    '_**W**_~_ti_~:_**T**_~_i_~:_**C**_~_si_~'),
			('$W_{fi} : T_i : C_{si}$',    '_**W**_~_fi_~:_**T**_~_i_~:_**C**_~_si_~'),
			('$Z_i : A_i$',                '_**Z**_~_i_~:_**A**_~_i_~'),
			('$Z_i : C_i$',                '_**Z**_~_i_~:_**C**_~_i_~'),
			('$A_i : C_i$',                '_**A**_~_i_~:_**C**_~_i_~'),
			('$Z_i : A_i : C_i$',          '_**Z**_~_i_~:_**A**_~_i_~:_**C**_~_i_~'),
			('$W_{ti} : Z_i$',             '_**W**_~_ti_~:_**Z**_~_i_~'),
			('$W_{ti} : A_i$',             '_**W**_~_ti_~:_**A**_~_i_~'),
			('$W_{ti} : C_i$',             '_**W**_~_ti_~:_**C**_~_i_~'),
			('$W_{ti} : Z_i : A_i$',       '_**W**_~_ti_~:_**Z**_~_i_~:_**A**_~_i_~'),
			('$W_{ti} : Z_i : C_i$',       '_**W**_~_ti_~:_**Z**_~_i_~:_**C**_~_i_~'),
			('$W_{ti} : A_i : C_i$',       '_**W**_~_ti_~:_**A**_~_i_~:_**C**_~_i_~'),
			('$W_{ti} : Z_i : A_i : C_i$', '_**W**_~_ti_~:_**Z**_~_i_~:_**A**_~_i_~:_**C**_~_i_~'),
			('$W_{fi} : Z_i$',             '_**W**_~_fi_~:_**Z**_~_i_~'),
			('$W_{fi} : A_i$',             '_**W**_~_fi_~:_**A**_~_i_~'),
			('$W_{fi} : C_i$',             '_**W**_~_fi_~:_**C**_~_i_~'),
			('$W_{fi} : Z_i : A_i$',       '_**W**_~_fi_~:_**Z**_~_i_~:_**A**_~_i_~'),
			('$W_{fi} : Z_i : C_i$',       '_**W**_~_fi_~:_**Z**_~_i_~:_**C**_~_i_~'),
			('$W_{fi} : A_i : C_i$',       '_**W**_~_fi_~:_**A**_~_i_~:_**C**_~_i_~'),
			('$W_{fi} : Z_i : A_i : C_i$', '_**W**_~_fi_~:_**Z**_~_i_~:_**A**_~_i_~:_**C**_~_i_~'),
			#('$\\sfrac{-2}{3}$',    '⁻⅔'),
			#('$\\sfrac{1}{3}$',     '⅓'),
			('$Z_i$',               '_**Z**_~_i_~'),
			('$A_i$',               '_**A**_~_i_~'),
			('$C_i$',               '_**C**_~_i_~'),
			('$S_{0j}$',            '_**S~0j~**_'),
			('$\\epsilon_{ij}$',    '_**ϵ~ij~**_'),
			('$R^2$',               '_R_^2^'),
			('$m$',                 '_m_'),
			('$R_m^2$',             '_R~m~_^2^'),
			('$R_c^2$',             '_R~c~_^2^'),
			('$\\chi^2$',           'χ^2^'),
			('$d^\\prime$',         'd′')
			]
with open(infile, 'r') as f:
	with open(outfile, 'w') as g:
		for line in f:
			for tex, uni in formulae:
				line = line.replace(tex, uni)
			if '![' in line:
				line = line.replace('.eps ', '.png ')
			if '\\ref{' in line:
				line = line.replace('\\ref{sec-bkgd}', 'II.')
				line = line.replace('\\ref{sec-gen-meth}', 'III.')
				line = line.replace('\\ref{sec-resp-window}', 'D.')
				line = line.replace('\\ref{sec-gen-stats}', 'E.')
				line = line.replace('\\ref{fig-', '')
				line = line.replace('\\ref{tab', '')
				line = line.replace('}', '')
			g.write(line)
