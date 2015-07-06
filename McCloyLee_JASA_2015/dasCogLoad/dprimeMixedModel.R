#! /usr/bin/R
# ============================
# Script 'd-prime mixed model'
# ============================
# This script reads in d-prime values from a psychophysics experiment
# (dasCogLoad) and models the conditions with mixed effects regression.
#
# Author: Dan McCloy <drmccloy@uw.edu>
# License: BSD (3-clause)


das <- read.delim('processedData/divAttnSemData.tsv', 
				  colClasses=c(subj="character", trial="numeric", sem="logical",
				  	           div="logical", adj="logical", catg="character", 
				  	           attn="character", words="character",
				  	           targ_words="character", foil_words="character",
				  	           times="character", targ_times="character",
				  	           foil_times="character", press_times="character",
				  	           targs="numeric", foils="numeric", hits="numeric", 
				  	           miss="numeric", snum="numeric", 
				  	           targ_loc="character", foil_loc="character",
				  	           rawrt="character", rt_hit="character", 
				  	           rt_fht="character", ="character"rt, 
				  	           presses="numeric", fht="numeric", 
				  	           sty="numeric", fal="numeric",
				  	           crj="numeric", frj="numeric"))

#setwd("/home/drmccloy/Experiments/dasCogLoad")
#dp <- read.delim('processedData/dprimeData.tsv', row.names=1, colClasses=c('character', rep('numeric', 16)))
#dp <- read.delim('processedData/trialLevelData.tsv',
dp <- read.delim('processedData/AggregatedFinal.tsv',
                 colClasses=c(subj="character", trial="character", div="character",
                              adj="character", idn="character", num="character",
                              cond="character", code="character",
                              cond_code="character", targ="numeric",
                              notg="numeric", foil="numeric", hit="numeric",
                              miss="numeric", #fht="numeric", sty="numeric",
                              fal="numeric", crj="numeric",
                              #snd="numeric", lft="numeric",
                              dprime="numeric", rt="character",
                              rt_hit="character", rt_fht="character",
                              #rt_snd="character", rt_lft="character",
                              rtch="numeric", rtch_hit="numeric",
                              rtch_fht="numeric",
                              #rtch_snd="numeric", rtch_lft="numeric",
                              hrate="numeric", #lrate="numeric"
                              frate="numeric" #,
                              #sandwich="numeric", leftover="numeric"
                              ))

parse.list <- function(x, dtype=as.character) {
  y <- gsub("[", "", x, fixed=TRUE)
  y <- gsub("]", "", y, fixed=TRUE)
  y <- gsub("'", "", y, fixed=TRUE)
  y <- lapply(strsplit(y, ", "), dtype)
}

dp <- within(dp, {
  trial     <- parse.list(trial, as.integer)
  div       <- parse.list(div, as.logical)
  adj       <- parse.list(adj, as.logical)
  idn       <- parse.list(idn, as.logical)
  #num       <- parse.list(num)
  cond      <- parse.list(cond)
  cond_code <- parse.list(cond)
  rt        <- parse.list(rt, as.numeric)
  rt_hit    <- parse.list(rt_hit, as.numeric)
  rt_fht    <- parse.list(rt_fht, as.numeric)
})


# simplify / reduce
df <- with(dp, data.frame(subj=subj, dprime=dprime))  #size=num, ident=idn, config=adj,
df <- within(df, {
  div <- unlist(dp$div)
  size <- dp$num
  ident <- dp$idn
  config <- dp$adj
})
df <- df[with(df, div & !size %in% 'bth' & ident %in% c(TRUE, FALSE) & config %in% c(TRUE, FALSE)),]
df <- within(df, {
  config <- unlist(config)
  config <- ifelse(config, "adj", "sep")
  config <- factor(config, levels=c("sep", "adj"), ordered=TRUE)
  ident <- unlist(ident)
  ident <- ifelse(ident, "same", "diff")
  ident <- factor(ident, levels=c("diff", "same"), ordered=TRUE)
  size <- factor(size, levels=c("six", "thr"), ordered=TRUE)
})
df <- df[with(df, order(subj, config, ident, size)),]
# set deviation coding for contrasts
contrasts(df$size) <- contr.sum
contrasts(df$ident) <- contr.sum
contrasts(df$config) <- contr.sum
contrasts(df$size) <- 0.5 * contrasts(df$size)
contrasts(df$ident) <- 0.5 * contrasts(df$ident)
contrasts(df$config) <- 0.5 * contrasts(df$config)


library(lme4)
library(ez)

mixmod1 <- lmer(dprime ~ size * ident * config + (1|subj), data=df)
mixmod2 <- lmer(dprime ~ size + ident + config + size:ident + size:config + ident:config + (1|subj), data=df)

anova1 <- ezANOVA(data=df, dv=dprime, wid=subj, within=list(size, ident, config))

sink('dprimeANOVASummaries.txt')
cat('# # # # # # # # #\n')
cat('# ezANOVA: full #\n')
cat('# # # # # # # # #\n\n')
print(anova1)
cat('\n\n\n')
cat('# # # # # # # # # # #\n')
cat('# Mixed Model: full #\n')
cat('# # # # # # # # # # #\n\n')
print(summary(mixmod1))
cat('\n\n\n')
cat('# # # # # # # # # # # # # # # # # # # # # # # #\n')
cat('# Mixed Model: removed three-way interaction  #\n')
cat('# # # # # # # # # # # # # # # # # # # # # # # #\n\n')
print(summary(mixmod2))
sink()

#par(mfrow=c(1, 3))
#plot(factor(df$ident), df$dprime)
#plot(factor(df$size), df$dprime)
#plot(factor(df$config), df$dprime)
