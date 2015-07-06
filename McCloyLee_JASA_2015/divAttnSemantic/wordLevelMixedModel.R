#! /usr/bin/R
# ============================
# Script 'wordLevelMixedModel'
# ============================
# This script reads in word-level values from a psychophysics experiment 
# (divAttnSemantic) and models target detection as a function of trial
# parameters using mixed effects regression.
# 
# Author: Dan McCloy <drmccloy@uw.edu>
# License: BSD (3-clause)

library(zoo)
library(lme4)
setwd("/home/dan/Documents/experiments/drmccloy/divAttnSemantic")

wl <- read.delim("analysis/wordLevelData.tsv",
                 colClasses=c(subj="character", trial="integer", sem="logical",
                              div="logical", adj="logical", stream="integer",
                              attn="logical", catg="character", word="character",
                              targ="logical", foil="logical", odbl="logical",
                              onset="numeric", dur="numeric", rawRT="numeric"))

wl <- wl[with(wl, order(subj, sem, trial, onset)), ]  # sort w/in trial by onset
wl$rawRT[wl$rawRT < 0] <- NA                     # convert -1 RTs to NA
wl$rawRT[wl$rawRT > 1.25] <- NA
rownames(wl) <- NULL                             # renumber rows


# # # # # # # # # # # # # # # # # # #
# insert indicator of stray presses #
# # # # # # # # # # # # # # # # # # #
# first log reaction times to targets and foils
wl <- within(wl, {
    rt <- ifelse(odbl & !is.na(rawRT), rawRT, NA)
    # now find blocks of 3 words in a row that were not oddballs, but fell
    # within the plausible RT window (0.4-1.25 seconds)
    sp1 <- rollapply(seq(nrow(wl)), 3, 
                     function(i) all(is.na(rt[i])) & all(!is.na(rawRT[i])), 
                     fill=FALSE)
    # next, discard cases where there was a target adjacent to that block of
    # three non-targets
    sp2 <- rollapply(seq(nrow(wl)), 5,
                     function(i) any(sp1[i]) & all(is.na(rt[i])),
                     fill=FALSE)
    sp <- apply(cbind(sp1, sp2), 1, all)
    sp1 <- NULL
    sp2 <- NULL
    rawRT <- NULL
    # finally, handle cases where there are 2 TRUEs in a row
    stray <- c(sapply(seq(nrow(wl)-1), function(i) sp[i] && !sp[i+1]), FALSE)
    press <- !is.na(rt) | stray
    hit <- !is.na(rt) & targ            # hits
    fal <- (!is.na(rt) & foil) | stray  # FAs (foil presses + stray presses)
})
# sanity check: hits + false alarms == presses
stopifnot(with(wl, sum(press) == sum(hit, fal)))


# VERIFY
# ix <- as.integer(rownames(wl[wl$sp,]))
# ix <- union(ix-3, ix)
# ix <- union(ix+1, ix)
# ix <- union(ix+1, ix)
# ix <- sort(union(ix+1, ix))
# foo <- wl[ix,]
# foo[1:100,]
# foo[101:200,] # etc...


# IGNORE SELECTIVE ATTN TRIALS?


# SET UP FACTORS / CONTRASTS FOR MODELING
wl <- within(wl, {
    truth <- factor(ifelse(targ, "target",
                           ifelse(foil, "foil", "neither")),
                    levels=c("neither", "target", "foil"))
    div <- factor(div, levels=c(TRUE, FALSE))
    adj <- factor(adj, levels=c(TRUE, FALSE))
    sem <- factor(sem, levels=c(TRUE, FALSE))
    # contrasts
    contrasts(truth) <- contr.treatment
    contrasts(div) <- contr.sum
    contrasts(adj) <- contr.sum
    contrasts(sem) <- contr.sum
    #subj <- factor(subj)
})


# MODELING
# empty model (random subj intercept only)
mm_empty <- glmer(press ~ (1|subj), 
                  data=wl, family=binomial(link="probit")
)
relgrad <- with(mm_empty@optinfo$derivs, solve(Hessian, gradient))
max(abs(relgrad))

# partial model (full fixef structure, random intercept only)
mm_part <- glmer(press ~ truth*sem*div*adj + (1|subj), 
                 data=wl, family=binomial(link="probit")
)
relgrad <- with(mm_part@optinfo$derivs, solve(Hessian, gradient))
max(abs(relgrad))

sink("glmerModelSummaries.txt")
cat("MM_EMPTY\n\n")
print(summary(mm_empty))
cat("\n\nMM_PART\n\n")
print(summary(mm_part), correlation=TRUE)
cat("\n\nMODEL COMPARISON\n\n")
print(anova(mm_part, mm_empty))
cat("\n\n")
print(summary(anova(mm_part, mm_empty)))
cat("\n\nBITS OF INFORMATION DIFFERENCE\n\n")
cat("(AIC(mm_empty) - AIC(mm_part)) * log2(exp(1))")
print((AIC(mm_empty)-AIC(mm_part))*log2(exp(1)))
sink()

fixef(mm_part)
confint(mm_part, method='Wald')

# full model
# mm_full <- glmer(press ~ truth*sem*div*adj + (1+truth*sem*div*adj|subj), 
#                  data=wl,
#                  family=binomial(link="probit"),
#                  verbose=2,
#                  #nAGQ=10,
#                  #control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=30000)),
# )




