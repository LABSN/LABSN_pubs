#! /usr/bin/R
# ============================
# Script 'wordLevelMixedModel'
# ============================
# This script reads in word-level values from a psychophysics experiment
# (dasCogLoad) and models target detection as a function of trial parameters
# using mixed effects regression.
#
# Author: Dan McCloy <drmccloy@uw.edu>
# License: BSD (3-clause)

library(lme4)
setwd("/home/drmccloy/Experiments/dasCogLoad")
#setwd("/home/dan/Documents/academics/research/auditoryAttention/drmccloy/dasCogLoad")

exp <- "divAttnSem"
#exp <- "dasCogLoad"

if (exp %in% "dasCogLoad") {
    wl <- read.delim("processedData/wordLevelData.tsv",
                     colClasses=c(subn="integer", subj="character", trial="integer",
                                  div="logical", adj="logical", idn="logical",
                                  num="character", cond="character", stream="integer",
                                  attn="logical", catg="character", word="character",
                                  targ="logical", foil="logical", odbl="logical",
                                  onset="numeric", dur="numeric", rt="numeric",
                                  hit="logical", fht="logical", sty="logical",
                                  fal="logical", miss="logical", crj="logical",
                                  notg="logical", date="character"))

    wl <- wl[with(wl, order(subn, trial, onset)), ]  # sort within trial by word onset
    wl$rt[wl$rt < 0] <- NA                           # convert -1 RTs to NA
    rownames(wl) <- NULL                             # renumber rows
    wl$date <- NULL                                  # ditch datestamp column
} else {
    wl <- read.delim("processedData/divAttnSemWordLevelData.tsv",
                     colClasses=c(subn="integer", subj="character", trial="integer",
                                  sem="logical", div="logical", adj="logical",
                                  stream="integer", attn="logical", catg="character",
                                  word="character", targ="logical", foil="logical",
                                  odbl="logical", onset="numeric", dur="numeric",
                                  rt="numeric", hit="logical", fht="logical",
                                  sty="logical", fal="logical", miss="logical",
                                  crj="logical", notg="logical"))

    wl <- wl[with(wl, order(subn, trial, onset)), ]  # sort within trial by word onset
    wl$rt[wl$rt < 0] <- NA                           # convert -1 RTs to NA
    rownames(wl) <- NULL                             # renumber rows
}

# # # # # # # # # # # # # # # # # # #
# insert indicator of stray presses #
# # # # # # # # # # # # # # # # # # #
# first log reaction times to targets and foils
wl <- within(wl, {
    # discard unused columns
    tloc <- NULL
    floc <- NULL
    onset_og <- NULL
    word_og <- NULL
    catg_og <- NULL
    rtch <- NULL
    rtch_hit <- NULL
    rtch_fht <- NULL
    # add boolean press / nopress column
    press <- !is.na(rt)
})
# sanity check: hits + false alarms == presses
stopifnot(with(wl, sum(press) == sum(hit, fal)))


# SET UP FACTORS / CONTRASTS FOR MODELING
wl <- within(wl, {
    subj <- factor(subj)
    truth <- ifelse(targ, "target", ifelse(foil, "foil", "neither"))
    truth <- factor(truth, levels=c("neither", "target", "foil"))
    contrasts(truth) <- contr.treatment
    colnames(contrasts(truth)) <- levels(truth)[-1]
    if (exp %in% "dasCogLoad") {
        # these all get sum contrasts (they're binary and mutually orthogonal)
        adj <- factor(adj, levels=c(TRUE, FALSE))
        idn <- factor(idn, levels=c(TRUE, FALSE))
        num <- factor(num, levels=c("thr", "six"), labels=c("three", "six"))
        contrasts(adj) <- contr.sum
        contrasts(idn) <- contr.sum
        contrasts(num) <- contr.sum
        contrasts(adj) <- contrasts(adj) * 0.5
        contrasts(idn) <- contrasts(idn) * 0.5
        contrasts(num) <- contrasts(num) * 0.5
        colnames(contrasts(adj)) <- levels(adj)[1]
        colnames(contrasts(idn)) <- levels(idn)[1]
        colnames(contrasts(num)) <- levels(num)[1]
    } else {
        # sum contrasts for semantic vs phonetic
        sem <- factor(sem, levels=c(TRUE, FALSE))
        contrasts(sem) <- contr.sum
        contrasts(sem) <- contrasts(sem) * 0.5
        colnames(contrasts(sem)) <- levels(sem)[1]
        # helmert contrasts for div[adj] vs div[sep] & div[adj+sep] vs sel
        sel <- ifelse(div, ifelse(adj, "adj", "sep"), "sel")
        sel <- factor(sel, levels=c("sep", "adj", "sel"))
        contrasts(sel) <- contr.helmert
        contrasts(sel) <- contrasts(sel) / rep(c(2, 3), each=3)
        colnames(contrasts(sel)) <- levels(sel)[-1]
    }
})

# MAKE SURE WE DIDN'T DO ANYTHING STUPID
if (exp %in% "dasCogLoad") {
    check.factors <- lapply(na.omit(wl[, c("truth", "adj", "idn", "num")]), table)
} else {
    check.factors <- lapply(na.omit(wl[, c("truth", "sem", "sel")]), table)
}


# MODELING
# null model (random subj intercept only)
mm_zero <- glmer(press ~ (1|subj),
                 data=wl, family=binomial(link="probit"),
                 control=glmerControl(optCtrl=list(maxfun=20000)))
if (exp %in% "dasCogLoad") {
    save(mm_zero, file="documents/stats/glmerZeroModel.Rdata")
} else {
    save(mm_zero, file="documents/stats/glmerZeroModel_DivAttnSem.Rdata")
}
mm_null <- glmer(press ~ truth + (1|subj),
                 data=wl, family=binomial(link="probit"),
                 control=glmerControl(optCtrl=list(maxfun=20000)))

#relgrad <- with(mm_null@optinfo$derivs, solve(Hessian, gradient))
#print(max(abs(relgrad)))

# partial model (full fixef structure, random intercept only)
if (exp %in% "dasCogLoad") {
    save(mm_null, file="documents/stats/glmerNullModel.Rdata")
    mm_part <- glmer(press ~ truth*adj*idn*num + (1|subj),
                     data=wl, family=binomial(link="probit"),
                     control=glmerControl(optCtrl=list(maxfun=20000)))
    sink("documents/stats/glmerModelSummaries.txt")
    save(mm_part, file="documents/stats/glmerModel.Rdata")
} else {
    save(mm_null, file="documents/stats/glmerNullModel_DivAttnSem.Rdata")
    mm_part <- glmer(press ~ truth*sel*sem + (1|subj),
                     data=wl, family=binomial(link="probit"))
    sink("documents/stats/glmerModelSummaries_DivAttnSem.txt")
    save(mm_part, file="documents/stats/glmerModel_DivAttnSem.Rdata")
}
# write to file
cat("MM_NULL\n\n")
print(summary(mm_null))
cat("\n\nMM_PART\n\n")
print(summary(mm_part), correlation=TRUE)
cat("\n\nMODEL COMPARISON\n\n")
print(anova(mm_part, mm_null))
cat("\n\n")
print(summary(anova(mm_part, mm_null)))
cat("\n\nBITS OF INFORMATION DIFFERENCE\n\n")
cat("(AIC(mm_null) - AIC(mm_part)) * log2(exp(1))\n")
print((AIC(mm_null)-AIC(mm_part))*log2(exp(1)))
sink()

#relgrad <- with(mm_part@optinfo$derivs, solve(Hessian, gradient))
#print(max(abs(relgrad)))

# fixef(mm_part)
# confint(mm_part, method="Wald")

# full model
# mm_full <- glmer(press ~ truth*adj*idn*num + (1+truth*adj*idn*num|subj),
#                  data=wl,
#                  family=binomial(link="probit"),
#                  verbose=2,
#                  #nAGQ=10,
#                  #control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=30000)),
# )
