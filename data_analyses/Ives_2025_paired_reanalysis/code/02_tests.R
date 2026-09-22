
# Authors: Tyler Sagendorf, Ashley Ives
# Purpose: akes output of script "0a" and performs differential abundance analysis using the functions "limmaFit.R" and "limmaDEA.R".

# LOCAL EDIT BEGIN setup
# ORIGINAL (commented out):
# REPLACEMENT:
# Adapted from source/authors/1_DEA.R to reproduce the published supplement.
# Our only analysis addition is the complete-case export for Julia at the end.
# Run from the analysis directory.
.libPaths(c(".R-library", .libPaths()))
options(digits = 16)
dir.create("results", showWarnings = FALSE)

# LOCAL EDIT END setup
library(dplyr)
# LOCAL EDIT BEGIN plot_packages
# ORIGINAL (commented out):
# library(ggpubr)
# library(ggrepel)
# REPLACEMENT:
# Diagnostic plots are omitted below.
# LOCAL EDIT END plot_packages
library(tidyverse)
library(grid) 
# LOCAL EDIT BEGIN packages
# ORIGINAL (commented out):
# library(proDA)
# REPLACEMENT:
# Use the saved MSnSet and the authors' normalization function.
library(MSnbase)
library(MSnSet.utils)
# LOCAL EDIT END packages

# 0. Load in functions of differential abundance analysis------------------------
# LOCAL EDIT BEGIN sources
# ORIGINAL (commented out):
# source("limmaFit.R")
# source("limmaDEA.R")
# REPLACEMENT:
source("source/authors/limmaFit.R")
source("source/authors/limmaDEA.R")
# LOCAL EDIT END sources

# 1. Load data generated in "0a_loading_intensity_data.R" and convert to usable format.  

# LOCAL EDIT BEGIN input
# ORIGINAL (commented out):
# load("msnset_humanislet_int_log2center_oct2024_forTyler_wmodanno.RData")
# REPLACEMENT:
# Output of 01_reconstruct.R, before unavailable modification annotation.
load("data/msnset_humanislet_int_log2center_oct2024_forTyler.RData")
# LOCAL EDIT END input

m <-  mlc 

m <- m[, with(pData(m), order(Pair, Treated))]

fData(m) <- fData(m) %>%
  dplyr::rename(n_unexpected_modifications = `#unexpected modifications`)

## Keep features that have complete data for at least two pairs ----------------
pair_design <- model.matrix(~ 0 + m$Pair)

mat_nonmiss <- apply(!is.na(exprs(m)), 2, as.integer)
rownames(mat_nonmiss) <- rownames(m)
n_nonmiss <- mat_nonmiss %*% pair_design

keep <- apply(n_nonmiss == 2L, 1, function(xi) sum(xi) >= 2L)
table(keep)
m2 <- m[keep, ]

## Normalize data using median polish-------------------------------------------
m2 <- MSnSet.utils::normalizeByGlob(m2, method = "once")

# LOCAL EDIT BEGIN MDS_plot
# ORIGINAL (commented out):
# ## MDS plot, color by donor ----
# idx <- match(m2$Pair, sort(unique(m2$Pair)))
# 
# col <- c("black", "red", "green", "blue", "orange", "purple")
# col <- col[idx]
# 
# labels <- c(m2$Pair)
# 
# pdf(file = "humanislet_processed_MSnSet_MDS_plot_2.pdf",
#     width = 6, height = 5)
# plotMDS(exprs(m2), col = col, labels=labels)
# dev.off()
# 
# REPLACEMENT:
# Omit the diagnostic plot. Filtering and normalization above are unchanged.
# LOCAL EDIT END MDS_plot
# 2. Perform differential analysis --------------------------------------------

# Create fitted object. The block argument makes this a paired comparison
fit <- limmaFit(object = m2,
                model.str = "~ 0 + Treated",
                contrasts = c("TreatedTreated - TreatedNottreated"),
# LOCAL EDIT BEGIN model_plot
# ORIGINAL (commented out):
#                 block = "Pair", plot = TRUE,
# REPLACEMENT:
                block = "Pair", plot = FALSE,
# LOCAL EDIT END model_plot
                trend = TRUE, robust = TRUE)

# LOCAL EDIT BEGIN legacy
# ORIGINAL (commented out):
# REPLACEMENT:
# The original wrapper uses the installed limma defaults. On newer limma,
# redo only moderation with legacy prior fitting for the published comparison.
if ("legacy" %in% names(formals(limma::eBayes))) {
  fit <- limma::eBayes(fit, trend = TRUE, robust = TRUE, legacy = TRUE)
}
# LOCAL EDIT END legacy
# DEA results table
res <- limmaDEA(fit = fit) %>%
  arrange(P.Value) %>%
  mutate(contrast = gsub("Treated(?! )", "", contrast, perl = TRUE))

# Identify list columns
list_columns <- sapply(res, is.list)

# Convert list columns to character vectors to make them CSV-friendly
res <- res %>%
  mutate(across(where(is.list), ~ sapply(., toString)))

x_final <- as.data.frame(m) %>%
  t()

# 3. Add data completeness, i.e. what percentage of samples proteoform is observed in.

rowwise_na_summary <- apply(x_final, 1, function(x) sum(!is.na(x))) %>%
  as.data.frame() %>%
  rownames_to_column(var = "PF")

rowwise_na_summary$notna <- rowwise_na_summary$.

rowwise_na_summary <- rowwise_na_summary %>%
  mutate(MissingPercent = (notna/ncol(x_final)))

#it's NOTna/total number, so 1.0 means PF is in all samples 
res <- res %>%
  left_join(rowwise_na_summary, by="PF")

# # Write to CSV after conversion
# write.csv(res, "humanislets_DEA_results_2.csv", row.names = FALSE)

# LOCAL EDIT BEGIN histogram
# ORIGINAL (commented out):
# # Check shape of p-value histogram
# pdf(file = "humanislet_DEA_p-value_histogram_2.pdf",
#     height = 5, width = 6)
# hist(res$P.Value, breaks = seq(0, 1, 0.05),
#      xlab = "P-Value",
#      main = NA)
# dev.off()
# 
# REPLACEMENT:
# Omit the diagnostic histogram.
# LOCAL EDIT END histogram
#save data, will be input for "Figure3_volcano_plots_rawp.R"
# LOCAL EDIT BEGIN outputs
# ORIGINAL (commented out):
# save(res, file="res_forvolcano.RData")
# write.csv(res, file= "logFC_TDislet.csv")
# REPLACEMENT:
res$feature_id <- res$PF
write.csv(res, "results/authors_model_reconstructed.csv", row.names = FALSE, na = "")
# LOCAL EDIT END outputs


# LOCAL EDIT BEGIN export_differences
# ORIGINAL (commented out):
# REPLACEMENT:
# Complete cases for our Julia analysis: treated minus control for each donor.
L <- exprs(m2)
control <- which(m2$Treated == "Nottreated")
treated <- which(m2$Treated == "Treated")
stopifnot(length(control) == 6L, length(treated) == 6L,
          identical(m2$Pair[control], m2$Pair[treated]),
          !anyDuplicated(m2$Pair[control]))
D <- L[, treated, drop = FALSE] - L[, control, drop = FALSE]
colnames(D) <- m2$Pair[control]
D <- D[complete.cases(D), , drop = FALSE]
write.csv(data.frame(feature_id = rownames(D), D, check.names = FALSE),
          "data/paired_differences_complete6.csv", row.names = FALSE)
writeLines(capture.output(sessionInfo()), "results/analysis_sessionInfo.txt")
# LOCAL EDIT END export_differences
