
# Authors: Tyler Sagendorf, Ashley Ives
# Purpose: akes output of script "0a" and performs differential abundance analysis using the functions "limmaFit.R" and "limmaDEA.R".

library(dplyr)
library(ggpubr)
library(ggrepel)
library(tidyverse)
library(grid) 
library(proDA)

# 0. Load in functions of differential abundance analysis------------------------
source("limmaFit.R")
source("limmaDEA.R")

# 1. Load data generated in "0a_loading_intensity_data.R" and convert to usable format.  

load("msnset_humanislet_int_log2center_oct2024_forTyler_wmodanno.RData")

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

## MDS plot, color by donor ----
idx <- match(m2$Pair, sort(unique(m2$Pair)))

col <- c("black", "red", "green", "blue", "orange", "purple")
col <- col[idx]

labels <- c(m2$Pair)

pdf(file = "humanislet_processed_MSnSet_MDS_plot_2.pdf",
    width = 6, height = 5)
plotMDS(exprs(m2), col = col, labels=labels)
dev.off()

# 2. Perform differential analysis --------------------------------------------

# Create fitted object. The block argument makes this a paired comparison
fit <- limmaFit(object = m2,
                model.str = "~ 0 + Treated",
                contrasts = c("TreatedTreated - TreatedNottreated"),
                block = "Pair", plot = TRUE,
                trend = TRUE, robust = TRUE)

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

# Check shape of p-value histogram
pdf(file = "humanislet_DEA_p-value_histogram_2.pdf",
    height = 5, width = 6)
hist(res$P.Value, breaks = seq(0, 1, 0.05),
     xlab = "P-Value",
     main = NA)
dev.off()

#save data, will be input for "Figure3_volcano_plots_rawp.R"
save(res, file="res_forvolcano.RData")
write.csv(res, file= "logFC_TDislet.csv")


