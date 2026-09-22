# Quantile normalization and probe filters from source/methylation_data.qmd.
# Run from data_analyses/methylation; optionally supply an existing raw-data folder.
args <- commandArgs(trailingOnly = TRUE)
raw_directory <- if (length(args)) args[1] else "source/data"
if (length(args)) stopifnot(dir.exists(raw_directory))
dir.create("source", showWarnings = FALSE)
dir.create("data", showWarnings = FALSE)
if (!dir.exists(raw_directory)) {
    archive <- "source/methylAnalysisDataV3.tar.gz"
    if (!file.exists(archive))
        download.file("https://ndownloader.figshare.com/files/7896205", archive, mode = "wb")
    untar(archive, exdir = "source")
}

library(minfi)
library(IlluminaHumanMethylation450kanno.ilmn12.hg19)
library(IlluminaHumanMethylation450kmanifest)
BiocParallel::register(BiocParallel::SerialParam())

targets <- read.metharray.sheet(raw_directory, pattern = "SampleSheet.csv")
targets <- targets[targets$Sample_Source %in% c("M28", "M29", "M30"), ]
stopifnot(nrow(targets) == 10L)
rg <- read.metharray.exp(targets = targets)
keys <- paste(targets$Sample_Source, targets$Sample_Group, sep = ".")
stopifnot(!anyDuplicated(keys))
sampleNames(rg) <- keys
detP <- detectionP(rg)
stopifnot(all(colMeans(detP) < 0.05))

normalized <- preprocessQuantile(rg)
detP <- detP[match(featureNames(normalized), rownames(detP)), ]
normalized <- normalized[rowSums(detP < 0.01) == ncol(normalized), ]
normalized <- dropLociWithSnps(normalized)
cross_reactive <- read.csv(file.path(raw_directory, "48639-non-specific-probes-Illumina450k.csv"))
normalized <- normalized[!featureNames(normalized) %in% cross_reactive$TargetID, ]
M <- getM(normalized)
stopifnot(nrow(M) == 439918L, all(is.finite(M)))

write.csv(data.frame(probe = rownames(M), M, check.names = FALSE),
    "data/arrays.csv", row.names = FALSE)
write.csv(data.frame(sample = keys, donor = targets$Sample_Source,
    cell_type = targets$Sample_Group), "data/samples.csv", row.names = FALSE)
writeLines(capture.output(sessionInfo()), "data/R_session.txt")
