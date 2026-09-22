# Run once from the analysis directory. Install unmodified, pinned packages.
dir.create(".R-library", showWarnings=FALSE)
.libPaths(c(normalizePath(".R-library"), .libPaths()))
options(repos=c(CRAN="https://cloud.r-project.org"))
needed <- setdiff(c("remotes", "BiocManager", "tidyverse", "readxl", "statmod", "pbapply", "RPostgres"),
                  rownames(installed.packages()))
if (length(needed)) install.packages(needed)
options(repos=BiocManager::repositories())
BiocManager::install(c("MSnbase", "limma", "edgeR"), ask=FALSE, update=FALSE)
remotes::install_github(c(
  "PNNL-Comp-Mass-Spec/TopPICR@0acc4215bda381d313d85e17519625c4a680db58",
  "PNNL-Comp-Mass-Spec/MSnSet.utils@9c66e50f4d1f17a1db7dacba8ed777cd1c6e0941"
), dependencies=NA, upgrade="never", build_vignettes=FALSE, force=TRUE)
# Keep the MSnSet.utils pin rather than following PNNL.DMS.utils' unpinned remote.
remotes::install_github(
  "PNNL-Comp-Mass-Spec/PNNL.DMS.utils@1a4d8ed74692b55958c1937c592c21d42a25a94c",
  dependencies=FALSE, upgrade="never", build_vignettes=FALSE, force=TRUE
)
