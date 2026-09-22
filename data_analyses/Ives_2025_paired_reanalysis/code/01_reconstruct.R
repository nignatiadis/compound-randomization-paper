# Authors: Ashley Ives, James Fulcher 
# Purpose: Converts the output of TopPIC searches to an MSnSet object that stores label-free quantification data. Final data is on a relative log2-scale.

# LOCAL EDIT BEGIN setup
# ORIGINAL (commented out):
# REPLACEMENT:
# Run from the analysis directory; use packages installed by code/00_install.R.
.libPaths(c(".R-library", .libPaths()))
dir.create("data", showWarnings=FALSE)
dir.create("results", showWarnings=FALSE)
# LOCAL EDIT END setup
library(tidyverse)
library(TopPICR) 
library(MSnbase)
library(MSnSet.utils)
library(PNNL.DMS.utils)

# 1. Load TopPIC out files -----------------------------------------------------

# LOCAL EDIT BEGIN 1
# ORIGINAL (commented out):
# if("toppic_output_humanislet.RData" %in% list.files()){
#   load("toppic_output_humanislet.RData")
# }else{
#   toppic_output <- read_TopPIC_DMS(5570)
#   save(toppic_output, file = "toppic_output_humanislet.RData")
# }
# 
# REPLACEMENT:
# Read the deposited archives and assign the two tables used below.
archives <- c(prsms="TopPIC_Results_PrSMs.zip", ms1_features="TopPIC_Results_MS1_Features.zip")
for (kind in names(archives)) {
  dest <- file.path("source", kind)
  if (!dir.exists(dest)) unzip(file.path("source", archives[[kind]]), exdir=dest)
}
# PNNL.DMS.utils::get_results_for_single_job.dt, with local-file edits below.
# https://github.com/PNNL-Comp-Mass-Spec/PNNL.DMS.utils/blob/1a4d8ed74692b55958c1937c592c21d42a25a94c/R/PNNL_DMS_utils.R#L436
# LOCAL EDIT: Give the local copy a distinct name.
# get_results_for_single_job.dt <- function(pathToFile, fileNamePttrn, expected_multiple_files = FALSE)
read_local_result <- function(pathToFile, fileNamePttrn, expected_multiple_files = FALSE)
{
   pathToFile <- as.character(pathToFile)
   
   # LOCAL EDIT: pathToFile is already a local filename; skip file lookup.
#    if (.Platform$OS.type == "unix") {
#       pathToFile <- get_url_from_dir_and_file(pathToFile, fileNamePttrn)
#    } else if (.Platform$OS.type == "windows") {
#       local_folder <- pathToFile
#       pathToFile <- list.files(path=local_folder,
#                                pattern=fileNamePttrn,
#                                full.names=TRUE)
#    } else {
#       stop("Unknown OS type.")
#    }
#    
   # if(length(pathToFile) == 0){
   #   stop("can't find the results file")
   # }
   # if(length(pathToFile) > 1){
   #   stop("ambiguous results files")
   # }
   # 
   # results <- read_tsv(pathToFile, col_types=readr::cols(), progress=FALSE)
   # 
   # dataset <- strsplit(basename(pathToFile), split=fileNamePttrn)[[1]]
   # out <- data.table(Dataset=dataset, results)
   # return(out)
   
   # checks all the file number problems
   if (length(pathToFile) == 0) {
      stop("can't find the results file")
   }
   if (length(pathToFile) > 1 & !expected_multiple_files) {
      stop("ambiguous results files")
   }
   
   short_dataset_names <- unlist(strsplit(basename(pathToFile), 
                                          split = fileNamePttrn))
   
   read_fun <- read_tsv
   if(grepl(".csv$", pathToFile))
      read_fun <- read_csv
   
# LOCAL EDIT: Qualify the function imported inside the original package.
#    out <- llply(pathToFile,
   out <- plyr::llply(pathToFile, 
                read_fun,
                col_types = readr::cols(),
                guess_max = Inf,
                progress = FALSE) %>%
      #lapply(function(xi) { dplyr::select(xi, -one_of("Dataset"))}) %>%
      # map(dplyr::select, -any_of("Dataset")) %>%
      lapply(function(xi) dplyr::select(xi, -any_of("Dataset"))) %>% 
      setNames(short_dataset_names) %>%
      enframe(name = "Dataset") %>% 
      unnest(value) %>%
# LOCAL EDIT: Qualify the function imported inside the original package.
#       data.table()
      data.table::data.table()
   return(out)
}

# TopPICR::read_TopPIC_DMS, with only its input supplied locally.
# https://github.com/PNNL-Comp-Mass-Spec/TopPICR/blob/0acc4215bda381d313d85e17519625c4a680db58/R/read_toppic_DMS.R
# LOCAL EDIT: Accept the tables read above instead of a DMS package number.
# read_TopPIC_DMS <- function(data_package_num){
read_TopPIC_tables <- function(toppic_output){

  # LOCAL EDIT: Tables are supplied directly, so no database query is needed.
#   jobRecords <- get_job_records_by_dataset_package(data_package_num)
#   toppic_output <- get_results_for_multiple_jobs.dt(jobRecords, expected_multiple_files = TRUE)
# 
  # LOCAL EDIT: Qualify the package's internal version-detection function.
#   toppic_version <- determine_toppic_version(toppic_output)
  toppic_version <- TopPICR:::determine_toppic_version(toppic_output)

  ids <- bind_rows(toppic_output$`_TopPIC_PrSMs.txt`)

  ids <- ids %>%
    dplyr::mutate(
      `Feature apex` = `Feature apex time`,
      firstAA = `First residue`,
      lastAA = `Last residue`,
      AccMap = `#Protein hits`,
      mz = (`Precursor mass` + Charge * 1.007276466621) / Charge,
      Gene = sub(".*GN=(\\S+).*", "\\1", `Protein description`),
      UniProtAcc = `Protein accession`,
      isDecoy = grepl("^DECOY", `Protein accession`))

  # annotation type. Note this is UniProt specific
  ids <- ids %>%
    dplyr::mutate(
      AnnType = dplyr::case_when(grepl("^(DECOY_)?sp\\|[^-]*-\\d+\\|.*",
                                       `Protein accession`) ~ "VarSplic",
                                 grepl("^(DECOY_)?tr.*",
                                       `Protein accession`) ~ "TrEMBL",
                                 grepl("^(DECOY_)?sp\\|[^-]*\\|.*",
                                       `Protein accession`) ~ "SwissProt",
                                 TRUE ~ NA_character_))

  # adding cleanSeq
  ids <- ids %>%
    dplyr::mutate(
      cleanSeq = gsub("\\[.+?\\]", "", Proteoform),
      cleanSeq = gsub("\\(|\\)", "", cleanSeq),
      cleanSeq = sub("^[A-Z]?\\.(.*)\\.[A-Z]?", "\\1", cleanSeq))


  # If there is not a special amino acid change the NA to -. This is
  # necessary otherwise special amino acids get incorrectly copied in the
  # next step.
  ids <- ids %>%
    dplyr::mutate(
      `Special amino acids` = dplyr::case_when(
        is.na(`Special amino acids`) ~ "-",
        TRUE ~ as.character(`Special amino acids`)))



  # Extract just the UniProt accession. This is the second element (when
  # splitting by |) of the `Protein accession` variable output by TopPIC.
  ids <- ids %>%
    dplyr::mutate(UniProtAcc = sub("[^|]*\\|([^|]*)\\|[^|]*","\\1", UniProtAcc))


  # Only keep variables we need throughout TopPICR.
  ids <- ids %>%
    dplyr::select(
      Dataset,
      `Scan(s)`,
      `Retention time`,
      `Feature apex`,
      Charge,
      mz,
      `Precursor mass`,
      `Adjusted precursor mass`,
      `Feature intensity`,
      `E-value`,
      `#unexpected modifications`,
      AccMap,
      MIScore,
      Proteoform,
      `Special amino acids`,
      `Protein accession`,
      `Protein description`,
      # `First residue`,
      # `Last residue`,
      firstAA,
      lastAA,
      `Proteoform mass`,
      `Proteoform-level Q-value`,
      UniProtAcc,
      Gene,
      AnnType,
      cleanSeq,
      isDecoy)

  # Fill in missing values for all accessions that match a given sequence.
  # This step is necessary because TopPIC only includes information for
  # the `Protein accession` and `Protein description` variables when a
  # proteoform matches multiple sequences. All other variables are left
  # blank.


  cols_to_fill <- c("Scan(s)", "Retention time", "Feature apex", "Charge",
                    "mz", "Precursor mass", "Adjusted precursor mass",
                    "Feature intensity", "E-value",
                    "#unexpected modifications",
                    "AccMap", "MIScore", "Proteoform", "Proteoform mass",
                    "Proteoform-level Q-value", "cleanSeq")

  ids <- ids %>%
    tidyr::fill(all_of(cols_to_fill), .direction = "down")

  # ids <- ids %>%
  #    tidyr::fill(`Scan(s)`:Proteoform, .direction = "down") %>%
  #    tidyr::fill(`Proteoform mass`, .direction = "down") %>%
  #    tidyr::fill(`Proteoform-level Q-value`, .direction = "down") %>%
  #    tidyr::fill(`cleanSeq`, .direction = "down")


  # Taking care of feature data
  feat <- bind_rows(toppic_output$`_ms1.feature`)
  # Only keep the MS1 variables we use throughout TopPICR.
  if(toppic_version == "1.7.0" || "Monoisotopic_mass" %in% colnames(feat)){
    feat <- feat %>%
      dplyr::rename(Mass = Monoisotopic_mass,
                    Time_apex = Apex_time)
  }
  feat <- feat %>%
    dplyr::select(Dataset, Mass, Intensity, Time_apex)


  return(list(ms2identifications = ids, ms1features = feat))
}



folders <- c(`_TopPIC_PrSMs.txt`="source/prsms", `_ms1.feature`="source/ms1_features")
public_tables <- lapply(names(folders), function(suffix) {
  files <- sort(list.files(folders[[suffix]], pattern=paste0(suffix, "$"), full.names=TRUE))
  stopifnot(length(files) == 36L)
  data.table::rbindlist(lapply(files, read_local_result, fileNamePttrn=suffix))
})
names(public_tables) <- names(folders)
toppic_output <- read_TopPIC_tables(public_tables)
ids <- toppic_output$ms2identifications
feat <- toppic_output$ms1features

# LOCAL EDIT END 1
# remove NA annotations. We can't handle them at the FDR filter tuning step
# Those are non-uniprot IDs, like short ORFs and contaminants
ids <- ids %>%
  dplyr::filter(!is.na(AnnType))

ids <- ids %>%
  mutate(CV = as.character(str_sub(Dataset,-2,-1))) %>%
  mutate(Dataset = as.character(str_sub(Dataset,1,-5))) 
feat <- feat %>%
  mutate(CV = as.character(str_sub(Dataset,-2,-1))) %>%
  mutate(Dataset = as.character(str_sub(Dataset,1,-5))) 

# 2. Polish proteoform spectral matches prior to making MSnSet------------------ 

# Remove erroneous genes ---------------
x_err <- rm_false_gene(ids) 


# LOCAL EDIT BEGIN 2
# ORIGINAL (commented out):
# fst <- Biostrings::readAAStringSet(
#   file.path(r"(\\gigasax\DMS_FASTA_File_Archive\Dynamic\Forward\)",
#             "ID_008274_9D2E95FB.fasta"))
# 
# # assign the length of the total gene to proteoforms, needed for later plotting 
# add_protein_length <- function(x, fst_obj){
#   fst_df <- data.frame(`Protein accession` = names(fst_obj),
#                        protLength = BiocGenerics::width(fst_obj), 
#                        row.names = NULL, 
#                        check.names = FALSE) %>%
#     mutate(`Protein accession` = word(`Protein accession`))
#   x <- inner_join(x, fst_df)
#   return(x)
# }
# 
# x_err <- add_protein_length(x_err, fst)
# 
# REPLACEMENT:
# Skip protein-length annotation: the institutional FASTA file is unavailable here.
x_err$protLength <- NA_integer_

# LOCAL EDIT END 2
compute_fdr(x_err) #sanity check fdr 

x_aug <- x_err

# FDR control ------------------------------------------------------------------

# Find the E-value threshold. FDR is calculated at Gene level.
the_cutoff <- find_evalue_cutoff(x = x_aug,
                                 fdr_threshold = 0.01)

# Apply the actual FDR control/filter with the threshold values from above.
x_fdr <- apply_evalue_cutoff(x = x_aug,
                             e_vals = the_cutoff)

compute_fdr(x_fdr) #sanity check 

# Proteoform inference ---------------
x_ipf <- infer_prot(x_fdr)

x_inferred <- set_pf_level(x_ipf)

compute_fdr(x_ipf)

# Align retention time ---------------------------------------------------------

# Create the model that will be used to align each retention time. The model is
# created between a reference data set and all other data sets.
the_model <- form_model(
  x = x_inferred,
  ref_ds = find_ref_ds(x = x_inferred),
  control = loess.control(surface = "direct"), # Use direct to avoid NAs.
  span = 0.5,
  family = "symmetric")

# Align retention times according to the model created previously.
x_art <- align_rt(
  x = x_inferred,
  model = the_model,
  var_name = "Retention time"
)

# Recalibrate the mass ---------------

# Calculate the error between the `Precursor mass` and the `Adjusted precursor
# mass`. This acts as the model for recalibrating the mass. A reference data set
# is not used when calculating the mass error model.
x_error <- calc_error(x = x_art,
                      ref_ds = find_ref_ds(x = x_inferred))

x_error$rt_sd <- 300
# Recalibrate the mass according the the errors computed previously.
x_rcm <- recalibrate_mass(x = x_art,
                          errors = x_error,
                          var_name = "Precursor mass") 

# Cluster ----------------------------------------------------------------------
# Key step. But doesn't take too long.
x_cluster <- cluster(x = x_rcm,
                     errors = x_error,
                     method = "single",
                     height = 3,
                     min_size = 2)


# Group clusters ---------------------------------------------------------------
x_recluster <- create_pcg(x = x_cluster,
                          errors = x_error,
                          n_mme_sd = 5,
                          n_rt_sd = 3,
                          #ppm_cutoff = 2.1,
                          n_Da = 3)

# LOCAL EDIT BEGIN 3
# ORIGINAL (commented out):
# save(x_recluster, file="x_recluster_oct2024_int.RData")
# REPLACEMENT:
# Write generated objects under data/.
save(x_recluster, file="data/x_recluster_oct2024_int.RData")
# LOCAL EDIT END 3

x_meta <- create_mdata(x = x_recluster,
                       errors = x_error,
                       n_mme_sd = 5,
                       n_rt_sd = 3
) %>%
  mutate(PF = paste(Gene, pcGroup, sep = "_"))

# LOCAL EDIT BEGIN 4
# ORIGINAL (commented out):
# save(x_meta, file="x_meta.RData")
# REPLACEMENT:
# Write generated objects under data/.
save(x_meta, file="data/x_meta.RData")
# LOCAL EDIT END 4


# Align unidentified features --------------------------------------------------

# Align the unidentified feature retention times with the model created by the
# identified feature retention times.
feat_art <- align_rt(
  x = feat,
  model = the_model,
  var_name = "Time_apex")

# Recalibrate unidentified feature mass ----------------------------------------

# Recalibrate the unidentified feature masses with the model created by the
# identified feature masses.
feat_rcm <- recalibrate_mass(
  x = feat_art,
  errors = x_error,
  var_name = "Mass")


feat_rtv <- match_features(ms2 = x_recluster,
                           ms1 = feat_rcm,
                           errors = x_error,
                           n_mme_sd = 5,
                           n_rt_sd = 3
)


# 3. Create and save MSnSet object---------------------------------------------- 

x <- feat_rtv %>%
  mutate(feature_name = paste(Gene, pcGroup, CV, sep = "_"),
         sample_name = Dataset) %>%
  dplyr::select(-Dataset)

x_expr <- x %>%
  pivot_wider(id_cols = "feature_name",
              names_from = "sample_name",
              values_from = "Intensity") %>%
  as.data.frame() %>%
  {rownames(.) <- .$feature_name;.} %>%
  dplyr::select(-feature_name) %>%
  as.matrix()

x_feat <- x %>%
  group_by(Gene,pcGroup, CV, feature_name) %>%
  summarize(median_intensity = median(Intensity),
            count = n(),
            .groups = "keep") %>%
  ungroup() %>%
  left_join(x_meta, by=c("Gene","pcGroup")) %>%
  mutate(proteoform_id = paste(Gene, pcGroup, sep="_")) %>%
  as.data.frame() %>%
  {rownames(.) <- .$feature_name;.}

x_pheno <- x %>%
  separate(sample_name, into=c("TD", "islet", "Letter","Treated", "Time", "Biorep"), remove = FALSE) %>%
  distinct(sample_name, Letter, Treated,Time, Biorep) %>%
  as.data.frame() %>%
  mutate(Pair = case_when(Letter == "A" ~ "Pair1",
                          Letter == "H" ~ "Pair1",
                          Letter == "B" ~ "Pair2",
                          Letter == "D" ~ "Pair2",
                          Letter == "C" ~ "Pair3",
                          Letter == "N" ~ "Pair3",
                          Letter == "E" ~ "Pair4",
                          Letter == "J"~ "Pair4",
                          Letter == "F" ~ "Pair5",
                          Letter == "K" ~ "Pair5",
                          Letter == "G" ~ "Pair6",
                          Letter == "M" ~ "Pair6",
                          Letter == "I" ~ "Pair7",
                          Letter == "L" ~ "Pair7")) %>% #assign an arbitrary identifier for each donor, letters correspond to LCMS runs  
  {rownames(.) <- .$sample_name;.} 

m <- MSnSet(x_expr, x_feat[rownames(x_expr),], x_pheno[colnames(x_expr),])

# LOCAL EDIT BEGIN 6
# ORIGINAL (commented out):
# save(x_feat, file="featuredata_beforerollup.RData")
# REPLACEMENT:
# Write generated objects under data/.
save(x_feat, file="data/featuredata_beforerollup.RData")
# LOCAL EDIT END 6

# Rrollup of intensity data, this is creating one quantitation value per proteoform across three compensation voltages 
m <- rrollup(m, "proteoform_id", rollFun = "/", verbose = FALSE)

#recover proteoform info, it all get's lost in rrollup step 
x_meta <- x_meta %>%
  mutate(proteoform_id = paste(Gene, pcGroup, sep="_")) %>%
  as.data.frame() %>%
  {rownames(.) <- .$proteoform_id;.}
fData(m) <- x_meta[featureNames(m),]

# LOCAL EDIT BEGIN 7
# ORIGINAL (commented out):
# save(m, file="msnset_humanislet_int_notnormalized.RData")
# REPLACEMENT:
# Write generated objects under data/.
save(m, file="data/msnset_humanislet_int_notnormalized.RData")
# LOCAL EDIT END 7

mlc <- log2_zero_center(m)

# LOCAL EDIT BEGIN 8
# ORIGINAL (commented out):
# save(mlc, file="msnset_humanislet_int_log2center_oct2024_forTyler.RData")
# REPLACEMENT:
# Write generated objects under data/.
save(mlc, file="data/msnset_humanislet_int_log2center_oct2024_forTyler.RData")
# LOCAL EDIT END 8

# LOCAL EDIT BEGIN 9
# ORIGINAL (commented out):
# # 4. Annotate modifications/ unknown mass shifts from open mod search using UniMod database 
# 
# x <- fData(mlc)
# 
# #reformat Nterm acetyl from new version ([Acetyl]-aa to old version (aa)[Acetyl ])
# x <- x %>% 
#   mutate(Proteoform = case_when(str_detect(Proteoform, "\\[Acetyl]-") ~ paste0(substr(gsub(pattern = "\\[Acetyl]-", x = Proteoform, replacement = ""), 1,2), "(", substr(gsub(pattern = "\\[Acetyl]-", x = Proteoform, replacement = ""), 3,3), ")[Acetyl]", substr(gsub(pattern = "\\[Acetyl]-", x = Proteoform, replacement = ""),4,nchar(gsub(pattern = "\\[Acetyl]-", x = Proteoform, replacement = ""))))
#                                 , !str_detect(Proteoform, "\\[Acetyl]-") ~ Proteoform)) 
# 
# unimods <- TopPICR::create_mod_data(
#   mod_file = "TopPIC_Dynamic_Mods.txt",
#   mod_path = ".",
#   use_unimod = TRUE
# )
# 
# annotate_Nterm_acetyls <- function(x, nterm_tol = 3, acetyl_id = "Acetyl"){
#   
#   temp_mod_names <- "mods"
#   if("mod_names" %in% names(x$mods[[1]]))
#     temp_mod_names <- "mod_names"
#   
#   for(i in 1:nrow(x)){
#     if(x[i,"firstAA"] <= nterm_tol){
#       y <- x[i,"mods"][[1]]
#       if(length(y$mods) > 0){
#         if(y$mods[1] == acetyl_id & y$mods_left_border[1] == 1){
#           y$mods[1] <- paste("N-", acetyl_id, sep="")
#           y[[temp_mod_names]][1] <- y$mods[1]
#           posi_str <- map2_chr(y$mods_left_border, y$mods_right_border, paste, sep="-")
#           y$mods_str <- paste(map2_chr(y[[temp_mod_names]], posi_str, paste, sep="@"), collapse=", ")
#           x[i,"mods"][[1]] <- list(y)
#         }
#       }
#     }
#   }
#   return(x)
# }
# 
# 
# x <- x %>%
#   mutate(mods = map(Proteoform, TopPICR:::extract_mods))
# 
# mass_annotation_table <- TopPICR:::get_mass_annotation_table(x, unimods, 0.3, 0.1)
# 
# x <- x %>%
#   mutate(mods = map(mods, TopPICR:::annotate_masses, mass_annotation_table, matching_tol = .Machine$double.eps))
# 
# x <- annotate_Nterm_acetyls(as.data.frame(x), nterm_tol = 3, acetyl_id = "Acetyl")
# rownames(x) <- x$proteoform_id
# 
# fData(mlc) <- x
# 
# save(mlc, file= "msnset_humanislet_int_log2center_oct2024_forTyler_wmodanno.RData")
# 
# REPLACEMENT:
# Skip modification annotation: TopPIC_Dynamic_Mods.txt is unavailable here; no substitute input is constructed.
# LOCAL EDIT END 9
# LOCAL EDIT BEGIN exports
# ORIGINAL (commented out):
# REPLACEMENT:
# Export the original pre-log intensity object for our paired analysis.
write.csv(data.frame(feature_id=rownames(exprs(m)), exprs(m), check.names=FALSE),
          "data/intensities.csv", row.names=FALSE, na="")
write.csv(data.frame(feature_id=x_meta$proteoform_id, x_meta, check.names=FALSE),
          "data/feature_metadata.csv", row.names=FALSE, na="")
writeLines(capture.output(sessionInfo()), "results/preprocessing_sessionInfo.txt")
# LOCAL EDIT END exports
