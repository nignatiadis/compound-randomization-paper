# Ives (2025) paired reanalysis

## Sources

- MassIVE MSV000097810: [TopPIC_Results_PrSMs.zip](https://massive.ucsd.edu/ProteoSAFe/DownloadResultFile?file=f.MSV000097810%2Fquant%2FTopPIC_Results_PrSMs.zip&forceDownload=true) and [TopPIC_Results_MS1_Features.zip](https://massive.ucsd.edu/ProteoSAFe/DownloadResultFile?file=f.MSV000097810%2Fupdates%2F2025-05-06_alchemistmatt_c000c597%2Fquant%2FTopPIC_Results_MS1_Features.zip&forceDownload=true), saved under `source/`.
- `source/pmic70044-sup-0001-suppmat.xlsx`: [published supplementary results](https://pmc.ncbi.nlm.nih.gov/articles/PMC12716117/), used to check all 839 reported rows.
- `source/authors/`: unchanged [authors' scripts](https://github.com/ashleyives/top_down_islets_cytokine/tree/f66043a5a79cb8648ab9bb879b9d0682aa371361). Our R copies mark changes with `LOCAL EDIT` and retain replaced lines as comments. Installed packages are unmodified.

## Run

From this directory, with R 4.5.1 and Julia 1.10.12:

The committed `data/paired_differences_complete6.csv` contains normalized log₂
treated-minus-control differences for 286 proteoforms and six donors. To run
the Julia analysis directly:

```sh
julia --project=../.. -e 'using Pkg; Pkg.instantiate()'
julia code/04_analyze.jl
```

To regenerate the paired differences from the source files and check the
published supplementary results:

```sh
Rscript code/00_install.R
export R_LIBS_USER="$PWD/.R-library"

Rscript code/01_reconstruct.R      # authors' quantification
Rscript code/02_tests.R            # authors' analysis; export complete-case differences
Rscript code/03_validate.R         # compare with the published supplement
```

Julia writes rejection counts, individual results, both variance priors, and
B2M/GCG measurements and p-values to `results/`. In VS Code, inspect `Z`, `B2M`,
and `GCG` directly; `donors` gives the column order. The script also displays
variance-fit and p-value histograms.
