# Methylation analysis

Ten Illumina 450k arrays from donors M28, M29 and M30 in
[GSE49667](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE49667).
`01_prepare.R` downloads the [workflow data](https://ndownloader.figshare.com/files/7896205),
quantile-normalizes the arrays, and removes probes failing detection (P ≥ .01
in any array), probes with SNPs, and cross-reactive probes. It retains 439,918 CpGs.
These are the preprocessing steps in our earlier `source/methylation_data.qmd`.

Julia fits an additive donor/cell-type model (4 residual df), testing four
contrasts: rTreg − naive, activated naive − naive, activated Treg − rTreg,
and activated Treg − activated naive. Ordinary limma uses residual variances;
our score uses an orbit-fitted quantile-limma prior (5 df, quantiles .25/.75,
median scale). Both 5% and 10% levels are reported; DDR uses τ = α/10.
Methods labelled `corrected` apply BH at α/1.93.

## Run

From this directory, with R and Julia 1.10.12:

```sh
# Install dependencies once.
Rscript -e 'if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager", repos="https://cloud.r-project.org"); BiocManager::install(c("minfi", "IlluminaHumanMethylation450kanno.ilmn12.hg19", "IlluminaHumanMethylation450kmanifest"))'
julia --project=../.. -e 'using Pkg; Pkg.instantiate()'

Rscript code/01_prepare.R
julia code/02_analyze.jl
julia code/03_tables.jl
```

An existing raw-data directory can be passed to `01_prepare.R` as its only
argument. Julia scripts also run from VS Code using `@__DIR__` paths.
Preprocessing was checked with R 4.5.1 and minfi 1.46.0.

`data/` contains normalized arrays and sample metadata. `results/` contains
counts, probe-level decisions, one `variance_priors.csv`, and LaTeX table rows
in the contrast order above. Each SeqStep+ donor swap is reported separately,
in donor order M28, M29, M30 where available. The GZ table row uses the subgroup
fixing M28 for naive activation and the full group otherwise.

`code/calibration.jl` contains exact shortcuts for the large number of probes;
it avoids computing every pair of orbit tails. Data and results are generated.
