# ML_benchmark_pipeline
The pipeline to evaluate ML and GWAS feature selection tools 
## Installation - Using the repository

MLbenchmark is build based on Snakemake and can be installed locally as following:

1.  Clone workflow into working directory

```
git clone https://github.com/Morteza-M-Saber/ML_benchmark_pipeline.
cd ML_benchmark_pipeline
```

2. Install dependencies:
   The easiest way to install ml_benchmark dependencies is through [`mamba`](https://github.com/mamba-org/mamba) (a modern alternative to `conda`):

```bash
conda install mamba=0.17.0 -c conda-forge
mamba env create --file environment.yml
```

3. Activate ml_benchmark environment

```bash
conda activate ml_benchmark
```

4. Edit config file, or use the command-line to pass arguments to MLbenchmark.

```
vim src/configML.yaml
```

5. Execute workflow

Example for ML model evaluation:
```
python main.py --snakefile src/benchmarkerML --config src/configfile.yaml --causal_variant_file_pathes data/examples/results_BacGWASim_40_quant/par.txt --phenotype_file_pathes data/examples/results_BacGWASim_40_quant/phen_pickle_files.txt  --simulation_file_pathes data/examples/results_BacGWASim_40_quant/sim_pickle_files.txt --method ml --mlModel lr --output output_directory

```

Example for GWAS model evaluation:
```
 python main.py --snakefile src/benchmarkerGWAS --config src/configfile.yaml --causal_variant_file_pathes data/examples/results_BacGWASim_40_quant/par.txt --phenotype_file_pathes data/examples/results_BacGWASim_40_quant/phen_phen_files.txt  --simulation_file_pathes data/examples/results_BacGWASim_40_quant/sim_vcf_files.txt --method gwas --gwasModel gemma --output output_directory_gemma

```