# ML_benchmark_pipeline
The pipeline to evaluate ML and GWAS feature selection tools using BacGWASim simulated genotype and phenotypes

## Requirements

Between parenthesis the versions the script was tested against:

```
  - snakemake=6.10.0
  - numpy=1.20.3
  - pandas=1.3.4
  - python=3.9.0
  - scikit-learn=1.0.1
  - scipy=1.7.1
  - seaborn=0.9.0
  - plink=1.90b6.21
  - gemma=0.98.3
  - pyseer=1.3.9
  - bcftools=1.14
  - matplotlib=3.4.3
  - yaml=0.2.5
  - xgboost=1.5.0
  - lightgbm=3.2.1
```

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

## benchmarking parameters

All the benchmarking parameters are included in `src/configfile.yaml` file and can be adjusted.
In case a parameter is specified as both command line argument and within configfile, The command line parameter will be used.

```
#input data(BacGWASim outputs) including simulated genomes, phenotypes and causal variants
##txt file including pathes to simulated genomes in .vcf(for GWAS tools)/.pickle(for ML tools) format
simulation_file_pathes: data/examples/results_BacGWASim_40_quant/sim_pickle_files.txt
##txt file including pathes to simulated phenotypes in .phen(for GWAS tools)/.pickle(for ML tools) format   
phenotype_file_pathes: data/examples/results_BacGWASim_40_quant/phen_pickle_files.txt   
##txt file including pathes to causal variant data   
causal_variant_file_pathes: data/examples/results_BacGWASim_40_quant/par.txt 
phenRep: 3  #Number of replicates 
method: ml    #'ml/gwas'
gwasModel: pyseer #pyseer/gemma
alpha: 1    #1(lasso)/0.0069(enet)
mlModel: lgbm       #lgbm/svc/lr/rf/xgb
cores: 3    #Number of cpu cores to be used for parallelization

#output data
outDir: data/examples/results_BacGWASim_40_quant/ml_benchmark_res
```