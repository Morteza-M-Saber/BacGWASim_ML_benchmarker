#!/bin/bash
#SBATCH --account=def-shapiro
#SBATCH --ntasks=1               # number of MPI processes
#SBATCH --cpus-per-task=32
#SBATCH --mem=128000M    # 128000 for full memory; default unit is megabytes, mem-per-cpu=2048M
#SBATCH --time=00-12:00           # time (DD-HH:MM)
#SBATCH --mail-user=morteza.mahmoudisaber@gmail.com
#SBATCH --mail-type=ALL

~/.local/easybuild/software/2017/Core/miniconda3/4.3.27/envs/MLBenchmarker/bin/python benchmarker.py --snakefile benchmarkerML --config configML.yaml \
                      --method ml --mlModel lr \
                      --outDir res/3000_f10k/lr
~/.local/easybuild/software/2017/Core/miniconda3/4.3.27/envs/MLBenchmarker/bin/python benchmarker.py --snakefile benchmarkerML --config configML.yaml \
                      --method ml --mlModel svc \
                      --outDir res/3000_f10k/svm
~/.local/easybuild/software/2017/Core/miniconda3/4.3.27/envs/MLBenchmarker/bin/python benchmarker.py --snakefile benchmarkerML --config configML.yaml \
                      --method ml --mlModel rf \
                      --outDir res/3000_f10k/rf
~/.local/easybuild/software/2017/Core/miniconda3/4.3.27/envs/MLBenchmarker/bin/python benchmarker.py --snakefile benchmarkerML --config configML.yaml \
                      --method ml --mlModel lgbm \
                      --outDir res/3000_f10k/lgbm
~/.local/easybuild/software/2017/Core/miniconda3/4.3.27/envs/MLBenchmarker/bin/python benchmarker.py --snakefile benchmarkerML --config configML.yaml \
                      --method ml --mlModel xgb \
                      --outDir res/3000_f10k/xgb
~/.local/easybuild/software/2017/Core/miniconda3/4.3.27/envs/MLBenchmarker/bin/python benchmarker.py --snakefile benchmarkerGWAS --config configGWAS.yaml \
                      --method gwas --gwasModel pyseer --alpha 1 \
                      --outDir res/3000_f10k/pyseerLasso
~/.local/easybuild/software/2017/Core/miniconda3/4.3.27/envs/MLBenchmarker/bin/python benchmarker.py --snakefile benchmarkerGWAS --config configGWAS.yaml \
                      --method gwas --gwasModel pyseer --alpha 0.0069 \
                      --outDir res/3000_f10k/pyseerEnet
~/.local/easybuild/software/2017/Core/miniconda3/4.3.27/envs/MLBenchmarker/bin/python benchmarker.py --snakefile benchmarkerGWAS --config configGWAS.yaml \
                      --method gwas --gwasModel gemma \
                      --outDir res/3000_f10k/gemma



