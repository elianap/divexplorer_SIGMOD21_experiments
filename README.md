# DivExplorer - SIGMOD 2021 - Reproducibility report

This repository contains the code to run and reproduce the experiments present in our **[SIGMOD 2021 paper](https://dl.acm.org/doi/abs/10.1145/3448016.3457284)** "Looking for Trouble: Analyzing Classifier Behavior via Pattern Divergence".

If you are interested in using DivExplorer, please refer to **[this repository](https://github.com/elianap/divexplorer)** and the corresponding **[PyPi package](https://github.com/elianap/divexplorer)**.

## Setting the environment

DivExplorer is implemented in python.

We can firstly set an environment. In following, there are the instructions using conda or vend.

Using conda

```shell
# Create the environment

conda create -n divexplorer-exp python=3.6.10

# To activate the environment:

source activate divexplorer-exp
```


Using venv

```shell
# Create a virtualenv folder
mkdir -p ~/venv-environments/divexplorer-exp

# Create a new virtualenv in that folder
python3 -m venv ~/venv-environments/divexplorer-exp

# Activate the virtualenv
source ~/venv-environments/divexplorer-exp/bin/activate
```

Once the env is activated, we install the dependencies.


# Install deps
```shell
pip install -r ./requirements.txt
```


## Project structure

    ├── README.md             <- The top-level README for developers using this project.
    │
    ├── datasets              <- Raw data from third party sources used in this project.
    │  ├── dataset.txt        <- File info
    │  └── processed          <- Processed datasets.
    │
    ├── divexplorer           <- The source code of the DivExplorer algorithm
    │
    ├── requirements.txt      <- The requirements file for reproducing the analysis environment
    │                         
    ├── discretize.py         <- Utils to discretize data
    ├── import_datasets.py    <- Utils to import the data
    ├── utils_print.py        <- Utils to output and format the data
    │
    ├── NB1_compas.ipynb      <- Notebook using the COMPAS dataset
    ├── NB2_adult.ipynb       <- Notebook using the COMPAS dataset
    │
    ├── prepare_venv.txt      <- Instrunction to set up the environment via venv
    │                         
    ├── prepare_conda.txt     <- Instrunction to set up the environment via conda
    │                         
    ├── run_experiments.py    <- Run ALL experiments, producing all the figures and tables of the paper
    │                         
    ├── E01_compas.py         <- Run all the experiments for the COMPAS dataset
    │                         
    ├── E02_adult.py          <- Run all the experiments for the adult dataset
    │                         
    ├── E03_artificial.py     <- Run all the experiments for the artificial dataset
    │                         
    ├── E04_redundancy.py     <- Run experiments with redundancy pruning
    │                         
    ├── E05a_compute_performance.py   <- Compute performance results
    │                         
    ├── E05b_plot_performance.py      <- Visualize performance results (require to run E05a first)
    │                         
    ├── E06_stats_dataset.py  <- Output the dataset statistics (Table 4)
    │                         
    └── E07_survey.py         <- Output the survey results and plot

## Running the experiments

### Running all experiments

We run ALL experiments, producing all the figures and tables reported in the paper via:

```shell
python run_experiments.py
```

The results are stored in the ./output folder. Specifically, we will find in ./output/figures all the figures (in pdf format) and in ./output/tables all the tables (in csv format) reported in the paper

### Running specific experiments

We can also reproduce specific results. 

```shell
python E0{exp-name}.py

```

#### Running COMPAS experiments

For example, we can run all the experiments associated with the COMPAS dataset with

```shell
python E01_compas.py

```

The script generates all the experiments associated with the COMPAS dataset. Specifically, it generates "table_1", "table_2", "table_3", "figure_1", "figure_2", "figure_3", "figure_5"of the paper.

#### Running adult experiments

To run all the experiments associated with the adult dataset:

```shell
python E02_adult.py

```

The script generates all the experiments associated with the adult dataset. Specifically, it generates "table_5", "table_6",  "figure_8", "figure_9", "figure_11" of the paper.

## Contributors

[Eliana Pastor][eliana], [Elena Baralis][elena] and [Luca de Alfaro][luca].



[eliana]: https://smartdata.polito.it/members/eliana-pastor/ "Eliana Pastor"

[elena]: https://smartdata.polito.it/members/elena-baralis/ "Elena Baralis"

[luca]: https://luca.dealfaro.com/ "Luca de Alfaro"
