Please, refer to the README.md of the repository for additional instructions.


DivExplorer is implemented in python.


########### Setting the environment

We can firstly set an environment. In following, there are the instructions using conda.



# Create the environment
conda create -n divexplorer-exp python=3.6.10

# To activate the environment:
source activate divexplorer-exp


########### Install the dependencies

Once the env is activated, we install the dependencies. For example, we can use pip:

pip install -r ./requirements.txt


########### Run all experiments


We run ALL experiments, producing all the figures and tables reported in the paper with:

python run_experiments.py

The results are stored in the ./output folder. Specifically, we will find in ./output/figures all the figures (in pdf format) and in ./output/tables all the tables (in csv format) reported in the paper



########### Run specific experiments

We can also reproduce specific results. 

For example, we can run all the experiments associated with the COMPAS dataset with

```shell
python E01_compas.py

```

The script generates all the experiments associated with the COMPAS dataset. Specifically, it generates "table_1", "table_2", "table_3", "figure_1", "figure_2", "figure_3", "figure_5"of the paper.

