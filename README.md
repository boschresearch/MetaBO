# MetaBO - Meta-Learning Acquisition Functions for Transfer Learning in Bayesian Optimization
This is the source code accompanying the paper *Meta-Learning Acquisition Functions for Transfer Learning in Bayesian Optimization* by Volpp et al., ICLR 2020. The paper can be found [here](https://arxiv.org/abs/1904.02642). The code allows to reproduce the results from the paper and to train neural acquisition functions on new problems.

## Purpose of the project
This software is a research prototype, solely developed for and published as part of the publication cited above. It will neither be maintained nor monitored in any way.

## Installation
Clone this repository and run

```shell
conda env create -f environment.yml
conda activate metabo
```

to create and activate a new conda environment named "metabo" with all python packages required to run the experiments.

## Run the code 
We provide:
 - Scripts to reproduce the results presented in the paper. These scripts are named evaluate_metabo_<experiment_name>.py. They load pre-trained network weights stored in /metabo/iclr2020/<experiment_name> to reproduce the results without the need of re-training neural acquisition functions. To run these scripts, execute

```shell
python evaluate_metabo_<experiment_name>.py
```

 - Scripts to re-train the aforementioned neural acquisition functions. These scripts are named train_metabo_<experiment_name>.py. To run these scripts, execute

```shell
python train_metabo_<experiment_name>.py
```
 
## License 
"Meta-Learning Acquisition Functions for Transfer Learning in Bayesian Optimization" is open-sourced under the APGL-3.0 license. See the [LICENSE](LICENSE) file for details.
