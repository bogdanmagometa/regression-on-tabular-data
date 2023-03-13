# Regression on the tabular data

## Project structure
- analysis.ipynb - Jupyter notebook with analysis
- modeling.py - Python module that trains the regression on the train data and predicts on the test features
- predictions.csv - file with model predictions for internship_hidden_test.csv (produced by modeling.py)
- README.md
- requirements.txt - Python libraries for installing with pip

## Prerequisites
- You need to have conda installed

## Environment Setup
Run the following in the project root directory to setup the conda environment:
```bash
$ conda create -n reg_on_tab_data python=3.9    # create new virtual env
$ conda activate reg_on_tab_data                # activate environment in terminal
$ conda install jupyter                         # install jupyter + notebook
$ pip install -r requirements.txt               # install python libraries used in analysis.ipynb and modeling.py
```

## Usage

In order to train the model on `internship_train.csv` and write the predictions for 
`internship_hidden_test.csv` into `predictions.csv`, run the following in the
project root directory:
```bash
$ python3 modeling.py
```

Run the following in the project root directory to start jupyter notebook server, then open the analysis.ipynb
```bash
$ jupyter notebook
```
