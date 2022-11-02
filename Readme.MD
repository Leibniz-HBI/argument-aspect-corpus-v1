# Dataset and experiments for the COLING2022 Argument Mining Workshop Paper Boundary Detection and Categorization of Argument Aspects via Supervised Learning

This repository contains code and data for the paper *Argument Mining Workshop
Paper Boundary Detection and Categorization of Argument Aspects via Supervised
Learning* presented during the 9th Workshop on Argument Mining at COLING2022.

## The Argument Aspect Corpus V1
In the datasets folder, you can find three files with connl formatted data for
each topic that we have described in the paper.
In the experiments folder are test confgurations for reproducing the papers
results. You can inspect the config files to see hyperparameters and the models
that we have used. This repository may also be a starting point to experiment with argument
aspect with other models or hyperparameters
You can replace models with other available models from huggingface.

The Corpus will also be made available on Zenodo.

## Installation
you need [pipenv](https://pipenv.pypa.io/en/latest/index.html) for dependency
installation. Then, run `pipenv sync` in order to create the environment.
**Note:** We needed a non-default pytorch version for compatability issues with our GPU, so there's a
custom source in the pipfile for pytorch. Familiarize yourself with your GPU
and how to install torch and cuda for it. The code needs a cuda-compatible GPU
in order to run.

## Run experiment and get results
Start a shell  in your virtual environment using `pipenv shell` before you use
any commands.

In the experiments folder, you find sub-folder for all experiments you need to
run in order to reproduce our results, with
a prefix for the coresponding dataset.
run the experiments using

`python run_experiment.py run experiments/<foldername>`

Experiments can take a long time, depending on your GPU.

**Note:** There's an issue with FLAIR regarding data pair evaluation, which we
fixed locally. You need to replace
`<your_home_directory>/.local/share/virtualenvs/argument-aspect-corpus-v1-XXXXXXXX/lib/python3.9/site-packages/flair/data.py`
with the file from this repository if you get an error saying: *AttributeError:
'DataPair' object has no attribute 'to_original_text'*


You can get evaluations of the models using

`python run_experiment.py results experiments/<foldername>`

For Sentence prediction and nervaluate use

`python sentence_predictions.py report experiments/<foldername>`

and

`python sentence_predictions.py nervaluate experiments/<foldername>`


## Citation

Please cite [our paper](https://aclanthology.org/2022.argmining-1.12/) when using this repository
