import click
import sys
import re
import yaml
from pathlib import Path
import pandas as pd
import classification


@click.group()
def cli():
    pass


@cli.command()
@click.argument('path', required=True)
def run(path):
    click.echo('running experiments')
    run_all_experiments(path)


@cli.command()
@click.argument('path', required=True)
def results(path):
    click.echo('Creating result tables')
    create_results(path)


def run_all_experiments(path):
    path = Path(path)
    with open(path / 'config.yml', 'r') as cfg:
        config = yaml.safe_load(cfg)
    corpora = path / 'corpora'
    for corpusdir in corpora.iterdir():
        if corpusdir.is_dir():
            for model in config['models']:
                if config['number_of_runs'] == 1:
                    classification.train_model(
                        corpusdir.name.split('_')[0],
                        corpusdir,
                        max_epochs=config['max_epochs'],
                        learning_rate=config['learning_rate'],
                        model=model
                    )
                else:
                    for run in range(1, config['number_of_runs'] + 1):
                        classification.train_model(
                            corpusdir.name.split('_')[0],
                            corpusdir,
                            max_epochs=config['max_epochs'],
                            learning_rate=config['learning_rate'],
                            model=model,
                            run_number=run
                        )


def get_results_from_log(logfile):
    run_results = {}
    with open(logfile, 'r') as log:
        table_section = False
        for line in log:
            if not table_section:
                if line.startswith('- F-score (micro) '):
                    run_results[('f_micro', 'value')] = float(line.strip().split('micro) ')[1])
                if line.startswith('- F-score (macro) '):
                    run_results[('f_macro', 'value')] = float(line.strip().split('macro) ')[1])
                if line.startswith('- Accuracy '):
                    run_results[('accuracy', 'value')] = float(line.strip().split('Accuracy ')[1])
                if line.startswith('By class:'):
                    table_section = True
            else:
                line = re.split(r'[ ]{2,}', line.lstrip())
                if line[0] and not line[0][0].isnumeric() and line[0] != 'precision':  # skip last line, header and empty lines
                    class_dict = {
                        (line[0], 'precision'): float(line[1]),
                        (line[0], 'recall'): float(line[2]),
                        (line[0], 'f-score'): float(line[3]),
                    }
                    run_results.update(class_dict)
    return run_results


def get_corpus_results(corpus):
    models = corpus / 'models'
    corpus_results = {}
    corpus_name = corpus.name.removeprefix('conll_gold_')
    for model in models.iterdir():
        all_runs = [
            get_results_from_log(run / 'training.log') for run in model.iterdir() if run.name.startswith('run')
        ]
        indx = pd.MultiIndex.from_tuples(all_runs[0].keys())
        runs_df = pd.DataFrame(all_runs, columns=indx, index=range(1, len(all_runs) + 1))
        runs_df.loc['mean'] = runs_df.mean()
        runs_df.loc['std'] = runs_df.std()
        runs_df.index.name = 'run_number'
        runs_df.to_csv(model / 'training_results.csv')
        corpus_results[(corpus_name, model.name)] = runs_df
    return corpus_results


def create_results(path):
    corpora = Path(path + '/corpora')
    results = {}
    for corpus in corpora.iterdir():
        if corpus.is_dir():
            results.update(get_corpus_results(corpus))
    summary = {}
    for key, df in results.items():
        summary[key + ('mean',)] = df.loc['mean']
        summary[key + ('std',)] = df.loc['std']
    summary_df = pd.DataFrame(summary).T
    # reorder so that the these colums are first
    first_cols = [('f_micro', 'value'), ('f_macro', 'value'), ('accuracy', 'value')]
    summary_df = summary_df[first_cols + [col for col in summary_df.columns if col not in first_cols]]
    summary_df.index.set_names(['labeling style', 'language model', 'value'], inplace=True)
    summary_df.to_csv(Path(path) / 'result_summary.csv')
    return results


if __name__ == '__main__':
    cli()
