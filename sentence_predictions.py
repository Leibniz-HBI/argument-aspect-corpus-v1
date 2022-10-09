import click
from pathlib import Path
from flair.data import DataPair
from flair.data import Sentence
from flair.models import SequenceTagger, TextClassifier, TextPairClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
from nervaluate import Evaluator


@click.group()
def cli():
    pass


@cli.command()
@click.argument('path', required=True)
def nervaluate(path):
    get_nervaluate_for_experiment(Path(path))


@cli.command()
@click.argument('path', required=True)
def report(path):
    get_sentence_performance_for_experiment(Path(path))


def prepare_datapair_data(sentence_file):
    """Reads file and prepares gold and prediction data.

    returns a list of gold labels for each sentence as well as a list of DataPairs for each sentence.
    This is what the DataPairClassifier needs as input for prediction
    """
    with open(sentence_file, 'r') as f:
        data = f.read()
    sentences_raw = data.split('\n\n')
    sentences_raw = [sentence for sentence in sentences_raw if sentence]  # remove empty lines
    sentence_chunks = [sentence.split('\n') for sentence in sentences_raw]
    sentences = []
    gold_labels = []
    for sentence_chunk in sentence_chunks:
        sentence_chunk = [chunk.split('\t') for chunk in sentence_chunk]
        sentence_pairs = [DataPair(Sentence(chunk[0]), Sentence(chunk[1])) for chunk in sentence_chunk]
        sentences.append(sentence_pairs)
        gold = {chunk[2] for chunk in sentence_chunk}
        if 'O' in gold:
            gold.remove('O')
        gold_labels.append(gold)
    return sentences, gold_labels


def get_sentence_labels_from_sentence_chunks(sentence, classifier):
    classifier.predict(sentence)
    return {pair.get_labels()[0].value for pair in sentence if pair.get_labels()}


def predict_sentences_from_chunks(sentences, gold_labels, model_path):
    classifier = TextPairClassifier.load(model_path / 'best-model.pt')
    predicted_labels = [get_sentence_labels_from_sentence_chunks(sentence,classifier) for sentence in sentences]
    return report_classification(gold_labels, predicted_labels, model_path)


def get_sentences_from_tokens(model_path):
    with open(model_path / 'test.tsv', 'r') as f:
        data = f.read()
    sentences = data.split('\n\n')
    sentences = [sentence for sentence in sentences if sentence]
    gold_labels = []
    predicted_labels = []
    for sentence in sentences:
        tokens = sentence.split('\n')
        tokens = [token.split(' ') for token in tokens]
        gold = {token[1] for token in tokens}
        if 'O' in gold:
            gold.remove('O')
        gold_labels.append(gold)
        pred = {token[2] for token in tokens}
        if 'O' in pred:
            pred.remove('O')
        predicted_labels.append(pred)
    return report_classification(gold_labels, predicted_labels, model_path)


def get_sentence_pair_performance(sentence_file, model_folder):
    perfomance_reports = []
    accuracies = []
    sentences, gold_labels = prepare_datapair_data(sentence_file)
    for directory in model_folder.iterdir():
        if directory.name.startswith('run'):
            report, accuracy = predict_sentences_from_chunks(sentences, gold_labels, directory)
            perfomance_reports.append(report)
            accuracies.append(accuracy)
    df_concat = pd.concat([pd.DataFrame(report).T for report in perfomance_reports])
    by_row_index = df_concat.groupby(df_concat.index)
    means = by_row_index.mean()
    means.to_csv(model_folder / 'sentence_classification_mean.csv')
    std = by_row_index.std()
    std.to_csv(model_folder / 'sentence_classification_std.csv')
    with open(model_folder / 'accuracy_report.txt', 'w') as f:
        f.write(f'accuracy mean: {np.mean(accuracies)}\n')
        f.write(f'accuracy std: {np.std(accuracies)}\n')


def get_token_performance(model_folder):
    perfomance_reports = []
    accuracies = []
    for directory in model_folder.iterdir():
        if directory.name.startswith('run'):
            report, accuracy = get_sentences_from_tokens(directory)
            perfomance_reports.append(report)
            accuracies.append(accuracy)
    df_concat = pd.concat([pd.DataFrame(report).T for report in perfomance_reports])
    by_row_index = df_concat.groupby(df_concat.index)
    means = by_row_index.mean()
    means.to_csv(model_folder / 'sentence_classification_mean.csv')
    std = by_row_index.std()
    std.to_csv(model_folder / 'sentence_classification_std.csv')
    with open(model_folder / 'accuracy_report.txt', 'w') as f:
        f.write(f'accuracy mean: {np.mean(accuracies)}\n')
        f.write(f'accuracy std: {np.std(accuracies)}\n')


def report_classification(gold_labels, predicted_labels, path):
    mlb = MultiLabelBinarizer()
    y_pred = mlb.fit_transform(predicted_labels)
    y_true = mlb.transform(gold_labels)
    accuracy = accuracy_score(y_true, y_pred)
    labels = mlb.classes_
    with open(path / 'sentence_classification.txt', 'w') as f:
        f.write(classification_report(y_pred=y_pred, y_true=y_true, target_names=labels))
    with open(path / 'sentence_accuracy', 'w') as acc_f:
        acc_f.write(str(accuracy))
    classification_report(y_pred=y_pred, y_true=y_true, target_names=labels)
    return classification_report(y_pred=y_pred, y_true=y_true, target_names=labels, output_dict=True), accuracy


def get_sentence_performance_for_experiment(experiment_dir):
    corpus_dir = experiment_dir / 'corpora'
    for corpus in corpus_dir.iterdir():
        models = corpus / 'models'
        for model_folder in models.iterdir():
            if corpus.name.startswith('conll'):
                get_token_performance(model_folder)
            if corpus.name.startswith('sentence'):
                get_sentence_pair_performance(corpus / 'sentence_gold_test.txt', model_folder)


def nervaluate_predictions(prediction_file):
    print(f'getting results for {prediction_file}')
    with open(prediction_file, 'r') as f:
        data = f.read()
    sentences = data.split('\n\n')
    sentences = [sentence for sentence in sentences if sentence]
    sentence_tokens = [sentence.split('\n') for sentence in sentences]
    true = ''
    pred = ''
    tags = set()
    for sentence in sentence_tokens:
        for token in sentence:
            token_parts = token.split(' ')
            word = token_parts[0]
            gold = token_parts[1]
            prediction = token_parts[2]
            tags.add(gold)
            tags.add(prediction)
            true += f'{word}\t{gold}\n'
            pred += f'{word}\t{prediction}\n'
    correct_tags = [tag[2:] for tag in tags]  # nervaluate cuts first to symbols due to expecting bioes tags
    evaluator = Evaluator(true, pred, tags=correct_tags, loader="conll")
    results, results_by_tag = evaluator.evaluate()
    print(results['ent_type'])
    return results['ent_type']  # we don't need other results right now


def nervaluate_model_folder(model_folder):
    result_list = []
    print(f'creating results for {model_folder}')
    for run in model_folder.iterdir():
        if run.name.startswith('run'):
            result_list.append(nervaluate_predictions(run / 'test.tsv'))
    runs_df = pd.DataFrame(result_list)
    runs_df.loc['mean'] = runs_df.mean()
    runs_df.loc['std'] = runs_df.std()
    runs_df.index.name = 'run_number'
    runs_df.to_csv(model_folder / 'nervalaute_results.csv')


def get_nervaluate_for_experiment(experiment_dir):
    corpus_dir = experiment_dir / 'corpora'
    for corpus in corpus_dir.iterdir():
        if corpus.name.startswith('conll'):
            models = corpus / 'models'
            for model_folder in models.iterdir():
                nervaluate_model_folder(model_folder)


if __name__ == '__main__':
    cli()
