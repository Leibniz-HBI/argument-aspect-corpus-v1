import logging
from pathlib import Path
import random
from flair.data import Corpus
from flair.datasets import ColumnCorpus, ClassificationCorpus, DataPairCorpus
from flair.embeddings import TransformerWordEmbeddings, SentenceTransformerDocumentEmbeddings, TransformerDocumentEmbeddings
from flair.models import SequenceTagger, TextClassifier, TextPairClassifier
from flair.trainers import ModelTrainer
import click


@click.group()
def cli():
    pass


@cli.command()
@click.option('--file', required=True, help='path to a conll file')
@click.option('--train_size', default=0.7, help='fraction of train size. default is 0.7.')
@click.option('--dev_size', default=0.2, help='fraction of dev size. default is 0.2.')
@click.option('--seed', default=947518, help='RNG seed.')
@click.option('--remove-other/--no-remove-other', ' /-O', default=True, help='remove sentences labeled other, default ')
def split(file, style, train_size, dev_size, seed, bioes, remove_other):
    click.echo('Splitting dataset')
    create_split(file, style, train_size=train_size, dev_size=dev_size, seed=seed, bioes=bioes, remove_other=remove_other)


@cli.command()
@click.option('--path', required=True, help='folder name of corpus. Must be sub directory of ./corpora')
def train(path):
    click.echo('Training')
    train_aspect_model(path)


def write_set(path, x_set, conll=True):
    with open(path, 'w') as f:
        for sentence in x_set:
            f.write(sentence)
            if conll:
                f.write("\n\n")
            else:
                f.write("\n")
    logging.info(f'written set to {path}')


def create_split(file, conll, output_path, train_size=0.7, dev_size=0.2, seed=947518, remove_other=True):
    """create split files for train, test and dev.

    Splits will be stored in the corpora path in a sub folder which is named:
    `<path>_<train_size>`


    Args:
        path: path where CoNNL file is
        style: 'token' or 'chunk' for which file to load
        train_size: the ratio of the corpus that should be used for training
    """
    random.seed(seed)
    with open(file, 'r') as f:
        data = f.read()
        if conll:
            sentences = data.split('\n\n')  # list of all sentences
        else:
            sentences = data.split('\n')
        orig_size = len(sentences)
    if remove_other:
        sentences = [x for x in sentences if x and 'OTHER' not in x]
    else:
        sentences = [x for x in sentences if x]
    logging.info(f'found {len(sentences)} sentences in the corpus, from {orig_size} total sentences')
    random.shuffle(sentences)
    train_split = int(train_size * len(sentences))
    logging.info(f'train set has {train_split} samples')
    dev_split = train_split + int(dev_size * len(sentences))
    logging.info(f'dev and test sets have {dev_split} samples')
    train_set = sentences[:train_split]
    dev_set = sentences[train_split:dev_split]
    test_set = sentences[dev_split:]
    Path(output_path).mkdir(parents=True, exist_ok=True)
    if file.name.startswith('sentence'):  # so that no empty lines are in datapair datasets
        conll = False
        write_set(output_path / 'sentence_gold_test.txt', test_set, conll=True)  # because we need that for testing
    write_set(output_path / 'train.txt', train_set, conll=conll)
    write_set(output_path / 'dev.txt', dev_set, conll=conll)
    write_set(output_path / 'test.txt', test_set, conll=conll)


def read_conll_corpus(path):
    columns = {0: 'index', 1: 'text', 2: 'aspect', 3: 'whitespace_after'}
    corpus: Corpus = ColumnCorpus(
        path,
        columns,
        train_file='train.txt',
        test_file='test.txt',
        dev_file='dev.txt'
    )
    return corpus


def read_fasttext_corpus(path):
    corpus: Corpus = ClassificationCorpus(
        path,
        test_file='test.txt',
        dev_file='dev.txt',
        train_file='train.txt',
        label_type='aspect',
    )
    return corpus


def read_datapair_corpus(path):
    corpus: Corpus = DataPairCorpus(
        path,
        test_file='test.txt',
        dev_file='dev.txt',
        train_file='train.txt',
        label_type='aspect',
    )
    return corpus


def train_model(type, path, max_epochs=10, learning_rate=5.0e-6, model='roberta-large', run_number=0):
    if type == 'conll':
        train_aspect_model(path, max_epochs=max_epochs, learning_rate=learning_rate, model=model, run_number=run_number)
    if type == 'fasttext':
        train_sentence_label_classification(path, max_epochs=max_epochs, learning_rate=learning_rate, model=model, run_number=run_number)
    if type == 'sentence':
        train_sentence_pair_classification(path, max_epochs=max_epochs, learning_rate=learning_rate, model=model, run_number=run_number)


def train_aspect_model(path, max_epochs=10, learning_rate=5.0e-6, model='roberta-large', run_number=0):
    corpus = read_conll_corpus(path)
    label_dict = corpus.make_label_dictionary(label_type="aspect")
    label_dict.add_item("O")
    embeddings = TransformerWordEmbeddings(
        model=model,
        layers="-1",
        subtoken_pooling="first",
        fine_tune=True,
        use_context=False,  # Dokument-Kontext nicht mit verwenden
    )
    tagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=label_dict,
        tag_type='aspect',
        use_crf=False,
        use_rnn=False,
        reproject_embeddings=False,
    )
    trainer = ModelTrainer(tagger, corpus)
    trainer.fine_tune(
        path / f'models/{model.split("/")[-1]}{f"/run_{run_number}" if run_number else ""}',
        use_final_model_for_eval=False,
        learning_rate=learning_rate,
        mini_batch_size=16,
        max_epochs=max_epochs
    )


def train_sentence_label_classification(path, max_epochs=10, learning_rate=5.0e-6, model='roberta-large', run_number=0):
    corpus = read_fasttext_corpus(path)
    label_dict = corpus.make_label_dictionary(label_type="aspect")
    #  document_embeddings = SentenceTransformerDocumentEmbeddings(model)
    document_embeddings = TransformerDocumentEmbeddings(model, fine_tune=True)
    classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, label_type='aspect')
    trainer = ModelTrainer(classifier, corpus)
    trainer.fine_tune(
        path / f'models/{model}{f"/run_{run_number}" if run_number else ""}',
        use_final_model_for_eval=False,
        learning_rate=learning_rate,
        mini_batch_size=16,
        max_epochs=max_epochs
    )


def train_sentence_pair_classification(path, max_epochs=10, learning_rate=5.0e-6, model='roberta-large', run_number=0):
    corpus = read_datapair_corpus(path)
    label_dict = corpus.make_label_dictionary(label_type="aspect")
    document_embeddings = TransformerDocumentEmbeddings(model, fine_tune=True)
    classifier = TextPairClassifier(
        document_embeddings=document_embeddings,
        label_type="aspect",
        label_dictionary=label_dict
    )
    trainer = ModelTrainer(classifier, corpus)
    trainer.fine_tune(
        path / f'models/{model}{f"/run_{run_number}" if run_number else ""}',
        use_final_model_for_eval=False,
        learning_rate=learning_rate,
        mini_batch_size=16,
        max_epochs=max_epochs
    )


if __name__ == '__main__':
    cli()
