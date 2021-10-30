from random import random
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from utils.paths import get_dataset_dir
from utils.utils import read_lines

class ReadabilityDataset(Dataset):

    def __init__(self, sentences, labels, tokenizer):

        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):

        return len(self.sentences)

    def __getitem__(self, idx):

        sentence = self.sentences[idx]
        label = self.labels[idx]
        
        label = torch.tensor(label, dtype=torch.long)
        
        return sentence, label

    def collate_fn(self, data):
        
        sentences = [_[0] for _ in data]
        labels = torch.stack([_[1] for _ in data], dim=0)
        
        encodings = self.tokenizer(
            sentences,
            truncation=True,
            padding=True,
            return_tensors='pt',
        )
        
        return {**encodings, "labels":labels}

class PredictDataset(Dataset):

    def __init__(self, sentences, tokenizer):

        self.sentences = sentences
        self.tokenizer = tokenizer

    def __len__(self):

        return len(self.sentences)

    def __getitem__(self, idx):

        return self.sentences[idx]

    def collate_fn(self, sentences):

        encodings = self.tokenizer(
            sentences,
            truncation=True,
            padding=True,
            return_tensors='pt',
        )

        return encodings

class NewselaPreprocessor(object):

    def __init__(self, random_seed):

        self.path = get_dataset_dir("newsela") / "newsela_articles_20150302.aligned.sents.txt"
        self.raw_lines = read_lines(self.path)
        self.sentences, self.labels = self.preprocessor()
        self.random_seed = random_seed

    def get_split(self):

        sentences_train, sentences_test, labels_train, labels_test = \
            train_test_split(self.sentences, self.labels, test_size=0.2, random_state=self.random_seed)
        sentences_valid, sentences_test, labels_valid, labels_test = \
            train_test_split(sentences_test, labels_test, test_size=0.5, random_state=self.random_seed)

        return sentences_train, sentences_valid, sentences_test, \
                labels_train, labels_valid , labels_test

    def get_labels(set):

        return

    def preprocessor(self):

        sentences = []
        labels = []
        
        for line in self.raw_lines:

            tabs = line.split('\t')
            level1, level2, sentence1, sentence2 = tabs[1][1], tabs[2][1], tabs[3], tabs[4]
            sentences.append(sentence1)
            sentences.append(sentence2)
            labels.append(int(level1))
            labels.append(int(level2))

        return sentences, labels

class PredictPreprocessor(object):

    def __init__(self, path):

        self.path = path
        self.raw_lines = read_lines(self.path)

    def preprocessor(self):

        sentences = []
        for line in self.raw_lines:
            sentence_pair = line.split('\t')
            sentences.extend(sentence_pair)

        return sentences
