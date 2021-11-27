####################################################################################################
# Tom Curran
# Natural Language Processing Assignment 3
# November 26, 2021
# --------------------------------------------------------------------------------------------------
# Resources:
# 1. https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/
# 2. https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT_only_first_wordpiece.ipynb#scrollTo=IEnlUbgm8z3B
####################################################################################################

from tqdm import trange
from pprint import pprint
import pandas as pd
import numpy as np
import torch
import tensorflow as tf
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset, random_split
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification, BertTokenizer
from sklearn.model_selection import train_test_split
import itertools

####################################################################################################
# Preprocess Data
####################################################################################################
class preprocessor:

    def __init__(self, filepath):

        self._raw_data = self._open_raw_file(filepath)
        self._clean_data = self._clean_raw_data(self._raw_data)
        self._dataframe = self._create_dataframe_from_text(self._clean_data)
        self._sentence_df = self._create_sentence_df(self._dataframe)
        
    def _open_raw_file(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        return lines

    def _clean_raw_data(self, raw_data):
        clean_data = []

        for line in raw_data:
            clean_data.append(line.strip().rstrip().split('\t'))

        return clean_data

    def _create_dataframe_from_text(self, clean_data):

        abstract_num = 1
        for line in clean_data:

            if line[0] == '':
                abstract_num += 1
                line.insert(0, '0')

            id_num = str(int(abstract_num)) + '_' + str(int(line[0]))
            line.insert(0, id_num)
            line.insert(1, abstract_num)
            
        clean_data.insert(0, ['1_0', 1, 0, '', ''])
        cols = ['idx', 'abstract_idx', 'abstract_token_idx', 'token', 'tag']
        
        df = pd.DataFrame(clean_data, columns=cols)
        
        labels_to_ids = {j: i for i, j in enumerate(df.tag.unique())}
        df['tag_id_num'] = [str(labels_to_ids.get(label)) for label in df.tag]
        
        return df

    def get_raw_data(self):
        return self.raw_data

    def get_clean_data(self):
        return self._clean_data

    def print_clean_data(self):
        for line in self._clean_data:
            print(line)

    def get_dataframe(self):
        return self._dataframe

    def get_tag_frequencies(self, dataframe):
        return dataframe.tag.value_counts()

    def get_labels_to_ids(self):
        return self._labels_to_ids

    def get_ids_to_labels(self):
        return self._ids_to_labels

    def _create_sentence_df(self, dataframe):

        values = []
        data = dataframe.fillna(method='ffill')

        abstract_indexes = data['abstract_idx'].unique()
        def join_array(x): return ' '.join(x)

        sentences = data.groupby('abstract_idx')['token'].apply(join_array)

        labels = data.groupby('abstract_idx')['tag'].apply(join_array)
        
        label_id_nums = data.groupby('abstract_idx')['tag_id_num'].apply(join_array)

        sentence_df = pd.DataFrame({
            'sentence': sentences,
            'sentence_tags': labels,
            'sentence_tags_id_nums': label_id_nums
        }).drop_duplicates().reset_index()

        return sentence_df

    def get_sentence_df(self):
        return self._sentence_df


####################################################################################################
# Create Tokens based on BERT model and tensorize the data
####################################################################################################

class Berterizer(Dataset, preprocessor):

    def __init__(self, dataframe, tokenizer):
        
        self._dataframe = dataframe
        self.max_length = max(list(map(len, dataframe.sentence.str.split())))
        self._tokenizer = tokenizer
        
    def encode_sentence(self, sentence):
        
        encoded_sentence = self._tokenizer.encode_plus(
            sentence, 
            add_special_tokens=True, 
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            )
        
        return encoded_sentence


FILEPATH='./data/S21-gene-train.txt'
TOKENIZER = BertTokenizerFast.from_pretrained('bert-base-uncased')

sentence_df = preprocessor(FILEPATH).get_sentence_df()
token_df = preprocessor(FILEPATH).get_dataframe()
berterizer = Berterizer(sentence_df, TOKENIZER)


input_ids = []
sentences = []
attention_masks = []
token_label_ids = np.array(list(map(int, token_df.tag_id_num.values.tolist())))

for i in trange(0, len(sentence_df)):
    
    _sentence = sentence_df.sentence.iloc[i]
    
    encoded_sentence = berterizer.encode_sentence(sentence_df.sentence.iloc[i])
    _attention_mask = torch.transpose(encoded_sentence['attention_mask'], 0, 1)
    _input_ids = torch.transpose(encoded_sentence['input_ids'], 0, 1)

    input_ids.append(_input_ids)
    sentences.append(_sentence)
    attention_masks.append(_attention_mask)
    
input_id_tensor = torch.cat(input_ids, dim=0)
attention_mask_tensor = torch.cat(attention_masks, dim=0)
token_labels_tensor = torch.tensor([token_label_ids])

torch_data = TensorDataset(input_id_tensor, attention_mask_tensor, token_labels_tensor)

train_pct = .8
train_size = int(torch_data[0][0].shape[0] * train_pct)
test_size = torch_data[0][0].shape[0] - train_size

train_data, test_data = random_split(torch_data, [train_size, test_size])
print('{:>5,} training samples'.format(torch_train_size))
print('{:>5,} validation samples'.format(torch_test_size))