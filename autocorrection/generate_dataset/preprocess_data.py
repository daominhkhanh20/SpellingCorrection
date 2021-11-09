from tensorflow.keras.preprocessing.text import Tokenizer
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from autocorrection.utils import *
from transformers import AutoTokenizer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path_csv', type=str, default='../dataset/Data/data.csv')

parser.add_argument('--num_valid_and_train', type=int, required=True,
                    help="Number sentence for train and validation")

parser.add_argument('--training_data_percent', type=float, required=True,
                    help="define ratio for training data")

parser.add_argument('--path_save_data', type=str, default='../dataset/Data/')

arg = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word")



def mask_word_correction(data, correct_token_id):
    label_errors = data.label_error.values.tolist()
    # replace the correct word = mask token id
    for i in range(len(label_errors)):
        for j in range(len(label_errors[i])):
            if label_errors[i][j] == 0:
                correct_token_id[i][j] = 0
    return correct_token_id


def calculate_ids(error_sentence):
    size_subwords = []
    input_ids = []
    size_words = []

    for sentence in error_sentence:
        temp = split_token(tokenizer, sentence)
        size_subwords.append(temp)
        input_ids.append(tokenizer.encode(sentence))
        size_words.append(get_size_word_in_sentence(sentence))
    return size_subwords, input_ids, size_words


def save_data(correct_token_id,
              error_token_id,
              char_error_token_id,
              input_ids,
              label_errors,
              size_subwords,
              size_words,
              n_words,
              n_chars,
              ):
    save_picke_file(
        arg.path_save_data + 'phobert.pkl',
        data={
            'correct_token_ids': correct_token_id,
            'error_ids': input_ids,
            'label_errors': label_errors,
            'size_subwords': size_subwords,
            'n_words': n_words
        }
    )

    save_picke_file(
        arg.path_save_data + 'trans.pkl',
        data={
            'correct_token_ids': correct_token_id,
            'error_ids': error_token_id,
            'label_errors': label_errors,
            'n_words': n_words,
        }
    )

    save_picke_file(
        arg.path_save_data + 'char_trans.pkl',
        data={
            'correct_token_ids': correct_token_id,
            'error_ids': error_token_id,
            'label_errors': label_errors,
            'char_token_ids': char_error_token_id,
            'size_word': size_words,
            'n_words': n_words,
            'n_chars': n_chars
        }
    )
    print("Save done")


def tokenize_sentence(data):
    ori_sentence = data.original_sentence.values.tolist()
    error_sentence = data.error_sentence.values.tolist()
    all_texts = ori_sentence + error_sentence

    print("Start tokenize")
    word_tokenizer = Tokenizer(oov_token='<unk>', lower=True)
    word_tokenizer.fit_on_texts(all_texts)
    char_tokenizer = Tokenizer(char_level=True, oov_token='<unk>', lower=True)
    char_tokenizer.fit_on_texts(all_texts)

    word_tokenizer.index_word[0] = '<mask>'
    word_tokenizer.word_index['<mask>'] = 0
    char_tokenizer.word_index['<mask>'] = 0
    char_tokenizer.index_word[0] = '<mask>'
    print("End tokenize")
    save_picke_file(arg.path_save_data, word_tokenizer)
    save_picke_file(arg.path_save_data, word_tokenizer)
    return word_tokenizer, char_tokenizer


def calculate_token_id(word_tokenizer, char_tokenizer, data):
    print("Start calculate token id")
    ori_sentence = data.original_sentence.values.tolist()
    error_sentence = data.error_sentence.values.tolist()
    correct_token_id = [word_tokenizer.texts_to_sequences([text])[0] for text in ori_sentence]
    error_token_id = [word_tokenizer.texts_to_sequences([text])[0] for text in error_sentence]
    char_error_token_id = [char_tokenizer.texts_to_sequences([text])[0] for text in error_sentence]
    correct_token_id = mask_word_correction(data, correct_token_id)
    print("End calculate token id")
    return correct_token_id, error_token_id, char_error_token_id


def main():
    data = pd.read_csv(arg.path_csv)[:arg.num_valid_and_train]
    word_tokenizer, char_tokenizer = tokenize_sentence(data)
    correct_token_id, error_token_id, char_error_token_id = calculate_token_id(word_tokenizer, char_tokenizer, data)
    size_subwords, input_ids, size_words = calculate_ids(data.error_sentence)
    save_data(correct_token_id, error_token_id,
              char_tokenizer, input_ids, data.label_error,
              size_words, size_words,
              len(word_tokenizer.word_index),
              len(char_tokenizer.word_index)
              )


if __name__ == '__main__':
    main()
