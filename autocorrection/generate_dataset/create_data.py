import os
import csv
from nltk.tokenize import sent_tokenize
from unicodedata import normalize as nl
import argparse
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from autocorrection.generate_dataset.generator import GeneratorDataset
from tqdm import tqdm
from transformers import AutoTokenizer
import re
from random import shuffle

tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word")

parser = argparse.ArgumentParser()

parser.add_argument('--file_text', type=str, required=True,
                    help='what is name for file text?')

parser.add_argument('--file_csv', type=str, required=True,
                    help='what is name for file csv?')

parser.add_argument('--n_sents', type=int, required=False, default=500000,
                    help='how many sentences do you want?')

arg = parser.parse_args()


def get_data():
    data = []
    catergories = ['Chinh tri Xa hoi', 'Phap luat', 'Kinh doanh', 'Van hoa', 'Suc khoe']
    path = '../dataset/VNTC/'
    print("START LOADING DATA")
    for folder in ['Test_Full', 'Train_Full']:
        path_temp = path + folder
        print(len(os.listdir(path_temp)))
        for catergory in os.listdir(path_temp):
            if catergory in catergories:
                for file_name in os.listdir(path_temp + '/' + catergory):
                    absolute_path = path_temp + '/' + catergory + '/' + file_name
                    with open(absolute_path, mode='r', encoding='utf-16') as file:
                        text = nl('NFC', file.read())
                        if len(text.split()) >= 4:
                            data.append(text)
    shuffle(data)
    print("LOAD DATA DONE")
    print(len(data))
    return data


def make_sentene(data, file_name):
    all_sentene = []
    file_text = open('../dataset/Data' + '/' + file_name, 'w')
    cnt_sentence = 0
    for text in tqdm(data):
        for line in text.splitlines():
            for sentence in sent_tokenize(line):
                size = len(sentence.split())
                flag = True
                for special_char in ['@', '#', '"', "'", '-', '<', '>', '\\', '=', '+', '%', '_']:
                    if special_char in sentence:
                        flag = False
                        break
                if flag and 3 < size < 40:
                    cnt_sentence += 1
                    sentence = re.sub('^([\w\d]\.)+', '', sentence)  # remove special word in the start sentence
                    all_sentene.append(sentence)
                    file_text.write(sentence + '\n')
    file_text.close()
    print(f"We have {cnt_sentence} sentences")
    return all_sentene


def generate_error(data, file_name):
    generator = GeneratorDataset()
    cnt = 0
    header = ['original_sentence', 'error_sentence', 'label_error', 'special_token', 'percent_error']
    with open('../dataset/Data' + '/' + file_name, mode='w') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(header)
        for sentence in tqdm(data):
            if cnt > arg.n_sents:
                break
            for p_err in [0.4, 0.6]:
                sentence = sentence.lower()
                original_sentence, sentence_error, onehot_labels, special_token, p_err = generator.add_noise(sentence,
                                                                                                             p_err)
                if original_sentence is not None and len(original_sentence.split()) == len(
                        sentence_error.split()) and len(tokenizer.encode(sentence_error)) < 512:
                    csv_writer.writerow(
                        [original_sentence.lower(), sentence_error.lower(), onehot_labels, special_token, p_err])
                    cnt += 1


if __name__ == "__main__":
    all_texts = get_data()
    data = make_sentene(all_texts, 'data.txt')

    # generate_error(data, 'data.csv')
