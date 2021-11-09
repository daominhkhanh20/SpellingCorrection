import pickle
import ast
from nltk import sent_tokenize, word_tokenize
import pandas as pd
import random
import json
import re
from collections import defaultdict
import torch
from torch import Tensor


def read_data(path):
    if 'csv' in path:
        data = pd.read_csv(path)
    elif '.json' in path:
        data = pd.read_json(path, lines=True)
    else:
        raise Exception('Not implemented')
    return data


def get_size_word_in_sentence(sentence: str):
    return [len(word) for word in sentence.split()]


def save_picke_file(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_file_picke(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def split_token(tokenizer, sentence: str):
    """
    Calculate number token for each word
    Example:
    tokenize("Xiin chàoo ngày mới") ==> ['Xi@@', 'in', 'ch@@', 'à@@', 'oo', 'ngày', 'mới']
    ==> result: [2,3,1,1]
    """

    list_words = sentence.split()
    tokens = tokenizer.tokenize(sentence)
    start = 0
    result = []
    size = len(list_words)
    for i in range(size):
        word = list_words[i]
        if i < size - 1:
            str_temp = ""
            start_temp = start
            while start < len(tokens) and str_temp != word:
                index = tokens[start].find('@')
                str_temp += tokens[start][:index] if index != -1 else tokens[start]
                start += 1
            result.append(start - start_temp)

        else:
            result.append(len(tokens) - start)
    assert sum(result) == len(tokens), "Number tokens not equal"
    return result


def calculate_size_subword(sentences, tokenizer):
    result = []
    for sentence in sentences:
        result.append(split_token(tokenizer, sentence))
    return result


def convert_string(string_list: str):
    """
    Convert string list to list
    "[1,2,3,4,5]" ==> [1,2,3,4,5]
    """
    return ast.literal_eval(string_list)


# def make_loader(dataset, batch_size: int, pad_id, pad_label_id):
#     return DataLoader(
#         dataset=dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=2,
#         collate_fn=MyCollate(pad_id, pad_label_id)
#     )

bang_nguyen_am = [['a', 'à', 'á', 'ả', 'ã', 'ạ'],
                  ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ'],
                  ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ'],
                  ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ'],
                  ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ'],
                  ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị'],
                  ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ'],
                  ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ'],
                  ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ'],
                  ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ'],
                  ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự'],
                  ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ']]

nguyen_am_to_ids = {}

for i in range(len(bang_nguyen_am)):
    for j in range(len(bang_nguyen_am[i])):
        nguyen_am_to_ids[bang_nguyen_am[i][j]] = (i, j)


def is_valid_vietnam_word(word):
    chars = list(word)
    nguyen_am_index = -1
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x != -1:
            if nguyen_am_index == -1:
                nguyen_am_index = index
            else:
                if index - nguyen_am_index != 1:
                    return False
                nguyen_am_index = index
    return True


def chuan_hoa_dau_tu_tieng_viet(word):
    if not is_valid_vietnam_word(word):
        return word

    chars = list(word)
    dau_cau = 0
    nguyen_am_index = []
    qu_or_gi = False
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x == -1:
            continue
        elif x == 9:  # check qu
            if index != 0 and chars[index - 1] == 'q':
                chars[index] = 'u'
                qu_or_gi = True
        elif x == 5:  # check gi
            if index != 0 and chars[index - 1] == 'g':
                chars[index] = 'i'
                qu_or_gi = True
        if y != 0:
            dau_cau = y
            chars[index] = bang_nguyen_am[x][0]
        if not qu_or_gi or index != 1:
            nguyen_am_index.append(index)
    if len(nguyen_am_index) < 2:
        if qu_or_gi:
            if len(chars) == 2:
                x, y = nguyen_am_to_ids.get(chars[1])
                chars[1] = bang_nguyen_am[x][dau_cau]
            else:
                x, y = nguyen_am_to_ids.get(chars[2], (-1, -1))
                if x != -1:
                    chars[2] = bang_nguyen_am[x][dau_cau]
                else:
                    chars[1] = bang_nguyen_am[5][dau_cau] if chars[1] == 'i' else bang_nguyen_am[9][dau_cau]
            return ''.join(chars)
        return word

    for index in nguyen_am_index:
        x, y = nguyen_am_to_ids[chars[index]]
        if x == 4 or x == 8:  # ê, ơ
            chars[index] = bang_nguyen_am[x][dau_cau]
            return ''.join(chars)

    if len(nguyen_am_index) == 2:
        if nguyen_am_index[-1] == len(chars) - 1:
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            chars[nguyen_am_index[0]] = bang_nguyen_am[x][dau_cau]
        else:
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
    else:
        x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
        chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
    return ''.join(chars)


def tokenizer_ids(data, tokenizer):
    ids = []
    for sentence in data:
        ids.append(tokenizer.texts_to_sequences([sentence]))
    return ids


def get_label(predict: Tensor):
    predict = torch.softmax(predict, dim=-1)
    predict = torch.argmax(predict, dim=-1)
    return predict.reshape(-1)


def train_val_splits(total_sample: int, training_data_percent: float):
    idxs = [i for i in range(total_sample)]
    random.shuffle(idxs)
    train_size = int(total_sample * training_data_percent)
    train_idxs = idxs[:train_size]
    val_idxs = idxs[train_size:]
    return train_idxs, val_idxs


def is_number(self, token):
    if token.isnumeric():
        return True
    return bool(re.match('(\d+[\.,])+\d', token))


def is_number(token):
    if token.isnumeric():
        return True
    return bool(re.match('(\d+[\.,])+\d', token))


def is_date(token):
    return bool(re.match('(\d+[-.\/])+\d+', token))


def is_special_token(token):
    return bool(re.match(
        '([a-z0-9A-Z_ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ]+[\+\*\^\@\#\.\&\/-])+[a-z0-9A-Z_ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ]+',
        token))


def mark_special_token(sentence):
    tokens = word_tokenize(sentence)
    index_special = defaultdict(list)
    for i in range(len(tokens)):
        if is_number(tokens[i]):
            index_special['numberic'].append(tokens[i])
            tokens[i] = 'numberic'
        elif is_date(tokens[i]):
            index_special['date'].append(tokens[i])
            tokens[i] = 'date'
        elif is_special_token(tokens[i]):
            index_special['specialw'].append(tokens[i])  # mark differ 'special' word
            tokens[i] = 'specialw'
    return " ".join(tokens), index_special


def replace_special_tokens(text):
    text = re.sub("\.+", ".", text)
    # text = re.sub("[:]", ".", text)
    return text


def preprocess_sentences(sents):
    sentences = sent_tokenize(sents)
    with open('autocorrection/dataset/phrase_reduce.json', 'r') as file:
        phrase_reduce = json.load(file)
    results = []
    for sentence in sentences:
        tokens = sentence.split()
        for i in range(len(tokens)):
            if tokens[i] in phrase_reduce.keys():
                tokens[i] = phrase_reduce[tokens[i]]
        sentence = " ".join(tokens)
        sentence, mark_replaces = mark_special_token(sentence)
        sentence = re.sub('[^\w\d]', " ", sentence)
        sentence = " ".join(sentence.split(" "))
        results.append([sentence.lower(), mark_replaces])
    return results
