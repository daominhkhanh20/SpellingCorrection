import string
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from autocorrection.utils import *

puncs = string.punctuation


# def check_valid(list_mistake):
#     for mistake in list_mistake:
#         spelling_word = mistake['text']
#         correct_words = mistake['suggest']
#         all_words = correct_words + [spelling_word]
#         for word in all_words:
#             if len(word.split())>1:
#                 return False
#     return True


def replace_special_punctuation(text):
    text = re.sub('\.+', '.', text)
    # text = re.sub(":",'.',text)
    return text


def preprocess_sentence(text):
    text, mark_replaces = mark_special_token(text)
    text = re.sub("[^\w\d]", " ", text)
    return re.sub("\s+", " ", text)


def truncate_data_frame(df):
    df['count'] = df.error_sentence.apply(lambda x: len(x.split()))
    df = df[df['count'] < 50]
    df = df[df['count'] > 5].reset_index(drop=True)
    return df


def calculate_label_error(df):
    label_errors = []
    modified = []
    n_error = 0
    for ori_sentence, error_sentence in zip(df.original_sentence, df.error_sentence):
        if len(ori_sentence.split()) == len(error_sentence.split()):
            temp = []
            ori_sentence = ori_sentence.split()
            error_sentence = error_sentence.split()
            exists = False
            for x, y in zip(ori_sentence, error_sentence):
                if x != y:
                    # print(x,y)
                    temp.append(1)
                    exists = True
                    n_error += 1
                else:
                    temp.append(0)
            if exists:
                modified.append(1)
            else:
                modified.append(0)
            label_errors.append(temp)
        else:
            print(ori_sentence)
            print('\n\n\n')
            print(error_sentence)
    df['label_errors'] = label_errors
    df['modified'] = modified
    df = df[df['modified'] == 1].reset_index(drop=True)
    df.drop(["modified"], axis=1, inplace=True)
    print(len(df))
    return df


def get_sentence(texts, mistakes):
    correct_sentences, error_sentences = [], []
    for idx,(text, all_mistake_sentence) in enumerate(zip(texts, mistakes)):
        satisfied = False
        error_sent = text
        number_char_addition = 0
        for i, mistake in enumerate(all_mistake_sentence):
            spell_word = mistake['text']
            words_suggest = mistake['suggest']
            word_replace = random.choice(words_suggest)
            if len(word_replace.split()) != len(spell_word.split()) or any(value in word_replace for value in puncs):
                break
            start_offset = int(mistake['start_offset'])
            index = start_offset + number_char_addition
            text = text[:index] + word_replace + text[index + len(spell_word):]
            number_char_addition += len(word_replace) - len(spell_word)
            if i == len(all_mistake_sentence) - 1:
                satisfied = True 
        if satisfied and len(sent_tokenize(error_sent)) == len(sent_tokenize(text)):
            correct_sentences.append(text)
            error_sentences.append(error_sent)
    print(len(error_sentences))
    return error_sentences, correct_sentences


def split_sentence(spelling_texts, correct_texts):
    sent_errors, sent_corrects = [], []
    for i in range(len(correct_texts)):
        text_error = spelling_texts[i]
        text_correct = correct_texts[i]
        text_error = sent_tokenize(text_error)
        text_correct = sent_tokenize(text_correct)
        if len(text_error) != len(text_correct):
            print(text_error)
            print(text_correct)
            print("Not equal")
        else:
            for e_sent, c_sent in zip(text_error, text_correct):
                if len(c_sent.split()) == len(e_sent.split()) and c_sent != e_sent:
                    if len(c_sent.split()) >= 20:
                        c_sent = re.split('[,.":;\n-]', c_sent)
                        e_sent = re.split('[,.":;\n-]', e_sent)
                        sent_corrects.extend(c_sent)
                        sent_errors.extend(e_sent)
                    else:
                        sent_errors.append(e_sent)
                        sent_corrects.append(c_sent)
    print(len(sent_errors))
    return sent_errors, sent_corrects


def make_dataframe(sent_errors, sent_corrects):
    df = {'original_sentence': sent_corrects, "error_sentence": sent_errors}
    df = pd.DataFrame(df)
    df = truncate_data_frame(df)
    df['original_sentence'] = df.original_sentence.apply(lambda x: preprocess_sentence(x))
    df['error_sentence'] = df.error_sentence.apply(lambda x: preprocess_sentence(x))
    mark_indexs = []
    for i, (x, y) in enumerate(zip(df.original_sentence, df.error_sentence)):
        if len(x.split()) != len(y.split()):
            mark_indexs.append(i)
    df = df.drop(mark_indexs)
    df = calculate_label_error(df)
    df.original_sentence = df.original_sentence.apply(lambda x: str(x).lower())
    df.error_sentence = df.error_sentence.apply(lambda x: str(x).lower())
    print(len(df))
    print(sum([sum(value) for value in df.label_errors]))
    df.to_csv("autocorrection/dataset/Datatest/data_test.csv", index=False)


def make_wiki_test():
    data = pd.read_json("autocorrection/dataset/Datatest/spelling_test.json", lines=True)
    # print(f"We have {len(data)} paragraph")
    spelling_texts, correct_texts = get_sentence(data.text.values.tolist(), data.mistakes.values.tolist())
    sent_errors, sent_corrects = split_sentence(spelling_texts, correct_texts)
    make_dataframe(sent_errors, sent_corrects)
