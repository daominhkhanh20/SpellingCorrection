from autocorrection.model.model import *
import pandas as pd
import nltk
import re
import string
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from autocorrection.utils import mark_special_token

data = pd.read_json('Dataset/spelling_test.json', lines=True)
texts = data.text.values.tolist()
mistakes = data.mistakes.values.tolist()
puncs = string.punctuation


def replace_special_punctuation(text):
    text = re.sub('\.+', '.', text)
    # text = re.sub(":",'.',text)
    return text


def replace_spelling_words():
    spelling_texts = []
    correct_texts = []
    for text, list_mistake in zip(texts, mistakes):
        error_text = text
        stastified = False
        for i, mistake in enumerate(list_mistake):
            spelling_word = mistake['text']
            correct_words = mistake['suggest']
            if len(correct_words) > 1:
                word_replace = random.choice(correct_words)
                if len(word_replace) == 0:
                    # print(len(correct_words))
                    correct_words.remove(word_replace)
                    word_replace = random.choice(correct_words)
            else:
                word_replace = correct_words[0]

            if len(word_replace.split()) > 1:
                break
            start_offset = int(mistake['start_offset'])
            if any(value in spelling_word for value in puncs) or any(value in word_replace for value in puncs):
                # print(spelling_word)
                # print(word_replace)
                pass
            text = text[:start_offset] + word_replace + text[start_offset + len(spelling_word):]
            if i == len(list_mistake) - 1:
                stastified = True
        if stastified is True:
            if len(sent_tokenize(error_text)) == len(sent_tokenize(text)):
                spelling_texts.append(error_text)
                correct_texts.append(text)

    correct_texts = [replace_special_punctuation(text) for text in correct_texts]
    spelling_texts = [replace_special_punctuation(text) for text in spelling_texts]
    return correct_texts, spelling_texts


def preprocess_sentence(text):
    text, mark_replaces = mark_special_token(text)
    text = re.sub("[^\w\d]", " ", text)
    return re.sub("\s+", " ", text)


def extract_sentence_for_testing(correct_texts, spelling_texts):
    sent_errors, sent_corrects = [], []
    for i in range(len(correct_texts)):
        text_error = spelling_texts[i]
        text_correct = correct_texts[i]
        text_error = sent_tokenize(text_error)
        text_correct = sent_tokenize(text_correct)
        if len(text_error) != len(text_correct):
            # for i in range(min(len(text_error, text_corrrect))):
            #   if len(text_error[i])
            print("Not equal")
        else:
            for sent1, sent2 in zip(text_error, text_correct):
                if sent1 != sent2 and len(sent1.split()) == len(sent2.split()):
                    sent_errors.append(sent1)
                    sent_corrects.append(sent2)
    df = {'original_sentence': sent_corrects, "error_sentence": sent_errors}
    df = pd.DataFrame(df)
    df['count'] = df.error_sentence.apply(lambda x: len(x.split()))
    df = df[df['count'] > 3].reset_index(drop=True)
    return df


def main():
    correct_texts, spelling_texts = replace_spelling_words()
    df = extract_sentence_for_testing(correct_texts, spelling_texts)
    df['original_sentence'] = df.original_sentence.apply(lambda x: preprocess_sentence(x))
    df['error_sentence'] = df.error_sentence.apply(lambda x: preprocess_sentence(x))
    print(len(df))


if __name__ == "__main__":
    main()