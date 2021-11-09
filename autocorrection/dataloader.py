from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from autocorrection.utils import *


class MyCollate:
    def __init__(self, pad_id: int, pad_label_error: int, is_bert: bool = False, is_train: bool = True):
        self.pad_id = pad_id
        self.pad_label_error = pad_label_error
        self.is_bert = is_bert
        self.is_train = is_train

    def __call__(self, batch):
        error_word_ids = [data['error_id'] for data in batch]
        data = {'error_ids': pad_sequence(error_word_ids, padding_value=self.pad_id, batch_first=True)}
        if self.is_train:
            label_errors = [data['label_error'] for data in batch]
            word_corrections = [data['word_correction'] for data in batch]
            data['word_corrections'] = pad_sequence(word_corrections, padding_value=self.pad_id, batch_first=True)
            data['label_errors'] = pad_sequence(label_errors, padding_value=self.pad_label_error, batch_first=True)

        if batch[0].get('char_error_id', None) is not None:
            char_error_ids = [data['char_error_id'] for data in batch]
            data['char_error_ids'] = pad_sequence(char_error_ids, padding_value=self.pad_id, batch_first=True)
            batch_splits = [data['batch_split'] for data in batch]
            data['batch_splits'] = batch_splits

        if self.is_bert:
            attention_masks = [data['attention_mask'] for data in batch]
            attention_masks = pad_sequence(attention_masks, padding_value=0, batch_first=True)
            data['attention_masks'] = attention_masks
            batch_splits = [data['batch_split'] for data in batch]
            data['batch_splits'] = batch_splits
        return data


class TransformerEncoderDataset(Dataset):
    def __init__(self, indexs,
                 token_ids=None,
                 data=None,
                 is_train: bool = True,
                 add_char_level: bool = False
                 ):
        self.indexs = indexs
        self.add_char_level = add_char_level
        self.is_train = is_train
        self.n_errors = 2
        self.mask_word_id = 0
        if self.is_train:
            self.token_ids = token_ids
            if self.add_char_level:
                self.n_chars = self.token_ids['n_chars']
                self.char_error_token_ids = self.token_ids['char_token_ids']
                self.batch_splits = self.token_ids['size_word']

            self.correct_token_ids = self.token_ids['correct_token_ids']
            self.n_words = self.token_ids['n_words']
            self.error_ids = self.token_ids['error_ids']
            self.label_errors = self.token_ids['label_errors']
        else:
            self.data = data
            self.error_sentence = self.data.error_sentence
            self.word_tokenizer = load_file_picke('Dataset/MaskLabel/word_tokenizer.pkl')
            self.error_ids = tokenizer_ids(self.error_sentence, self.word_tokenizer)
            if self.add_char_level:
                self.char_tokenizer = load_file_picke(
                    'Dataset/MaskLabel/char_tokenizer.pkl')
                self.char_error_token_ids = tokenizer_ids(self.error_sentence, self.char_tokenizer)
                self.batch_splits = get_size_word_in_sentence(self.error_sentence)

    def __len__(self):
        return len(self.indexs)

    def __getitem__(self, idx):
        if self.is_train:
            idx = self.indexs[idx]

        data = {'error_id': torch.tensor(self.error_ids[idx], dtype=torch.long)}
        if self.is_train:
            data['label_error'] = torch.tensor(self.label_errors[idx])
            data['word_correction'] = torch.tensor(self.correct_token_ids[idx])
        if self.add_char_level:
            data['char_error_id'] = torch.tensor(self.char_error_token_ids[idx], dtype=torch.long)
            data['batch_split'] = self.batch_splits[idx]

        return data


class PhoBertDataset(Dataset):
    def __init__(self, indexs,
                 token_ids=None,
                 data=None,
                 is_train: bool = True,
                 ):
        self.indexs = indexs
        self.n_errors = 2
        self.mask_token_id = 0
        self.is_train = is_train
        if is_train:
            self.token_ids = token_ids
            self.correct_token_ids = self.token_ids['correct_token_ids']
            self.n_words = self.token_ids['n_words']
            self.error_ids = self.token_ids['error_ids']
            self.label_errors = self.token_ids['label_errors']
            self.size_subwords = self.token_ids['size_subwords']
        else:
            self.data = data
            self.word_error_tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
            self.error_ids = tokenizer_ids(self.data.error_sentence, self.word_error_tokenizer)
            self.size_subwords = calculate_size_subword(self.data.error_sentence, self.word_error_tokenizer)

    def __len__(self):
        return len(self.indexs)

    def __getitem__(self, idx):
        if self.is_train:
            idx = self.indexs[idx]
        data = {'error_id': torch.tensor(self.error_ids[idx], dtype=torch.long)}
        data['attention_mask'] = torch.tensor([0] + [1] * (data['error_id'].size(0) - 2) + [0], dtype=torch.long)
        data['batch_split'] = self.size_subwords[idx]
        if self.is_train:
            data['label_error'] = torch.tensor(self.label_errors[idx])
            data['word_correction'] = torch.tensor(self.correct_token_ids[idx])
        return data
