from autocorrection.utils import *
from autocorrection.model.model import *
import numpy as np
from transformers import AutoTokenizer


class AutoCorrection:
    def __init__(self, model_name, test_for='70k', threshold_detection=0.7, threshold_correction=0.6):
        self.model_name = model_name
        self.threshold_correction = threshold_correction
        self.threshold_detection = threshold_detection
        self.word_tokenizer = load_file_picke(f'/content/Data/{test_for}/Mask{test_for}/word_tokenizer.pkl')
        self.char_tokenizer = load_file_picke(f'/content/Data/{test_for}/Mask{test_for}/char_tokenizer.pkl')
        self.phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word")
        if test_for == '70k':
            self.path_pretrained = '/content/Pretrained70k/model18.pth'
        elif test_for == '1400k':
            self.path_pretrained = '/content/Pretrained1400k/model8.pth'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(self.path_pretrained)

    def select_model(self):
        if self.model_name == 'phobert':
            model = PhoBertEncoder(n_words=len(self.word_tokenizer.word_index),
                                   n_labels_error=2
                                   ).to(self.device)
        elif self.model_name == 'SoftmaskBert':
            model = MaskedSoftBert(n_words=len(self.word_tokenizer.word_index),
                                   n_labels_error=2,
                                   mask_token_id=0).to(self.device)
        elif self.model_name == 'Charword':
            model = CharWordTransformerEncoding(n_words=len(self.word_tokenizer.word_index),
                                                n_chars=len(self.char_tokenizer.word_index),
                                                n_label_errors=2,
                                                mask_token_id=0).to(self.device)
        else:
            raise Exception('Not implement')

        return model

    def load_model(self, path):
        model = self.select_model()
        model_states = torch.load(path)
        model.load_state_dict(model_states['model'])
        model.eval()
        return model

    def make_inputs(self, sentence):
        data = []
        if self.model_name == 'phobert':
            input_ids = self.phobert_tokenizer.encode(sentence.lower())
            attention_mask = [0] + [1] * (len(input_ids) - 2) + [0]
            batch_splits = split_token(self.phobert_tokenizer, sentence)
            input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device).unsqueeze(dim=0)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(self.device).unsqueeze(dim=0)
            data = [input_ids, attention_mask, batch_splits]
        elif self.model_name == 'SoftmaskBert':
            word_ids = self.word_tokenizer.texts_to_sequences([sentence])[0]
            data = [torch.tensor(word_ids, dtype=torch.long).to(self.device).unsqueeze(dim=0)]
        elif self.model_name == 'Charword':
            word_ids = self.word_tokenizer.texts_to_sequences([sentence])[0]
            char_ids = self.char_tokenizer.texts_to_sequences([sentence])[0]
            batch_splits = get_size_word_in_sentence(sentence)
            words_ids = torch.tensor(word_ids, dtype=torch.long).unsqueeze(dim=0).to(self.device)
            char_ids = torch.tensor(char_ids, dtype=torch.long).unsqueeze(dim=0).to(self.device)
            data = [word_ids, char_ids, batch_splits]
        else:
            raise Exception('Not implemented')
        return data

    def restore_sentence(self, sentence, mark_replaces):
        tokens = word_tokenize(sentence)
        start_1, start_2, start_3 = 0, 0, 0
        for i in range(len(tokens)):
            if tokens[i] == 'numberic' and start_1 < len(mark_replaces['numberic']):
                tokens[i] = mark_replaces['numberic'][start_1]
                start_1 += 1
            elif tokens[i] == 'date' and start_2 < len(mark_replaces['date']):
                tokens[i] = mark_replaces['date'][start_2]
                start_2 += 1
            elif tokens[i] == 'specialw' and start_3 < len(mark_replaces['specialw']):
                tokens[i] = mark_replaces['specialw'][start_3]
                start_3 += 1
        sentence = " ".join(tokens)
        return sentence

    def argmax_tensor(self, detection_outputs, correction_outputs):
        detection_prob, detection_indexs = torch.max(detection_outputs, dim=-1)
        correction_prob, correction_indexs = torch.max(correction_outputs, dim=1)

        return detection_prob.detach().cpu().numpy(), \
               detection_indexs.detach().cpu().numpy(), \
               correction_prob.detach().cpu().numpy(), \
               correction_indexs.detach().cpu().numpy()

    def get_result(self, convert_word, sentence, detection_outputs, correction_outputs):
        words = sentence.split()
        detection_prob, detection_indexs, correction_prob, correction_indexs = \
            self.argmax_tensor(detection_outputs, correction_outputs)

        for index, value in enumerate(detection_prob):
            # if the probability for not the spell word is less then threshold, it's is spell word
            if value < self.threshold_detection and detection_indexs[index] == 0:
                detection_indexs[index] = 1

            if index in convert_word.keys():
                detection_indexs[index] = 1
                correction_indexs[index] = self.word_tokenizer.texts_to_sequences([convert_word[index]])[0][0]

        wrong_word_indexs = np.where(detection_indexs == 1)[0]
        word_predict = correction_indexs[wrong_word_indexs]
        word_predict = self.word_tokenizer.sequences_to_texts([word_predict])[0].split()
        print(detection_prob)
        print(detection_indexs)
        print(correction_prob)
        print(self.word_tokenizer.sequences_to_texts([correction_indexs]))
        if len(wrong_word_indexs) > 0:
            for index1, index2 in zip(wrong_word_indexs, range(len(word_predict))):
                # if a word is out of vocabulary, then the probability for word prediction need greater than a
                # threshold,else predict with normal
                if correction_prob[index1] > self.threshold_correction:
                    words[index1] = word_predict[index2]
        return " ".join(words), detection_indexs.tolist(), correction_indexs.tolist()

    def forward(self, original_sentence):
        convert_word = {}
        words = original_sentence.split()
        for idx, word in enumerate(words):
            word_norm = chuan_hoa_dau_tu_tieng_viet(word)
            if word != word_norm:
                convert_word[idx] = word_norm
                words[idx] = word_norm
        original_sentence = " ".join(words)
        data = self.make_inputs(original_sentence)
        detection_outputs, correction_outputs = self.model(*data)
        detection_outputs, correction_outputs = torch.softmax(detection_outputs, dim=-1), torch.softmax(
            correction_outputs, dim=-1)
        print(detection_outputs)
        sentence, detection_predict, correction_predict = self.get_result(convert_word, original_sentence,
                                                                          detection_outputs.squeeze(dim=0),
                                                                          correction_outputs.squeeze(dim=0))
        # print(detection_predict)
        # print(correction_predict)
        return sentence, detection_predict, correction_predict

    def correction(self, sentence):
        pairs = preprocess_sentences(sentence)
        results = ""
        for original_sentence, mark in pairs:
            sentence, _, _ = self.forward(original_sentence)
            if len(mark) > 0:
                sentence = self.restore_sentence(sentence, mark)
            results += sentence + '.'
        return results
