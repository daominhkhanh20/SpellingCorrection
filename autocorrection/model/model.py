import torch
from torch import Tensor
from torch import nn
from torch._C import dtype
from torch.nn.utils.rnn import pad_sequence
import math
from transformers import RobertaConfig, RobertaModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################
# Char Word Transformer Encoder Model#
######################################

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 256, dropout: float = 0.1, max_len: int = 400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(dim=1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(100000) / d_model))
        self.position_encoding = torch.zeros(max_len, d_model).to(device)
        self.position_encoding[:, 0::2] = torch.sin(position * div_term)
        self.position_encoding[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x: Tensor) -> Tensor:
        """x: shape [batch_size, seq_length, embedding_dim] --> return [batch_size, seq_length, embedding_dim]"""
        x += self.position_encoding[:x.size(1)]
        return self.dropout(x)


def generate_square_mask(sequence_size: int):
    mask = (torch.triu(torch.ones((sequence_size, sequence_size), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def generate_source_mask(src: Tensor, mask_token_id: int):
    src_mask = (src == mask_token_id)
    return src_mask


class CharEncoderTransformers(nn.Module):
    def __init__(self, n_chars: int, mask_token_id: int, d_model: int = 256, d_hid: int = 256, n_head: int = 4,
                 n_layers: int = 4,
                 dropout: float = 0.2):
        super(CharEncoderTransformers, self).__init__()
        self.position_encoding = PositionalEncoding(d_model, dropout, 512)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, n_layers)
        self.char_embedding = nn.Embedding(n_chars, d_model)
        self.d_model = d_model
        self.max_char = 50
        self.linear_char = nn.Linear(self.max_char * self.d_model, self.d_model)
        self.mask_token_id = mask_token_id
        self.init_weight()

    def init_weight(self):
        init_range = 0.1
        self.char_embedding.weight.data.uniform_(-init_range, init_range)

    def merge_embedding(self, embeddings: Tensor, sequence_split, mode='avg') -> Tensor:
        """
        :param embeddings: chars embedding [batch_size, length_seq, d_hid]
        :param sequence_split: number character for each word list[int]
        :param mode: calculate average or add embedding
        :return: [batch_size, num_words, embedding_dim]
        """
        original_sequence_split = sequence_split.copy()
        sequence_split = [value + 1 for value in sequence_split]  # plus space
        sequence_split[-1] -= 1  # remove for the last token
        embeddings = embeddings[:sum(sequence_split)]
        embeddings = torch.split(embeddings, sequence_split, dim=0)
        embeddings = [embedd[:-1, :] if i != (len(sequence_split) - 1) else embedd for i, embedd in
                      enumerate(embeddings)]

        if mode == 'avg':
            embeddings = pad_sequence(embeddings, padding_value=0, batch_first=True)  # n_word*max_length*d_hid
            seq_splits = torch.tensor(original_sequence_split).reshape(-1, 1).to(device)
            outs = torch.div(torch.sum(embeddings, dim=1), seq_splits)
        elif mode == 'add':
            embeddings = pad_sequence(embeddings, padding_value=0, batch_first=True)  # n_word*max_length*d_hid
            outs = torch.sum(embeddings, dim=1)
        elif mode == 'linear':
            embeddings =[
                torch.cat(
                    (
                        embedding_tensor.reshape(-1),
                        torch.tensor(
                            [0] * (self.max_char - embedding_tensor.size(0)) * self.d_model,
                            dtype=torch.long
                        ).to(device)
                    )
                )
                for embedding_tensor in embeddings
            ]
            embeddings = torch.stack(embeddings, dim=0)
            outs = self.linear_char(embeddings)
        else:
            raise Exception('Not Implemented')
        return outs

    def forward(self, src: Tensor,
                batch_splits,
                src_mask: Tensor = None,
                src_key_padding_mask: Tensor = None
                ) -> Tensor:
        """
        :param src: char token ids [batch_size, max_len(setence_batch)]
        :param batch_splits:
        :param src_mask:
        :param src_key_padding_mask: mask pad token
        :return: word embedding after combine from char embedding [batch_size*n_words*d_hid]
        """
        src_embeddings = self.char_embedding(src)  # batch_size * len_seq * embedding_dim
        src_embeddings = self.position_encoding(src_embeddings)
        if src_mask is None or src_mask.size(0) != src.size(1):
            src_mask = generate_square_mask(src.size(1))

        if src_key_padding_mask is None:
            src_key_padding_mask = generate_source_mask(src, self.mask_token_id)

        outputs = self.transformer_encoder(
            src_embeddings.transpose(0, 1),
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        ).transpose(0, 1)  # batch_size*len(sentence)*d_hid
        outputs = pad_sequence(
            [self.merge_embedding(embedding, sequence_split) for embedding, sequence_split in
             zip(outputs, batch_splits)],
            padding_value=0,
            batch_first=True
        )
        return outputs


class CharWordTransformerEncoding(nn.Module):
    def __init__(self, n_words: int, n_chars: int, n_label_errors: int,
                 mask_token_id: int,
                 use_detection_context: bool = True, d_model: int = 512, d_hid: int = 768,
                 n_head: int = 12, n_layers: int = 12, dropout: float = 0.2):
        super(CharWordTransformerEncoding, self).__init__()
        self.position_encoding = PositionalEncoding(d_model, dropout, 256)
        self.char_transformer_encoder = CharEncoderTransformers(n_chars, mask_token_id)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model + self.char_transformer_encoder.d_model, n_head, d_hid,
                                                        dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, n_layers)
        self.word_embedding = nn.Embedding(n_words, d_model)
        self.d_model = d_model
        self.mask_token_id = mask_token_id
        self.use_detection_context = use_detection_context
        self.fc1 = nn.Linear(d_hid, n_label_errors)
        if use_detection_context:
            self.softmax = nn.Softmax(dim=-1)
            self.linear_detection_context = nn.Linear(n_label_errors, 20)
            self.d_out_hid = d_hid + 20
        else:
            self.d_out_hid = d_hid
        self.fc2 = nn.Linear(self.d_out_hid, n_words)
        self.init_weight()

    def init_weight(self):
        init_range = 0.1
        self.word_embedding.weight.data.uniform_(-init_range, init_range)

    def forward(self, src_word_error_ids: Tensor,
                src_char_ids: Tensor,
                batch_splits,
                src_mask: Tensor = None,
                src_key_padding_mask: Tensor = None
                ):
        """
        :param src_word_error_ids: word token ids [batch_size, n_words]
        :param src_char_ids: char token ids [batch_size, seq_len]
        :param batch_splits:
        :param src_mask:
        :param src_key_padding_mask: mask pad token
        :return: detection outputs [batch_size * n_words * n_errors] and correction outputs [batch_size * n_words * n_words]
        """
        src_word_embeddings = self.word_embedding(src_word_error_ids)
        src_word_embeddings = self.position_encoding(src_word_embeddings)
        src_words_from_chars = self.char_transformer_encoder(src_char_ids,
                                                             batch_splits)  # batch_size*n_words*d_model_char
        src_word_embeddings = torch.cat((src_word_embeddings, src_words_from_chars),
                                        dim=-1)  # batch_size*n_words*(d_model_char+d_model_word)
        # if src_mask is None or src_mask.size(0) != src_word_error_ids.size(1):  # sequence_size
        #     src_mask = generate_square_mask(src_word_error_ids.size(1))

        if src_key_padding_mask is None:
            src_key_padding_mask = generate_source_mask(src_word_error_ids, self.mask_token_id)

        outputs = self.transformer_encoder(
            src_word_embeddings.transpose(0, 1),  # n_words * batch_size * hidden_size
            # mask=src_mask,  # n_words * n_words
            src_key_padding_mask=src_key_padding_mask  # batch_size * n_words * hidden_size
        ).transpose(0, 1)  # batch_size * n_words * d_hid
        detection_outputs = self.fc1(outputs)  # batch_size * n_words * n_errors
        if self.use_detection_context:
            detection_context = self.softmax(detection_outputs)
            detection_context = self.linear_detection_context(detection_context)  # batch_size * n_words * d_hid
            outputs = torch.cat((outputs, detection_context), dim=-1)  # batch_size * n_words * d_out_hid

        correction_outputs = self.fc2(outputs)
        return detection_outputs, correction_outputs


class PhoBertEncoder(nn.Module):
    def __init__(self, n_words: int, n_labels_error: int,
                 fine_tuned: bool = False, use_detection_context: bool = False):
        super(PhoBertEncoder, self).__init__()
        self.bert_config = RobertaConfig.from_pretrained('vinai/bartpho-word', return_dict=True,
                                                         output_hidden_states=True)
        self.bert = RobertaModel.from_pretrained('vinai/bartpho-word', config=self.bert_config)
        self.d_hid = self.bert.config.hidden_size
        self.detection = nn.Linear(self.d_hid, n_labels_error)
        self.use_detection_context = use_detection_context
        if self.use_detection_context:
            self.detection_context_layer = nn.Sequential(
                nn.Softmax(dim=-1),
                nn.Linear(n_labels_error,self.d_hid)
            )
        self.max_n_subword = 30
        self.linear_subword_embedding = nn.Linear(self.max_n_subword * self.d_hid, self.d_hid)
        self.fine_tuned = fine_tuned
        self.correction = nn.Linear(self.d_hid, n_words)
        self.is_freeze_model()

    def is_freeze_model(self):
        for child in self.bert.children():
            for param in child.parameters():
                param.requires_grad = self.fine_tuned

    def merge_embedding(self, sequence_embedding: Tensor, sequence_split, mode='avg'):
        sequence_embedding = sequence_embedding[1: sum(sequence_split) + 1]  # batch_size*seq_length*hidden_size
        embeddings = torch.split(sequence_embedding, sequence_split, dim=0)
        word_embeddings = pad_sequence(
            embeddings,
            padding_value=0,
            batch_first=True
        )
        if mode == 'avg':
            temp = torch.tensor(sequence_split).reshape(-1, 1).to(device)
            outputs = torch.div(torch.sum(word_embeddings, dim=1), temp)
        elif mode == 'add':
            outputs = torch.sum(word_embeddings, dim=1)
        elif mode == 'linear':
            embeddings = [
                torch.cat((
                    embedding_subword_tensor.reshape(-1),
                    torch.tensor([0] * (self.max_n_subword -embedding_subword_tensor.size(0)) * self.d_hid)
                ))
                for embedding_subword_tensor in embeddings
            ]
            embeddings = torch.stack(embeddings, dim=0)
            outputs = self.linear_subword_embedding(embeddings)
        else:
            raise Exception('Not Implemented')
        return outputs

    def forward(self, input_ids: Tensor,
                attention_mask: Tensor,
                batch_splits,
                token_type_ids: Tensor = None
                ):
        outputs = self.bert(input_ids, attention_mask)
        hidden_states = outputs.hidden_states
        stack_hidden_state = torch.stack(
                                    [hidden_states[-1], hidden_states[-2], hidden_states[-3], hidden_states[-4]],
                                    dim=0
                                )
        mean_hidden_state = torch.mean(stack_hidden_state, dim=0)
        outputs = pad_sequence(
            [self.merge_embedding(sequence_embedding, sequence_split) for sequence_embedding, sequence_split in
             zip(mean_hidden_state, batch_splits)],
            padding_value=0,
            batch_first=True
        )
        detection_outputs = self.detection(outputs)
        if self.use_detection_context:
            detection_context = self.detection_context_layer(detection_outputs)  # batch_size*seq_length*hidden_size
            outputs = outputs + detection_context

        correction_outputs = self.correction(outputs)
        return detection_outputs, correction_outputs


class GRUDetection(nn.Module):
    def __init__(self, n_words: int, n_labels_error: int, d_model: int = 512, d_hid: int = 512, n_layers: int = 2,
                 bidirectional: bool = True, dropout: float = 0.2):
        super(GRUDetection, self).__init__()
        self.word_embedding = nn.Embedding(n_words, d_model)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_hid,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.output_dim = d_hid * 2 if bidirectional else d_hid
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.output_dim, n_labels_error)

    def forward(self, src):
        """
        :param src: word error token ids
        :return: probability for each error type [batch_size, n_words, n_errors] and word error embedding [batch_size * seq_len * d_model]
        """
        embeddings = self.word_embedding(src)
        outputs, _ = self.gru(embeddings)  # batch_size*seq_length*(2*hidden_size)
        outputs = self.dropout(self.linear(outputs))
        return self.softmax(outputs), embeddings


class MaskedSoftBert(nn.Module):
    def __init__(self, n_words: int, n_labels_error: int, mask_token_id: int,
                 n_head: int = 8, n_layer_attn: int = 6, d_model: int = 512, d_hid: int = 512,
                 n_layers_gru: int = 2, bidirectional: bool = True, dropout: float = 0.2):

        super(MaskedSoftBert, self).__init__()
        self.detection = GRUDetection(n_words=n_words,
                                      n_labels_error=n_labels_error,
                                      d_model=d_model,
                                      n_layers=n_layers_gru,
                                      bidirectional=bidirectional
                                      )
        self.position_encoding = PositionalEncoding(d_model, dropout, max_len=128)
        self.encoder_layer = nn.TransformerEncoderLayer(d_hid, n_head, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, n_layer_attn)
        self.mask_token_id = mask_token_id
        self.correction = nn.Linear(d_hid, n_words)

    def forward(self, src: Tensor,
                src_mask: Tensor = None,
                src_key_padding_mask: Tensor = None
                ):
        """
        :param src: word error token ids
        :param batch_splits:
        :param src_mask:
        :param src_key_padding_mask:
        :return: detection outputs [batch_size * n_words * n_errors] and correction outputs [batch_size * n_words * n_words]
        """
        mask_embedding = self.detection.word_embedding(torch.tensor([[self.mask_token_id]]).to(device))
        detection_outputs, embeddings = self.detection(src)
        prob_correct_word = detection_outputs[:, :, 0].unsqueeze(2)  # batch_size * n_words *1
        # embedding: batch_size * n_words * d_model
        soft_mask_embedding = prob_correct_word * embeddings + (1 - prob_correct_word) * mask_embedding
        soft_mask_embedding = self.position_encoding(soft_mask_embedding)
        if src_mask is None or src_mask.size(0) != src.size(1):
            src_mask = generate_square_mask(src.size(1))

        if src_key_padding_mask is None:
            src_key_padding_mask = generate_source_mask(src, self.mask_token_id)
        outputs = self.transformer_encoder(
            soft_mask_embedding.transpose(0, 1),  # seq_len * batch_size * hidden_size
            mask=src_mask,  # seq_len * seq_len
            src_key_padding_mask=src_key_padding_mask  # batch_size*seq_len
        ).transpose(0, 1)  # batch_size * n_words * d_hid
        outputs +=embeddings
        correction_outputs = self.correction(outputs)
        return detection_outputs, correction_outputs
