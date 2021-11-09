import time
from collections import defaultdict

from torch import optim
from torch.utils.data import DataLoader, Dataset
from autocorrection.dataloader import *
from autocorrection.model.model import * 
from autocorrection.utils import *
from sklearn.metrics import classification_report

class ModelTrainer:
    def __init__(self, model_name, n_samples,
                 training_data_percent: float,
                 lr,
                 path_save_model=None,
                 n_epochs=100,
                 is_split_indexs=False,
                 lam: float = 0.5,
                 penalty_value = 0.2,
                 use_detection_context=False,
                 is_continuous_train=False,
                 path_pretrain_model=None,
                 is_transformer=False,
                 add_char_level=False,
                 is_bert=False,
                 fine_tuned=False,
                 batch_size=128):

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_samples = n_samples
        self.lam = lam
        self.penalty_value = penalty_value
        self.is_split_indexs = is_split_indexs
        self.is_continuous_train = is_continuous_train
        self.path_pretrain_model = path_pretrain_model
        self.fine_tuned = fine_tuned
        self.use_detection_context = use_detection_context
        self.add_char_level = add_char_level
        self.is_transformer = is_transformer
        self.is_bert = is_bert
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.token_ids = load_file_picke(f'../input/spellingcorrection1400k/NoMask1000k/{model_name}.pkl')
        if not self.is_split_indexs:
            self.train_idxs, self.val_idxs = train_val_splits(self.n_samples, training_data_percent)
        else:
            self.train_idxs = load_file_picke('../input/spellingcorrection1400k/Mask1000k/train_indexs.pkl')
            self.val_idxs = load_file_picke('../input/spellingcorrection1400k/Mask1000k/val_indexs.pkl')

        self.train_loader, self.val_loader = self.make_loader()
        self.select_model(model_name)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        if self.is_continuous_train:
            print("LOADING")
            model_state = torch.load(self.path_pretrain_model)
            self.model.load_state_dict(model_state['model'])
            self.optimizer.load_state_dict(model_state['optimizer'])
            print("LOADED DONE")
        self.loss_detection = nn.CrossEntropyLoss(weight=torch.tensor([self.penalty_value, 1- self.penalty_value]).to(device))
        self.loss_correction = nn.CrossEntropyLoss(ignore_index=self.pad_id)
        self.epoch_temp = 0

    def select_model(self, model_name: str):
        if model_name == "phobert":
            self.model = PhoBertEncoder(
                n_words=self.train_dataset.n_words,
                n_labels_error=2,
                fine_tuned=self.fine_tuned,
                use_detection_context=self.use_detection_context
            ).to(self.device)

        elif model_name == "trans":
            self.model = MaskedSoftBert(
                n_words=self.train_dataset.n_words,
                n_labels_error=2,
                mask_token_id=0
            ).to(self.device)

        elif model_name == "char_trans":
            self.model = CharWordTransformerEncoding(
                n_words=self.train_dataset.n_words,
                n_chars=self.train_dataset.n_chars,
                n_label_errors=2,
                mask_token_id=0,
                use_detection_context=self.use_detection_context
            ).to(self.device)

    def make_dataset(self):
        if self.is_transformer:
            train_dataset = TransformerEncoderDataset(
                indexs=self.train_idxs,
                token_ids=self.token_ids,
                add_char_level=self.add_char_level
            )
            val_dataset = TransformerEncoderDataset(
                indexs=self.val_idxs,
                token_ids=self.token_ids,
                add_char_level=self.add_char_level
            )
        else:
            train_dataset = PhoBertDataset(
                indexs=self.train_idxs,
                token_ids=self.token_ids
            )
            val_dataset = PhoBertDataset(
                indexs=self.val_idxs,
                token_ids=self.token_ids
            )
        return train_dataset, val_dataset

    def make_loader(self):
        self.train_dataset, self.val_dataset = self.make_dataset()
        if not self.is_split_indexs:
            save_picke_file('train_indexs.pkl', self.train_idxs)
            print("Save done")
            save_picke_file('val_indexs.pkl', self.val_idxs)
            print("Save done")

        self.pad_id = 0
        self.pad_label_error = 0
        self.n_errors = self.train_dataset.n_errors
        self.n_words = self.train_dataset.n_words

        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=MyCollate(self.pad_id, self.pad_label_error, is_bert=self.is_bert)
        )

        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=MyCollate(self.pad_id, self.pad_label_error, is_bert=self.is_bert)
        )
        return train_loader, val_loader

    def get_loss(self, detection_outputs, label_errors, correction_outputs, words_correction):
        loss1 = self.loss_detection(detection_outputs.reshape(-1, self.n_errors), label_errors.reshape(-1))
        loss2 = self.loss_correction(correction_outputs.reshape(-1, self.n_words),
                                     words_correction.reshape(-1))
        return self.lam * loss1 + (1 - self.lam) * loss2

    def train_one_epoch(self):
        train_loss = 0
        self.model.train()
        start_time = time.time()
        for idx, data in enumerate(self.train_loader):
            error_ids = data['error_ids'].to(self.device)
            words_correction = data['word_corrections'].to(self.device)
            label_errors = data['label_errors'].to(device)
            batch_splits = data.get('batch_splits', None)
            if self.add_char_level:
                char_error_ids = data['char_error_ids'].to(self.device)
                detection_outputs, correction_outputs = self.model(error_ids, char_error_ids, batch_splits)
            elif self.is_bert:
                attention_masks = data['attention_masks'].to(device)
                detection_outputs, correction_outputs = self.model(error_ids, attention_masks, batch_splits)
            else:
                detection_outputs, correction_outputs = self.model(error_ids)
            loss = self.get_loss(detection_outputs, label_errors, correction_outputs, words_correction)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            if idx % 500 == 0:
                print(idx, end=" ")
        #                 print(idx)
        #                 print(time.time()-start_time)
        #                 start_time=time.time()
        print()
        return train_loss / len(self.train_loader)

    def evaluate_model(self):
        val_loss = 0
        predict_detections, predict_corrections = [], []
        label_detections, label_corrections = [], []
        self.model.eval()
        print('-' * 30 + 'TIME FOR VALIDATING' + '-' * 30)
        for idx, data in enumerate(self.val_loader):
            error_ids = data['error_ids'].to(self.device)
            words_correction = data['word_corrections'].to(self.device)
            label_errors = data['label_errors'].to(device)
            batch_splits = data.get('batch_splits', None)
            if self.add_char_level:
                char_error_ids = data['char_error_ids'].to(self.device)
                detection_outputs, correction_outputs = self.model(error_ids, char_error_ids, batch_splits)
            elif self.is_bert:
                attention_masks = data['attention_masks'].to(self.device)
                detection_outputs, correction_outputs = self.model(error_ids, attention_masks, batch_splits)
            else:
                detection_outputs, correction_outputs = self.model(error_ids)

            loss = self.get_loss(detection_outputs, label_errors, correction_outputs, words_correction)
            val_loss += loss.item()
            predict_detections.append(get_label(detection_outputs))
            predict_corrections.append(get_label(correction_outputs))
            label_detections.append(label_errors.reshape(-1))
            label_corrections.append(words_correction.reshape(-1))
            if idx % 500 == 0:
                print(idx, end=" ")
        print()
        # detection
        predict_detections = torch.cat(predict_detections).reshape(-1)
        label_detections = torch.cat(label_detections).reshape(-1)
        temp1 = label_detections.detach().cpu().numpy()
        temp2 = predict_detections.detach().cpu().numpy()
        print(classification_report(temp1, temp2))
        total_prediction_detection_correct = torch.sum(predict_detections == label_detections).item()
        n_samples = predict_detections.size(0)

        indexs_true_predict = (predict_detections == 1).nonzero(as_tuple=True)[0]
        indexs_true = (label_detections == 1).nonzero(as_tuple=True)[0]
        number_wrong = torch.sum(label_detections == 1).item()
        
        
        index_true_predict_correct = []
        for index in indexs_true.detach().cpu().numpy():
            if predict_detections[index].item() == 1:
                index_true_predict_correct.append(index)
                
        total_label_1_correct = len(index_true_predict_correct)
        # correction
        predict_corrections_v1 = torch.cat(predict_corrections).reshape(-1)[index_true_predict_correct]
        label_corrections_v1 = torch.cat(label_corrections).reshape(-1)[index_true_predict_correct]
        total_prediction_correction_correct = torch.sum(predict_corrections_v1 == label_corrections_v1).item()
        
        predict_corrections_v2 = torch.cat(predict_corrections).reshape(-1)[indexs_true]
        label_corrections_v2 = torch.cat(label_corrections).reshape(-1)[indexs_true]
        total_correction_correct = torch.sum(predict_corrections_v2 == label_corrections_v2).item()
        print("Total label 1 correct:", total_label_1_correct)
        print("Acc label 1 correct:", total_label_1_correct/number_wrong)
        print("Total predict word wrong:", indexs_true_predict.size(0))
        print("Total predict word wrong correct:", total_prediction_correction_correct)
        print("Total number word wrong:", number_wrong)
        print(n_samples)
        print(total_correction_correct)
        if indexs_true_predict.size(0) != 0:
            return val_loss / len(
                self.val_loader), total_prediction_detection_correct / n_samples, total_prediction_correction_correct / number_wrong, total_prediction_correction_correct / indexs_true_predict.size(
                0)
        else:
            return val_loss / len(
                self.val_loader), total_prediction_detection_correct / n_samples, 0, 0

    def save_model(self, epoch: int, history: dict):
        model_states = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(model_states,f"model{epoch}.pth")
        print("Save model done")
        with open(f"history{epoch}.pkl", "wb") as file:
            pickle.dump(history, file)
        print("Save history done")

    def fit(self):
        history = defaultdict(list)
        best_acc = 0
        for epoch in range(self.n_epochs):
            print('-' * 30 + 'TIME FOR TRAINING' + '-' * 30)
            start_time = time.time()
            self.epoch_temp = epoch
            train_loss = self.train_one_epoch()
            val_loss, val_acc_detect, val_acc_correction, val_precision_correction = self.evaluate_model()
            history['train_loss'].append(train_loss)
            history['val_acc_detect'].append(val_acc_detect)
            history['val_acc_correction'].append(val_acc_correction)
            history['val_loss'].append(val_loss)
            print(
                f"EPOCH:{epoch}---Val acc detect:{val_acc_detect}---Val acc correction:{val_acc_correction}--Val precision correction:{val_precision_correction}---Train "
                f"loss:{train_loss}---Val loss:{val_loss}---Time:{time.time() - start_time}"
            )
            if (val_acc_detect + 2*val_acc_correction) / 3 > best_acc:
                self.save_model(epoch, history)
                best_acc = (val_acc_detect + 2*val_acc_correction) / 3
                
            print("Current acc:",(val_acc_detect + 2*val_acc_correction) / 3)
            print("Best acc:", best_acc)