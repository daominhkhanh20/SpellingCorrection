import argparse

from torch._C import has_lapack
from autocorrection.model.trainer import ModelTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True,
                    help="Model name")

parser.add_argument('--n_samples', type=int, default=int(8e5),
                    help="Number sample for training and valid")

parser.add_argument('--training_data_percent', type=float, default=0.8)

parser.add_argument('--lr', type=float, default=1e-5,
                    help="learning rate value")

parser.add_argument('--is_continuous_train', type=bool, default=False,
                    help="continue train with the previous model")

parser.add_argument('--path_pretrain_model', type=str, default=None)

parser.add_argument('--is_split_indexs', type=bool, default=True, 
                    help="Using the default split data for training and validation")
                    
parser.add_argument('--n_epoch', type=int, default=100,
                    help="number epochs")

parser.add_argument('--lam', type=float, default=0.7,
                    help="lambda  for control the important level of detection loss value")

parser.add_argument('--penalty_value', type=float, default=0.3,
                    help="parameter controlled the contribution of the label 0 in the detection loss")

parser.add_argument('--use_detection_context', type=bool, default=True)

parser.add_argument('--is_transformer', type=bool, default=False)

parser.add_argument('--add_char_level', type=bool, default=False,
                    help="Combine embedding char level to word embedding")

parser.add_argument('--is_bert', type=bool, default=False,
                    help="Are we need train with Phobert?")

parser.add_argument('--fine_tuned', type=bool, default=False,
                help='fine tuned pretrained model') 

parser.add_argument('--batch_size', type=int, default=128,
                    help="Batch size samples")

arg = parser.parse_args()


def main():
    trainer = ModelTrainer(
        model_name=arg.model_name,
        n_samples=arg.n_samples,
        training_data_percent=arg.training_data_percent,
        lr=arg.lr,
        is_continuous_train=arg.is_continouus_train,
        path_pretrain_model=arg.path_pretrain_model,
        is_split_indexs=arg.is_split_indexs,
        n_epochs=arg.n_epochs,
        lam=arg.lam,
        penalty_value=arg.penalty_value,
        use_detection_context=arg.use_detection_context,
        is_transformer=arg.is_transformer, 
        add_char_level=arg.add_char_level,
        fine_tuned=arg.fine_tuned,
        batch_size=arg.batch_size
    )
    trainer.fit()


if __name__ == "__main__":
    main()
