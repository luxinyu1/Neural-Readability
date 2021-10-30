import argparse
from bert.predict import bert_predict
from lstm.predict import lstm_predict

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--arch",
                        type=str,
                        required=True,
                        help="bilstm or bert")
    parser.add_argument("--input-file-path",
                        type=str,
                        required=True,
                        help="Input file path")
    parser.add_argument("--model-ckpt-name",
                        type=str,
                        required=True,
                        help="The name of the model checkpoint which to be used to make prediction.")
    parser.add_argument("--bert-model-name",
                        type=str,
                        required=True, 
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--model-training-task",
                        type=str,
                        required=True,
                        help="The name of the task to train, should be 'onestopenglish', 'weebit', 'newsela' or 'ucbeniki'")

    parser.add_argument("--predict-batch-size",
                        default=128,
                        type=int,
                        help="Batch size for predicting.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--no-cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--do-lower-case",
                        type=bool,
                        default=False,
                        help="Set this to True if you are using an uncased model.")

    args = parser.parse_args()

    if args.arch == 'bert':
    
        bert_predict(args)
    
    elif args.arch == 'lstm':

        lstm_predict(args)

if __name__ == '__main__':
    main()