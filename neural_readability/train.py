import os
import torch
import argparse
import time
import logging
import pandas as pd
from tqdm import tqdm, trange
from sklearn import model_selection
from transformers import AutoTokenizer, AutoConfig, AdamW
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

from utils.paths import CHECKPOINT_DIR, LOG_DIR
from utils.consts import task_num_labels
from lstm.dataloader import ReadabilityDataset, NewselaPreprocessor
from lstm.model import BiLSTM, LSTM

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    filename=f"{LOG_DIR}/{time.time()}.bilstm.finetune.txt",
    filemode='w'
)
logger = logging.getLogger(__name__)

def evaluate(model, dataloader, device, args, stage="Valid"):
    
    logger.info("***** Running Evaluation *****")
    logger.info("  Stage = %s", stage)
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    model.eval()
    
    preds_all = []
    labels_all = []
    
    for step, batch in enumerate(dataloader):
        
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to("cpu") # The lengths must be placed on cpu
        labels = batch["labels"].to(device)
        
        with torch.no_grad():
            logits = model(input_ids, lengths)

        preds = torch.argmax(logits, dim=1).to("cpu").tolist()
        labels = batch["labels"].to("cpu").tolist()
        
        preds_all.extend(preds)
        labels_all.extend(labels)
        
    accuracy = accuracy_score(labels_all, preds_all)
    precision = precision_score(labels_all, preds_all, average='weighted')
    recall = recall_score(labels_all, preds_all, average='weighted')
    f1 = f1_score(labels_all, preds_all, average='weighted')
    qwk = cohen_kappa_score(labels_all, preds_all, weights="quadratic")
    
    logger.info(f"Accuracy: {accuracy:.2f}")
    logger.info(f"Precision: {precision:.2f}")
    logger.info(f"Recall: {recall:.2f}")
    logger.info(f"F1: {f1:.2f}")
    logger.info(f"QWK: {qwk:.2f}")

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--task-name",
                        type=str,
                        default='newsela',
                        help="The name of the task to train, should be 'onestopenglish', 'weebit', 'newsela' or 'ucbeniki")
    parser.add_argument("--output-dir",
                        default=str(CHECKPOINT_DIR),
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    parser.add_argument("--num-train-epochs",
                        type=int,
                        default=100,
                        help='Total number of training epochs to perform.')
    parser.add_argument("--train-batch-size",
                        default=128,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval-batch-size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--embedding-dim",
                        default=200,
                        type=int,
                        help="The size of embedding dim.")
    parser.add_argument("--hidden-dim",
                        default=128,
                        type=int,
                        help="The size of hidden dim.")
    parser.add_argument("--do-train",
                        type=bool,
                        default=True,
                        help="Whether to run training.")
    parser.add_argument("--do-valid",
                        type=bool,
                        default=True,
                        help="Whether to run validation on the valid set.")
    parser.add_argument("--do-test",
                        type=bool,
                        default=True,
                        help="Whether to run testing on the test set.")
    parser.add_argument("--log-interval",
                        type=int,
                        default=100,
                        help="Log interval(Step) in training.")
    parser.add_argument("--valid-interval",
                        type=int,
                        default=1,
                        help="Validation interval(epoch) in training.")
    parser.add_argument("--save-interval",
                        type=int,
                        default=3,
                        help="Save interval(epoch) in training.")
    parser.add_argument("--learning-rate",
                        default=1e-3,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="Random seed for initialization")
    parser.add_argument("--bert-tokenizer",
                        type=str,
                        default="bert-base-cased",
                        help="The name of bert model.")
    parser.add_argument("--max-seq-length",
                        type=int,
                        default=512,
                        help="Each sequence will be padded to the max sequence length.")
    parser.add_argument("--do-lower-case",
                        type=bool,
                        default=False,
                        help="Set this to True if you are using an uncased model.")
    parser.add_argument("--no-cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    preprocessors = {
        "newsela": NewselaPreprocessor
    }

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_test` must be True.")

    task_name = args.task_name.lower()

    if task_name not in preprocessors:
        raise ValueError("Task not found: %s" % (task_name))

    num_labels = task_num_labels[task_name]

    os.makedirs(args.output_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(args.bert_tokenizer)
    vocab_size = config.vocab_size

    tokenizer = AutoTokenizer.from_pretrained(args.bert_tokenizer, do_lower_case=args.do_lower_case)

    preprocessor = preprocessors[task_name](args.seed)

    sentences_train, sentences_valid, sentences_test, labels_train, labels_valid , labels_test = preprocessor.get_split()

    model = LSTM(vocab_size, num_labels, device, args)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    loss_function = torch.nn.CrossEntropyLoss()

    train_dataset = ReadabilityDataset(
        sentences=sentences_train,
        labels=labels_train,
        tokenizer=tokenizer
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=0,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    valid_dataset = ReadabilityDataset(
        sentences=sentences_valid,
        labels= labels_valid,
        tokenizer=tokenizer
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        num_workers=0,
        collate_fn=valid_dataset.collate_fn
    )
    test_dataset = ReadabilityDataset(
        sentences=sentences_test,
        labels=labels_test,
        tokenizer=tokenizer
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        num_workers=0,
        collate_fn=test_dataset.collate_fn
    )

    if args.do_train:

        logger.info("***** Running training *****")
        logger.info("  Num training examples = %d", len(sentences_train))
        logger.info("  Num validing examples = %d", len(sentences_valid))
        logger.info("  Num testing examples = %d", len(sentences_test))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num Epochs = %d", args.num_train_epochs)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):

            epoch_loss = []

            model.train()

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

                input_ids = batch["input_ids"].to(device)
                lengths = batch["lengths"].to("cpu") # The lengths must be placed on cpu
                labels = batch["labels"].to(device)

                logits = model(input_ids, lengths)
                loss = loss_function(logits, labels)

                loss.backward()

                if (step+1) % args.log_interval == 0:
                    
                    logger.info(f"[Epoch {epoch+1}] @ step {step+1} loss: {loss.item():.2f}")

                epoch_loss.append(loss.item())

                optimizer.step()
                optimizer.zero_grad()

            if args.do_valid and (epoch+1) % args.valid_interval == 0:
                evaluate(model, valid_dataloader, device, args)

            if (epoch+1) % args.save_interval == 0 or (epoch+1) == args.num_train_epochs:
                os.makedirs(CHECKPOINT_DIR / "bilstm", exist_ok=True)
                torch.save(model, CHECKPOINT_DIR / "bilstm" / f"checkpoint{epoch+1}.pt")
            
            logger.info(f"Epoch {epoch+1} loss: {sum(epoch_loss):.2f} Avg step loss: {sum(epoch_loss)/len(epoch_loss):.2f}")
            
        if args.do_test:
            evaluate(model, test_dataloader, device, args, stage="Test")

if __name__ == '__main__':
    
    main()