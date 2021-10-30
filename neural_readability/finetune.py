import os
import time
import logging
import argparse
import torch
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, f1_score, cohen_kappa_score

from bert.dataloader import ReadabilityDataset, NewselaPreprocessor
from utils.paths import CHECKPOINT_DIR, LOG_DIR
from utils.consts import task_num_labels

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    filename=f"{LOG_DIR}/{time.time()}.bert.finetune.txt",
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
        
        batch = {_: feature.to(device) for _, feature in batch.items()}
        
        with torch.no_grad():
            logits = model(**batch)[1]

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

    parser.add_argument("--bert-model", default='bert-base-cased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task-name",
                        default='newsela',
                        type=str,
                        help="The name of the task to train, should be 'onestopenglish', 'weebit', 'newsela' or 'ucbeniki'")
    parser.add_argument("--output-dir",
                        default=str(CHECKPOINT_DIR),
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max-seq-length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
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
    parser.add_argument("--do-lower-case",
                        type=bool,
                        default=False,
                        help="Set this to True if you are using an uncased model.")
    parser.add_argument("--train-batch-size",
                        default=128,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval-batch-size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning-rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num-train-epochs",
                        default=15,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup-proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no-cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

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

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    
    preprocessor = preprocessors[task_name](args.seed)

    sentences_train, sentences_valid, sentences_test, labels_train, labels_valid , labels_test = preprocessor.get_split()

    model = AutoModelForSequenceClassification.from_pretrained(args.bert_model, num_labels = num_labels, output_attentions=False, output_hidden_states=False)
    model.to(device)

    num_training_steps = None

    if args.do_train:

        num_training_steps = int((len(sentences_train) / args.train_batch_size) * args.num_train_epochs)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    if args.do_train:

        logger.info("***** Running training *****")
        logger.info("  Num training examples = %d", len(sentences_train))
        logger.info("  Num validing examples = %d", len(sentences_valid))
        logger.info("  Num testing examples = %d", len(sentences_test))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num Epochs = %d", args.num_train_epochs)

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

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):

            epoch_loss = []
            
            model.train()

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

                batch = {_: feature.to(device) for _, feature in batch.items()}

                loss = model(**batch)[0]
                loss.backward()

                if (step+1) % args.log_interval == 0:
                    
                    logger.info(f"[Epoch {epoch+1}] @ step {step+1} loss: {loss:.2f}")

                epoch_loss.append(loss.item())

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if args.do_valid and (epoch+1) % args.valid_interval == 0:
                evaluate(model, valid_dataloader, device, args)
                
            if (epoch+1) % args.save_interval == 0 or (epoch+1) == args.num_train_epochs:
                os.makedirs(CHECKPOINT_DIR / args.bert_model, exist_ok=True)
                model.save_pretrained(CHECKPOINT_DIR / args.bert_model / f"checkpoint{epoch+1}", state_dict=model.state_dict())
                tokenizer.save_vocabulary(str(CHECKPOINT_DIR / args.bert_model / f"checkpoint{epoch+1}"))
            
            logger.info(f"Epoch {epoch+1} loss: {sum(epoch_loss):.2f} Avg step loss: {sum(epoch_loss)/len(epoch_loss):.2f}")
            
        if args.do_test:
            evaluate(model, test_dataloader, device, args, stage="Test")

if __name__ == "__main__":
    main()