import torch
import pickle
from tqdm import tqdm, trange
from pathlib import Path
from transformers import AutoTokenizer

from utils.paths import CHECKPOINT_DIR
from utils.consts import task_num_labels
from .dataloader import PredictDataset, PredictPreprocessor
from torch.utils.data import DataLoader

def lstm_predict(args):

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name, do_lower_case=args.do_lower_case)

    num_labels = task_num_labels[args.model_training_task]

    model_path = CHECKPOINT_DIR / args.arch / f"{args.model_ckpt_name}.pt"
    model = torch.load(model_path)
    model.to(device)

    preprocessor = PredictPreprocessor(args.input_file_path)
    sentences = preprocessor.preprocessor()
    dataset = PredictDataset(sentences, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.predict_batch_size,
        num_workers=0,
        shuffle=False,
        collate_fn=dataset.collate_fn
    )

    all_preds = []

    for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):

        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to("cpu") # The lengths must be placed on cpu

        with torch.no_grad():

            logits = model(input_ids, lengths)

        preds = torch.argmax(logits, dim=1).to("cpu").tolist()

        all_preds.extend(preds)

    output_file_path = Path(args.input_file_path).parent / 'preds.txt'

    with open(output_file_path, "wb") as f:

        pickle.dump(all_preds, f)