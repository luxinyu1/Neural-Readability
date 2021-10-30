import torch
import pickle
from tqdm import tqdm, trange
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils.paths import CHECKPOINT_DIR
from utils.consts import task_num_labels
from .dataloader import PredictDataset, PredictPreprocessor

def bert_predict(args):

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    model_path = CHECKPOINT_DIR / args.bert_model_name / args.model_ckpt_name

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name, do_lower_case=args.do_lower_case)

    num_labels = task_num_labels[args.model_training_task]

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        num_labels = num_labels, 
        output_attentions=False, 
        output_hidden_states=False
    )
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

        batch = batch.to(device)

        with torch.no_grad():

            logits = model(**batch).logits

        preds = torch.argmax(logits, dim=1).to("cpu").tolist()

        all_preds.extend(preds)

    output_file_path = Path(args.input_file_path).parent / 'preds.pickle'

    with open(output_file_path, "wb") as f:

        pickle.dump(all_preds, f)