import pickle
import torch
import transformers
import nltk
import argparse
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM

from utils.consts import CAUSALLM_MODEL_NAMES, MASKEDLM_MODEL_NAMES
from utils.utils import read_lines

def calc_rsrs_score_mlm(model, tokenizer, sentence, device):

    # Note that some tokenizer will do subword segmentation, for example:
    # >>> tokenizer.tokenize("Negotiations continue.")
    # >>> ['N', '##ego', '##tia', '##tions', 'continue', '.']
    # In this case, RSRS score could deviate from its design goal.
    
    rsrss = []

    for sent in nltk.sent_tokenize(sentence):

        encodings = tokenizer(sent, return_tensors='pt')
        input_ids = encodings.input_ids
        s = input_ids.shape[1] # sentence length

        wnlls = []

        for i in range(s):

            target_ids = torch.full_like(input_ids, -100)
            target_ids[:,i] = input_ids[:,i]

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            with torch.no_grad():

                outputs = model(input_ids, labels=target_ids)
                word_neg_log_likelihood = outputs[0].item()
            
            wnlls.append(word_neg_log_likelihood)

        sorted_wnlls = sorted(wnlls)
        ranks = [sorted_wnlls.index(wnll)+1 for wnll in wnlls]
        rsrs = np.dot(np.sqrt(np.array(ranks)), np.array(wnlls)) / s
        rsrss.append(rsrs) # Sometimes samples may contain mutltiple sentences

    return np.mean(rsrss)

def calc_rsrs_score_clm(model, tokenizer, sentence, device):

    # Note that some tokenizer will do subword segmentation, for example:
    # >>> tokenizer.tokenize("Negotiations continue.")
    # >>> ['Neg', 'oti', 'ations', 'Ġcontinue', '.']
    # In this case, RSRS score could deviate from its design goal.

    rsrss = []

    for sent in nltk.sent_tokenize(sentence):

        encodings = tokenizer(sent, return_tensors='pt')
        input_ids = encodings.input_ids
        print(tokenizer.convert_ids_to_tokens(input_ids.squeeze(0)))
        s = input_ids.shape[1]
        
        wnlls = []

        # We treat the whole length in timestep i, 1<=i<s as input, not the ngrams.

        for i in range(1, s):

            current_input = input_ids[:,:i+1]
            current_target = torch.full_like(current_input, -100)

            current_target[:,i] = input_ids[:,i]

            current_input = current_input.to(device)
            current_target = current_target.to(device)

            with torch.no_grad():

                outputs = model(current_input, labels=current_target)
                word_neg_log_likelihood = outputs[0].item()

            wnlls.append(word_neg_log_likelihood)

        sorted_wnlls = sorted(wnlls)
        ranks = [sorted_wnlls.index(wnll)+1 for wnll in wnlls]
        rsrs = np.dot(np.sqrt(np.array(ranks)), np.array(wnlls)) / (s - 1)
        rsrss.append(rsrs) # Sometimes samples may contain mutltiple sentences

    return np.mean(rsrss)

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name",
                        type=str,
                        required=True)
    parser.add_argument("--input-file-path",
                        type=str,
                        required=True,
                        help="Input file path")

    parser.add_argument("--do-lower-case",
                        type=bool,
                        default=False,
                        help="Set this to True if you want to do lower case.")
    parser.add_argument("--no-cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    scores = []

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=args.do_lower_case)

    if args.model_name in CAUSALLM_MODEL_NAMES:

        model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

        sentences = read_lines(args.input_file_path)

        for s in sentences:

            rsrs = calc_rsrs_score_clm(model, tokenizer, s, device)
            scores.append(rsrs)

    elif args.model_name in MASKEDLM_MODEL_NAMES:

        model = AutoModelForMaskedLM.from_pretrained(args.model_name).to(device)

        sentences = read_lines(args.input_file_path)

        for s in sentences:

            rsrs = calc_rsrs_score_mlm(model, tokenizer, s, device)
            scores.append(rsrs)
    
    else:
        
        raise ValueError("Please check the model name!")

    output_file_path = Path(args.input_file_path).parent / 'rsrs.pickle'

    with open(output_file_path, "wb") as f:

        pickle.dump(scores, f)

if __name__ == '__main__':

    main()
