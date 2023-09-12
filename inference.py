from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import argparse
import json

parser = argparse.ArgumentParser(description='')

parser.add_argument('--src_lang', required=True, help='source language')
parser.add_argument('--tgt_lang', required=True, help='target language')
parser.add_argument('--gpu', required=True, help='gpu')
parser.add_argument('--test_file', required=True, help='test path')

args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

model_checkpoint = "./models/mbart-large-50-400k/checkpoint-400000"

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

source_lang = args.src_lang
target_lang = args.tgt_lang
 
with open(f"./data/test/{args.test_file}.txt")as f:
    test_prof = f.readlines()
testset = list(map(lambda s: s.strip(), test_prof))

translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=source_lang, tgt_lang=target_lang, device=device, max_length=200, batch_size=20)
predictions = translator(testset)
predictions = [predictions[i]['translation_text'] for i in range(len(predictions))]


with open("./outputs/output.txt", 'w') as outfile:
    for elem in predictions:
        outfile.write(f"{elem}\n")