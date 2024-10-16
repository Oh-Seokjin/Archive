import torch
import pyarrow as pa
import pandas as pd
import json
import evaluate
import numpy as np
import os
import argparse
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

parser = argparse.ArgumentParser(description='')

parser.add_argument('--src_lang', required=True, help='source language')
parser.add_argument('--tgt_lang', required=True, help='target language')
parser.add_argument('--gpu', required=True, help='gpu')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TOKENIZERS_PARALLELISM"]="false"

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

def preprocess_function(examples):
    inputs = [example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets)
    return model_inputs

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def make_datasetdict(data):
    temp = {}
    temp['id']=[]
    temp['translation']=[]
    for i in range(len(data)):
        temp['id'].append(str(i))
        temp['translation'].append({source_lang:data[i]['원문'], target_lang:data[i]['최종번역문']})

    temp = pd.DataFrame(temp)
    dataset = Dataset(pa.Table.from_pandas(temp))

    return dataset

source_lang = args.src_lang
target_lang = args.tgt_lang

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50")
tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50", src_lang=source_lang, tgt_lang=target_lang)

with open(f'./data/train/train.json') as f:    
    trainset= json.load(f)

with open('./data/valid/valid.json') as f:
    validset= json.load(f)

train_datasetdict = make_datasetdict(trainset)
valid_datasetdict = make_datasetdict(validset)

train_dataset = train_datasetdict.map(preprocess_function, batched=True)
valid_dataset = valid_datasetdict.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

metric = evaluate.load("sacrebleu")

training_args = Seq2SeqTrainingArguments(
    output_dir=f"./models/mbart-latge-50-full",
    evaluation_strategy="epoch", 
    optim="adamw_hf",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=11,
    num_train_epochs=15,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
    save_strategy="epoch",
    seed=1004,
    load_best_model_at_end=True,
    metric_for_best_model="eval_bleu",
    gradient_accumulation_steps=1,
    dataloader_num_workers=4
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()