
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import json
import evaluate
import torch
import argparse
from comet import download_model, load_from_checkpoint

parser = argparse.ArgumentParser(description='')

parser.add_argument('--src_lang', required=True, help='source language')
parser.add_argument('--tgt_lang', required=True, help='target language')
parser.add_argument('--gpu', required=True, help='gpu')

args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

model_checkpoint = "./models/mbart-large-50-2nd/checkpoint-400000"

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# comet_path = download_model("Unbabel/wmt20-comet-da")
# comet = load_from_checkpoint(comet_path)
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")

source_lang = args.src_lang
target_lang = args.tgt_lang

src_references = []
tgt_references = []

with open('/home/osj/speech_synthesis/data/test/test.json') as f:
    test_current = json.load(f)

for elem in test_current[:10000]:
    src_references.append(elem['원문'])
    tgt_references.append(elem['최종번역문'])
print("로드 완료")
translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=source_lang, tgt_lang=target_lang, device=device, max_length=200, batch_size=16)
predictions = translator(src_references)
predictions = [predictions[i]['translation_text'] for i in range(len(predictions))]
print("번역 완료")
# compute scores
bleu_score = bleu.compute(predictions=predictions, references=tgt_references)
chrf_score= chrf.compute(predictions=predictions, references=tgt_references)
# comet_score = comet.predict([{'src':src_references, 'mt':predictions, 'ref':tgt_references}], batch_size=8, gpus=1)

with open("./results.txt", "a") as f:
    result = f"bleu: {bleu_score['score']}\nchrf: {chrf_score['score']}\n"# comet: {comet_score['scores'][0]}\n\n"
    f.write(result)  
