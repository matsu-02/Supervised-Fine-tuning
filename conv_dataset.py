# coding: utf-8
#%%
import os

DEFAULT_CACHE_DIR = '/content/cache'
# DEFAULT_CACHE_DIR = os.path.expanduser('~/.cache') # デフォルトの.cacheディレクトリのパスを取得
os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
# Hugging Face関連の他のツールが使用する可能性のあるホームディレクトリ
os.environ['HF_HOME'] = DEFAULT_CACHE_DIR

import torch
import unicodedata
import json
# import datetime
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict #. load_from_disk, concatenate_datasets

def main():
  base_model_path = 'sbintuitions/sarashina2.2-1b-instruct-v0.1'
  cutoff_len = 8192
  seed = 42

  datasets = load_dataset(
      'shi3z/alpaca_cleaned_ja_json', split='train+test', cache_dir=DEFAULT_CACHE_DIR
    ).filter(
      lambda x: x['input']==''
    )
  # dataset_all = concatenate_datasets([datasets[d] for d in datasets.keys()])
  print(datasets)
  ds_splits = datasets.train_test_split(test_size=0.4, seed=seed)
  ds_test = ds_splits['test'].train_test_split(test_size=0.5, seed=seed)
  datasets_splits = DatasetDict({
    'train': ds_splits['train'],
    'valid': ds_test['train'],
    'test':  ds_test['test'],
    })
  del ds_splits, ds_test
  print(datasets_splits)

  base_data_dir = '/content/gdrive/MyDrive/Colab Notebooks/datasets/250407'
  data_output_dir = f'{base_data_dir}/tokenized'
  os.makedirs(data_output_dir, exist_ok=True)

  message = {
    'system'   : 'あなたは優秀な日本人のアシスタントです。',
    'user'     : '{instruction}',
    'assistant': '{response}',
    }
  # data_files = {
  #   "train" : f"{base_data_dir}/train.json",
  #   "valid" : f"{base_data_dir}/valid.json",
  #   # "test": f"{base_data_dir}/test.json",
  #   }
  # dataset = load_dataset("json", data_files=data_files, cache_dir=DEFAULT_CACHE_DIR)
  
  tokenizer = AutoTokenizer.from_pretrained(base_model_path, cache_dir=DEFAULT_CACHE_DIR)
  if tokenizer.pad_token_id == None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    print(tokenizer.pad_token, tokenizer.pad_token_id)
  print(f"{tokenizer.padding_side=}")
  print(f"{tokenizer.truncation_side=}")
  # tokenizer.padding_side = 'right"
  # tokenizer.truncation_side = 'right'
  response_template = tokenizer.encode('<|assistant|>', add_special_tokens=False)
  # ignore_index用
  print(f"{response_template=}") # [567. 90040. 250. 103899. 512]
  print(f"{torch.multiprocessing.cpu_count()=}")

  remove_columns = datasets_splits['train'].column_names
  processed_dataset = datasets_splits.map(
    build_prompt,
    batched=True,
    num_proc=2,
    remove_columns=remove_columns, ## 元のカラムは除く
    fn_kwargs=dict(
      tokenizer=tokenizer,
      message=message,
      cutoff_len=cutoff_len,
      ),
    load_from_cache_file=False,
    desc="map train valid datasets"
  )
  print(processed_dataset)

  # dataset['test'] = load_dataset("json", data_files=f"{base_data_dir}/test.json". split='train'. cache_dir=DEFAULT_CACHE_DIR)
  processed_dataset['test'] = datasets_splits['test'].map(
    build_prompt,
    batched=True,
    num_proc=2,
    # remove_columns=remove_columns, ## 元のカラムは除く
    fn_kwargs=dict(
      tokenizer=tokenizer,
      message=message,
      cutoff_len=cutoff_len,
      is_test=True,
      ),
    load_from_cache_file=False,
    desc="map test dataset"
  )

  print(processed_dataset['train'])
  print(processed_dataset['train']['input_ids'][0])
  print('train input_ids = ',tokenizer.convert_ids_to_tokens(processed_dataset['train']['input_ids'][0]))
    
    ## 保存
  if data_output_dir:
    os.makedirs(data_output_dir, exist_ok=True)
    print(f"Saving datasets to {data_output_dir}...")
    processed_dataset.save_to_disk(data_output_dir)
    print("done.")


def build_prompt(samples, tokenizer, message, cutoff_len, is_test=False):
  # prompts = []
  tokenized_prompt = {'input_ids':[], 'attention_mask':[]}
  for i in range(len(samples['instruction'])):
    text = samples['instruction'][i]
    response = samples['output'][i]
    tokenized = tokenize(tokenizer, message, cutoff_len, text, response, is_test)
    tokenized_prompt['input_ids'].append( tokenized['input_ids'] )
    tokenized_prompt['attention_mask'].append( tokenized['attention_mask'] )
  return tokenized_prompt

# 再起関数
def tokenize(tokenizer, message, cutoff_len, text, response, is_test):
  if is_test:
    prompt = tokenizer.apply_chat_template([
      {"role": "system", "content": message['system']},
      {"role": "user",   "content": message['user'].format(instruction=text)},
      ], tokenize=False, add_generation_prompt=True)
    prompt += response # '## 応答:¥n'
  else:
    prompt = tokenizer.apply_chat_template([
      {"role": "system",    "content": message['system']},
      {"role": "user",      "content": message['user'].format(instruction=text)},
      {"role": "assistant", "content": message['assistant'].format(response=response)},
      ], tokenize=False, add_generation_prompt=False) # add_generation_prompt: <|assistant|>を追加するかどうか

  tokenized_prompt = tokenizer(
    normalize(prompt),
    # truncation=True,
    # max_length=cutoff_len,
    # return_tensors='pt', # ptの場合Tensor型の2次元、なしの場合Listの1次元
    add_special_tokens=False,
    return_length=True,
    )
  diff_token = tokenized_prompt['length'][0] - cutoff_len

  if diff_token > 0: # stage == 'predict'
    diff_len = len(tokenizer.decode(tokenizer(normalize(text),add_special_tokens=False)['input_ids'][-diff_token:], skip_special_tokens=True))
    text = text[:-diff_len]
    if len(text)<50: # 入力が50文字禾満で出力が長すぎる場合
      return tokenized_prompt
    return tokenize(tokenizer, message, cutoff_len, text, response, is_test)
  else:
    return tokenized_prompt

def normalize(text):
  text = text.strip()
  # text = text.translate({
  text = unicodedata.normalize('NFKC', text)
  return text


# %%
if __name__=='__main__':
  main()

