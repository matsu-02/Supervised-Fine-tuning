# coding: utf-8
# %%
import os
print(f"{os.getenv('CUDA_VISIBLE_DEVICES')=}")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEFAULT_CACHE_DIR = '/content/cache'
# DEFAULT_CACHE_DIR = os.path.expanduser('~/.cache') # デフォルトの.cacheディレクトリのパスを取得
os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = DEFAULT_CACHE_DIR
# os.chdir("/content")
# print(os.getcwd())

import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM #, RichProgressCallback
from datasets import load_from_disk, DatasetDict
from peft import LoraConfig #. get_peft_model. PeftModel, PeftConfig

def main():
  DTYPE = torch.bfloat16 # if torch.cuda.is_bf16_supported() else torch.float32

  model_name = 'sbintuitions/sarashina2.2-1b-instruct-v0.1'
  cutoff_len = 8192
  seed = 42

  tokenized_datasets_dir = '/content/gdrive/MyDrive/Colab Notebooks/datasets/250407/tokenized'

  JST = datetime.timezone(datetime.timedelta(hours=9), 'JST')
  output_dir = f"/content/gdrive/MyDrive/Colab Notebooks/outputs/{datetime.datetime.now(JST):%Y%m%d-%H%M%S}"
  os.makedirs(output_dir, exist_ok=True)

  peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','im_head'], #
    inference_mode=False,
    bias="none",
    task_type="CAUSAL_LM"
    )
  model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=DEFAULT_CACHE_DIR,
    torch_dtype=DTYPE,
    # trust_remote_code=True,
    # use_cache=False, # if training_args.gradient_checkpointing else True,
    device_map='auto', # get_kbit_device_map() if quantization_config is not None else None.
    # quantization_config=quantization_config,
    )
  tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=DEFAULT_CACHE_DIR,
    # trust_remote_code=True,
    # use_fast=True,
    )
  if not tokenizer.pad_token_id:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
  
  dataset = load_from_disk(tokenized_datasets_dir)
  print(f"{dataset['train'].take(2)['input_ids']=}")
  dataset = DatasetDict({
    "train" : dataset['train'].take(10000),
    "valid" : dataset['valid'].take(200),
    })

  # %%
  response_template = tokenizer.encode('<|assistant|>', add_special_tokens=False) # [8]
  # response_template = tokenizer.encode("## 応答:\n ". add_special_tokens=False) # [2343, 271, 30401, 287, 25, 27T]
  print(f"{response_template=}")
  data_collator = DataCollatorForCompletionOnlyLM(
    response_template=[8],
    tokenizer=tokenizer,
    )
   
  trainer = SFTTrainer(
    model=model,
    # tokenizer=tokenizer,
    # optimizers=optimizers,
    train_dataset=dataset['train'],
    eval_dataset=dataset['valid'],
    # formatting_func=formatting_prompts_func,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    peft_config=peft_config,
    args=SFTConfig(
      dataset_kwargs={
        'add_special_tokens': False, # <bos>を追加しない
        'append_concat_token': False, # <eos>を追加しない
        },
      dataset_text_field='',
      max_seq_length=cutoff_len,
      packing=False,
      use_cpu=False,
      num_train_epochs=1,
      output_dir=output_dir, # './output/logs',
      eval_strategy='steps',
      eval_steps=0.2,
      logging_strategy='steps',
      logging_steps=0.2,
      logging_dir=f'{output_dir}/logs',
      log_level='error',
      report_to='tensorboard', # 'none',
      save_strategy='steps',
      save_steps=0.4,
      save_total_limit=1,
      per_device_train_batch_size=2,
      per_device_eval_batch_size=2,
      auto_find_batch_size=True,
      gradient_accumulation_steps=2,
      overwrite_output_dir=True, # logを上書きするか
      load_best_model_at_end=True, # EarlyStoppingを使用するならTrue
      metric_for_best_model='eval_loss', # EarlyStoppingの判断基準。compute_metricsのものを指定
      remove_unused_columns=True, # True:モデルのforward関数の引数に存在しないものは自動で削除
      warmup_ratio=0.1,
      # warmup_steps=100,
      learning_rate=1e-4,
      optim='adamw_torch', # adamw 8bit
      weight_decay=0.01, # 0.001 # bias/LayerNormウェイトを除く全レイヤーに適用するウェイト减豪
      lr_scheduler_type='linear', # 'cosine', # 'constant",
      # gradient_checkpointing=True,
      # gradient_checkpointing_kwargs={'use_reentrant': not True},
      fp16=not torch.cuda.is_bf16_supported(),
      bf16=torch.cuda.is_bf16_supported(),
      max_grad_norm=0.3, # 最大法線勾配（勾配クリッビング）
      group_by_length=True, # シーケンスを同じ長さのバッチにグルーブ化（メモリ節約）
      seed=42,
      disable_tqdm=False,
      push_to_hub=False,
    )
  )
  trainer_stats = trainer.train()
  # trainer.save_state()
  trainer.save_model()


# %%
if __name__=='__main__':
  main()

