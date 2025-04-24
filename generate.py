# coding: utf-8
# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOO'] = 'spawn'
print(os.getcwd())
DEFAULT_CACHE_DIR = '/content/cache'
os.environ['HF_HOME'] = DEFAULT_CACHE_DIR

import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM #. RichProgressCallback
from datasets import load_from_disk, DatasetDict
from peft import LoraConfig # , get_peft_model, PeftModel, PeftConfig
from peft import PeftConfig, PeftModel
from vllm import LLM, SamplingParams

def main():

  cutoff_len = 8192
  output_lora = '/content/gdrive/MyDrive/Colab Notebooks/outputs/20250420-134322'
  # modeT_name = 'Aratako/sarashina2.1-1b-sft'
  model_name = PeftConfig.from_pretrained(output_lora).base_model_name_or_path

  # ベースモデルの読み込み
  base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=DEFAULT_CACHE_DIR,
    torch_dtype='auto',
    device_map='auto',
    )

  # Rinnaのトークナイザーでは、「use_fast=False」が必要
  tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=DEFAULT_CACHE_DIR,
    # trust_remote_code=True,
    # use_fast=True,
    )
  # PEFT（LoRA）の読み込み
  peft_model = PeftModel.from_pretrained(base_model, output_lora, is_trainable=True)
  # マージモデル作成
  peft_model = peft_model.merge_and_unload()
  # 出力
  merged_dir = f"{output_lora}/merged"
  os.makedirs(merged_dir, exist_ok=True)
  peft_model.save_pretrained(merged_dir)
  tokenizer.save_pretrained(merged_dir)
  print(f"Saving to {merged_dir}")

  if not tokenizer.pad_token_id:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

  llm = LLM(
    model=merged_dir,
    trust_remote_code=True,
    download_dir=DEFAULT_CACHE_DIR,
    # pipeline_parallel_size=2,
    tensor_parallel_size=2,
    dtype='auto', # torch.bfloat16, #
    seed=42,
    # gpu_memory_utilization=0.9, # GPUの最大使用率
    max_model_len=8192,
    # max_num_seqs=2, # Maximum number of tokens per iteration.
    )

  sampling_params = SamplingParams(
    n=5,
    temperature=0.8,
    # frequency_penalty=0.3,
    top_p=0.9,
    seed=42,
    stop=[tokenizer.eos_token],
    # ignore_eos=False,
    max_tokens=cutoff_len,
    # stop_token_ids=None,
    )

  # print(f"{sampling_params.stop=}")
  print(getattr(sampling_params, "stop"))

  test_dataset = load_from_disk(f"{tokenized_datasets_dir}/test")
  print(test_dataset)
  i = 0
  messages = [
    { "role" : "system",    "content": "あなたは優秀な日本人のアシスタントです。" },
    { "role" : "user",      "content": test_dataset["instruction"][i] },
    # { "role" : "assistant", "content": test_dataset["output"][i] },
    ]
  prompt_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    )
  print(f"{prompt_text=}")
  outputs = llm.generate(prompt_text, sampling_params=sampling_params)
  print(f"{outputs=}")
  generations = []
  for req in outputs:
    print(req)
    for output in req.outputs: # CompletionOutput(index, text, token_ids, cumulative_logprob, logprobs, finish_reason, stop_reason)
      if output.finish_reason=='stop': # eos_tokenでstopした場合
        generations.append(output.text)
  print(f"{generations=}")

# %%
if __name__=='__main__':
  main()


