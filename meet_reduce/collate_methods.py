import re
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, BertTokenizerFast

# add a bunch of tiny code snippets, downloads daily.py to the master process __file__ location
try:
  from daily import *
except ImportError as e:
  import requests
  x = requests.get("https://gist.githubusercontent.com/yashbonde/62df9d16858a43775c22a6af00a8d707/raw/60c79e8fa29137c3b171197218d899e84308f55a/daily.py").content
  with open("daily.py", "wb") as f:
    f.write(x)
  from daily import *


class BinaryTargetCollate:
  def __init__(self, tokenizer, seqlen: int):
    self.tokenizer = tokenizer
    self.seqlen = seqlen

  def __call__(self, batch):
    # this model takes in the text and returns a tensor with 1
    # where casing has to happen, this does not compare the strings
    # side by side, but uses length of each tokenized word as a
    # proxy for the same.
    # WARN: input_batch_size != output_batch_size
    # WARN: very inefficient because of iterating over each word
    target_seq = " ".join([x["normalizedBody"] for x in batch])
    target_seq = re.sub(r"[.,\/#!$%\^&\*;:{}=\-_`~()]", "", target_seq)
    input_seq = target_seq.lower()
    input_tokens = []; target_tokens = []
    for x, y in zip(input_seq.split(), target_seq.split()):
      x_ = self.tokenizer(x)["input_ids"][1:-1]
      y_ = self.tokenizer(y)["input_ids"][1:-1]
      if not len(y_):
        continue
      b = int(len(x_) != len(y_))
      trg = [0 for _ in range(len(x_))]
      trg[0] = b
      input_tokens.extend(x_)
      target_tokens.extend(trg)
    input_ids = torch.Tensor(input_tokens).long()
    target_tokens = torch.Tensor(target_tokens).long()
    
    # now reshape this thing to correct number of batches and seqlen
    ml = self.seqlen - 2
    cutoff_idx = -(len(input_ids) % ml)
    input_ids = input_ids.contiguous()[:cutoff_idx].reshape(-1, ml)
    target_tokens = target_tokens[:cutoff_idx].reshape(-1, ml)
    
    # add [CLS] to the start and [SEP] at the end
    input_ids = torch.cat([
      torch.ones(len(input_ids), 1) * self.tokenizer.cls_token_id,
      input_ids,
      torch.ones(len(input_ids), 1) * self.tokenizer.sep_token_id,
    ], dim = 1).long()
    target_tokens = torch.cat([
      torch.ones(len(input_ids), 1) * self.tokenizer.cls_token_id,
      target_tokens,
      torch.ones(len(input_ids), 1) * self.tokenizer.sep_token_id,
    ], dim = 1).long()
    
    return {"input_ids": input_ids, "labels": target_tokens}


class SelfTargetCollate:
  def __init__(self, tokenizer, seqlen: int):
    self.tokenizer = tokenizer
    self.seqlen = seqlen

  def __call__(self, batch):
    # this method is pretty much similar to `BinaryTargetCollate`
    # and the only difference is that instead of returning a binary array
    # we only modify the tensor where we initially sent 1
    # WARN: input_batch_size != output_batch_size
    # WARN: very inefficient because of iterating over each word
    target_seq = " ".join([x["normalizedBody"] for x in batch])
    target_seq = re.sub(r"[.,\/#!$%\^&\*;:{}=\-_`~()]", "", target_seq)
    input_seq = target_seq.lower()
    input_tokens = []; target_tokens = []
    for x, y in zip(input_seq.split(), target_seq.split()):
      x_ = self.tokenizer(x)["input_ids"][1:-1]
      y_ = self.tokenizer(y)["input_ids"][1:-1]
      if not len(y_):
        continue
      b = int(len(x_) != len(y_))
      trg = x_.copy()
      trg[0] = y_[0]
      input_tokens.extend(x_)
      target_tokens.extend(trg)
    input_ids = torch.Tensor(input_tokens)
    target_tokens = torch.Tensor(target_tokens)
    
    # now reshape this thing to correct number of batches and seqlen
    ml = self.seqlen - 2
    cutoff_idx = -(len(input_ids) % ml)
    input_ids = input_ids.contiguous()[:cutoff_idx].reshape(-1, ml)
    target_tokens = target_tokens[:cutoff_idx].reshape(-1, ml)
    
    # add [CLS] to the start and [SEP] at the end
    input_ids = torch.cat([
      torch.ones(len(input_ids), 1) * self.tokenizer.cls_token_id,
      input_ids,
      torch.ones(len(input_ids), 1) * self.tokenizer.sep_token_id,
    ], dim = 1).long()
    target_tokens = torch.cat([
      torch.ones(len(input_ids), 1) * self.tokenizer.cls_token_id,
      target_tokens,
      torch.ones(len(input_ids), 1) * self.tokenizer.sep_token_id,
    ], dim = 1).long()
    
    return {"input_ids": input_ids, "labels": target_tokens}


# simple function for prod
def get_collate_fns(tokenizer, seqlen):
  return {
    "binary_flag": BinaryTargetCollate(tokenizer, seqlen),
    "self_target": SelfTargetCollate(tokenizer, seqlen)
  }


if __name__ == "__main__":
  cache_dir = folder(folder(__file__))

  dataset = load_dataset("reddit", cache_dir=cache_dir)
  tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
  # `AutoModelForMaskedLM` has head `AutoModel` has no head
  model = AutoModel.from_pretrained("bert-base-cased")

  train_split_ratio = 0.99
  seed = 4
  dd = dataset["train"].train_test_split(
  test_size = 1 - train_split_ratio,
  train_size = train_split_ratio,
  seed = seed
  )
  dstrain = dd["train"]
  dstest = dd["test"]

  # NOTE: since these collator functions sit outside the dataset it is difficult
  # to know what exactly will be the final batch size generally 25% smaller than
  # the batch_size given as an input to the model.
  COLLATE_METHODS = {
    "binary_flag": BinaryTargetCollate(tokenizer),
    "self_target": SelfTargetCollate(tokenizer)
  }

  collate_fn = COLLATE_METHODS["self_target"]
  loader = DataLoader(
    dstrain,
    batch_size=10,
    collate_fn=collate_fn,
    pin_memory=False,
    shuffle = True,
  )
  for i, x in enumerate(loader):
    break
