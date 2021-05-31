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
  x = requests.get("https://gist.githubusercontent.com/yashbonde/62df9d16858a43775c22a6af00a8d707/raw/0764da94f5e243b2bca983a94d5d6a4e4a7eb28a/daily.py").content
  with open("daily.py", "wb") as f:
    f.write(x)
  from daily import *


# ---- functions
# Merkle Diff Algorithm, same as the one used for git diff
def diff(e, f, i=0, j=0):
  #  Returns a minimal list of differences between 2 lists e and f
  #  requring O(min(len(e),len(f))) space and O(min(len(e),len(f)) * D)
  #  worst-case execution time where D is the number of differences.
  #  Documented at http://blog.robertelder.org/diff-algorithm/
  N,M,L,Z = len(e),len(f),len(e)+len(f),2*min(len(e),len(f))+2
  if N > 0 and M > 0:
    w,g,p = N-M,[0]*Z,[0]*Z
    for h in range(0, (L//2+(L%2!=0))+1):
      for r in range(0, 2):
        c,d,o,m = (g,p,1,1) if r==0 else (p,g,0,-1)
        for k in range(-(h-2*max(0,h-M)), h-2*max(0,h-N)+1, 2):
          a = c[(k+1)%Z] if (k==-h or k!=h and c[(k-1)%Z]<c[(k+1)%Z]) else c[(k-1)%Z]+1
          b = a-k
          s,t = a,b
          while a<N and b<M and e[(1-o)*N+m*a+(o-1)]==f[(1-o)*M+m*b+(o-1)]:
            a,b = a+1,b+1
          c[k%Z],z=a,-(k-w)
          if L%2==o and z>=-(h-o) and z<=h-o and c[k%Z]+d[z%Z] >= N:
            D,x,y,u,v = (2*h-1,s,t,a,b) if o==1 else (2*h,N-a,M-b,N-s,M-t)
            if D > 1 or (x != u and y != v):
              return diff(e[0:x],f[0:y],i,j)+diff(e[u:N],f[v:M],i+u,j+v)
            elif M > N:
              return diff([],f[N:M],i+N,j+N)
            elif M < N:
              return diff(e[M:N],[],i+M,j+M)
            else:
              return []
  elif N > 0: #  Modify the return statements below if you want a different edit script format
    return [{"operation": "delete", "position_old": i+n} for n in range(0,N)]
  else:
    return [{"operation": "insert", "position_old": i,"position_new":j+n} for n in range(0,M)]

def shift_idx(target_string, tokenizer):
    target_seq = re.sub(r"[.,\/#!$%\^&\*;:{}=\-_`~()]", "", target_string)
    input_seq = target_seq.lower()
    t_tokens = tokenizer.tokenize(target_seq)
    t_tokens_lower = [x.lower() for x in t_tokens]
    i_tokens = tokenizer.tokenize(input_seq)

    ds = diff(i_tokens, t_tokens_lower)
    init_idx = sorted(list(set([x["position_old"] for x in ds])))
    return init_idx, target_seq, input_seq


# ---- classes

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


class FastBinaryCollater:
  def __init__(self, tokenizer, seqlen, *args, **kwargs):
    self.tokenizer = tokenizer
    self.seqlen = seqlen

  def diff_arr(self, x, y):
    # automanage 2D lists
    if isinstance(x[0], list):
      return [self.diff_arr(x_, y_) for x_,y_ in zip(x,y)]
    else:
      out = diff(x, y)
      idx = list(set([x["position_old"] for x in out]))

      # in an initial version I was converting the idx to np.array using
      # idx = np.asarray(idx), this was useful when we were creating labels
      # but is caused memory leakage issue where 100 calls with batch size = 64
      # with 6 parallel cores took 54s with each iteration taking longer,
      # with this it takes ~13.4s with same configuration
      return idx

  def __call__(self, batch):
    M = self.seqlen
    tokenizer = self.tokenizer

    target_seq = [re.sub(r"[.,\/#!$%\^&\*;:{}=\-_`~()\n]", "", x["normalizedBody"].strip()) for x in batch]
    target_seq = [re.sub(r"\s+", " ", x) for x in target_seq]
    input_seq = [x.lower() for x in target_seq]

    target_tokens = [tokenizer.tokenize(x)[:M - 1] for x in target_seq]
    input_tokens = [tokenizer.tokenize(x)[:M - 1] for x in input_seq]
    diffs = self.diff_arr(input_tokens, target_tokens) # list of list

    input_tokens_pt = tokenizer(
      input_seq,
      padding = "longest",
      return_tensors = "pt",
      add_special_tokens = True,
      return_token_type_ids = False
    )
    input_tokens_pt = {k:v[:, :M] for k,v in input_tokens_pt.items()}
    target_tokens_pt = torch.zeros_like(input_tokens_pt["input_ids"])

    for i,d in enumerate(diffs):
      target_tokens_pt[i][d] = 1

    return {
      "labels": target_tokens_pt,
      "target_str": target_tokens,
      **input_tokens_pt
    }


class FastSelfTargetCollate:
  def __init__(self, tokenizer, seqlen: int):
    self.tokenizer = tokenizer
    self.seqlen = seqlen

  def __call__(self, batch):
    target_seq = " ".join([x["normalizedBody"] for x in batch])
    diff_idx, target_seq, input_seq = shift_idx(target_seq, self.tokenizer)
    input_ids = self.tokenizer(input_seq, return_tensors = "pt")["input_ids"]
    input_ids = input_ids[0][1:-1] # drop the flags
    target_tokens_core = self.tokenizer(target_seq, return_tensors = "pt")["input_ids"]
    target_tokens_core = target_tokens_core[0]
    target_tokens = input_ids.clone()
    if diff_idx:
      target_tokens[diff_idx] = target_tokens_core[diff_idx]

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
    "self_target": SelfTargetCollate(tokenizer, seqlen),
    "fast_binary_flag": FastBinaryCollater(tokenizer, seqlen),
    # "fast_self_target": FastSelfTargetCollate(tokenizer, seqlen)
  }


if __name__ == "__main__":
  import os
  cache_dir = folder(folder(__file__))
  hf_cache = os.path.join(cache_dir, "hf-cache/")

  dataset = load_dataset("reddit", cache_dir=cache_dir)
  tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased", cache_dir = hf_cache)
  # `AutoModelForMaskedLM` has head `AutoModel` has no head
  model = AutoModel.from_pretrained("bert-base-cased", cache_dir = hf_cache)

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
  COLLATE_METHODS = get_collate_fns(tokenizer, 512)

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
