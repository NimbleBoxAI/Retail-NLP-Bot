# @yashbonde - 3rd May, 2021 for NBX

import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from .utils import folder, GPT
from .functions import *

ENT_ZERO  = {
  "text": None,
  "start": 0,
  "end": 0,
  "entity": None
}


def get_model(name = "EleutherAI/gpt-neo-2.7B", cache_dir = ".model-cache/"):
  """load the model and tokenizer from hf-repos. Check out "hf.co/models" for
  more models.

  Note: try to provide a `cache_dir`as this will download all the relevant materials
  in this folder. Since the models can become very large it's a good idea to avoid
  downloading again and again. 

  Args:
      name (str, optional): hf model name. Defaults to "EleutherAI/gpt-neo-2.7B".
      cache_dir (str, optional): where to cache your model. Defaults to ".model-cache/".

  Returns:
      model, tokenizer
  """
  if os.getenv("NBX_LOCAL_DEV", False):
    name = "sshleifer/tiny-gpt2"
    print("Using local dev model with model name:", name)

  tokenizer = AutoTokenizer.from_pretrained(name, cache_dir = cache_dir)
  model = AutoModelForCausalLM.from_pretrained(name, cache_dir = cache_dir)
  device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
  model = model.to(device).eval()
  model = GPT(model, tokenizer)
  return model


class ProcessorGPT():
  def __init__(self, hf_backbone, cache_dir = "../hf-cache/"):
    print("Loading GPT Processor (Retail-Bot) ...")
    self.gpt = get_model(hf_backbone, cache_dir)

    self.ID_TO_TAG = {
      0: 'null',
      1: 'name',
      2: 'address',
      3: 'city',
      4: 'state',
      5: 'pincode',
      6: 'phone'
    }
    self.TAG_TO_ID = {v:k for k,v in self.ID_TO_TAG.items()}

  def process(self, text):
    # method to perform inference on input text and return industry standard api
    name = get_name(self.gpt, text)
    city = get_city(self.gpt, text)
    pc = get_pincode(self.gpt, text)

    # get the 
    t2 = text.replace(name, "")
    t2 = t2.replace(city, "")
    t2 = t2.replace(pc, "").strip()

    return {
      "text": text,
      "entities": [{
        "text": name,
        "entity": self.TAG_TO_ID["name"]
      }, {
        "text": city,
        "entity": self.TAG_TO_ID["city"]
      }, {
        "text": pc,
        "entity": self.TAG_TO_ID["pincode"]
      }, {
        "text": t2,
        "entity": self.TAG_TO_ID["address"]
      }]
    }


# BERT processor is the old method used for getting the address, now we use
# the generic GPT based Processor  above
class ProcessorBERT:
  def __init__(self, hf_backbone, np_path):
    print("Loading AddressTagger ...")
    assert "bert" in hf_backbone.lower(), f"Supports only BERT Models, got: {hf_backbone}"

    self.tokenizer = AutoTokenizer.from_pretrained(hf_backbone)
    self.device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    self.model = AutoModel.from_pretrained(hf_backbone).to(self.device)
    self.model.eval()
    self.model_config = self.model.config

    # instead of saving the entire torch model that can be very heavy we merge
    # weights and biases into a single matrix and store as numpy object
    #
    # self.wb.shape = [max_position_embeddings + 1, n] # n classes on which trained
    # w.shape = [max_position_embeddings, n]
    # b.shape = [1, n]
    #
    # you can train any model as you want with the code in the accompanying
    # README file.
    self.wb = np.load(np_path)

    # you can store and load this from any JSON file as well
    self.ID_TO_TAG = {
      0: 'null',
      1: 'name',
      2: 'address',
      3: 'city',
      4: 'state',
      5: 'pincode',
      6: 'phone'
    }
    self.TAG_TO_ID = {v:k for k,v in self.ID_TO_TAG.items()}

    print("... AddressTagger Module Loading Complete")

  def decode_predictions(self, inputs, labels):
    # does opposite of convert_to_tensors(...)
    ids = inputs["input_ids"]
    text = self.tokenizer.decode(ids[1:-1])
    spans = {}
    for i,l in enumerate(labels.tolist()):
      if l != 0: # ignore null_tokens
        if l in spans:
          spans[l].append(i)
        else:
          spans[l] = [i]
          
    ent = {
      self.ID_TO_TAG[k]: self.tokenizer.decode(ids[min(v):max(v) + 1])
      for k,v in spans.items()
    }
    
    for tag in self.TAG_TO_ID:
      if tag != "null" and tag not in ent:
        ent[tag] = None
    
    return {
      "text": text,
      "entities": ent
    }

  def process(self, text):
    # method to perform inference on input text and return industry standard api
    ids = self.tokenizer(text, return_tensors = "pt")
    with torch.no_grad():
      logits = self.model(**ids).last_hidden_state
      
      # now to the linear kernel
      # logits = logits @ weights + bias
      # max_classes = logits.argmax(-1)
      labels = (logits @ self.wb[:-1, :] + self.wb[-1]).argmax(-1)

    ids = {k:v[0] for k,v in ids.items()}
    out = self.decode_predictions(ids, labels[0])
    
    # convert to industry standard response
    text = out["text"]
    entities = []
    for ent in out["entities"]:
      if out["entities"][ent] is None:
        data = ENT_ZERO
        data["entity"] = ent
      else:
        try:
          data = {
            "text": out["entities"][ent],
            "start": text.index(out["entities"][ent]),
            "end": text.index(out["entities"][ent]) + len(out["entities"][ent]),
            "entity": ent
          }
        except:
          # happens because of tokenizer problem
          data = ENT_ZERO
          data["entity"] = ent
      entities.append(data)
    return {
      "text": text,
      "entities": entities
    }
