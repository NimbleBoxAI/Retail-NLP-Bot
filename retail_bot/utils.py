import os
import torch
import numpy as np

# ----- https://gist.github.com/yashbonde/62df9d16858a43775c22a6af00a8d707
def folder(x):
  # get the folder of this file path  
  return os.path.split(os.path.abspath(x))[0]

# ----- https://gist.github.com/yashbonde/cadb515b6c658f18147d948fac685c7b
class Response():
  """Class that makes getting generated results chill, simply `print(out)`"""
  def __init__(self, out, t):
    self.t = t
    self.sequences = out.sequences.cpu().tolist()
    self.scores = [x.cpu().numpy() for x in out.scores] if out.scores != None else None
    self.hidden_states = [
      [y.cpu().numpy() for y in x]
      for x in out.hidden_states
    ] if out.hidden_states != None else None
    self.attentions = [
      [y.cpu().numpy() for y in x]
      for x in out.attentions
    ] if out.attentions != None else None

    self.decoded = self.t.batch_decode(self.sequences, skip_special_tokens = True)

  def __repr__(self):
    str_ = ""
    for x in self.decoded:
      str_ += x + "\n"
      str_ += "-"* 70 + "\n"
    return str_

  def __len__(self):
    return len(self.decoded)

  def __getitem__(self, i):
    return self.decoded[i]
  
  def __iter__(self):
    for x in self.decoded:
      yield x


class GPT():
  """Make GPT a first class object and using it as simple as possible.
  First define the model and tokenizer
  
  >>> device = torch.device("cuda:0") if torch.cuda.is_available() else "CPU"
  >>> tokenizer = AutoTokenizer.from_pretrained(name)
  >>> model = AutoModelForCausalLM.from_pretrained(name, cache_dir = "../hf-cache/").eval().to(device)
  Make GPT wrapper: output is `Response`, a class with __repr__ overloaded so print gives the generation
  >>> gpt = GPT(model, tokenizer)
  >>> out = gpt("Hello world", n = 10, r = 2)
  >>> out
  ... Hello world!" "Hello?" "Yeah, I'm in
    ----------------------------------------------------------------------
    Hello world" command, that is, the command that
    ----------------------------------------------------------------------
  """
  def __init__(self, model, tokenizer):
    self.model = model
    self.tokenizer = tokenizer
    self.eot_id = tokenizer.eos_token_id
    self.device = self.model.device

  @torch.no_grad()
  def __call__(
    self,
    prompt: str,
    n: int = 16, # number of tokens
    r: int = 1, # number of sequences
    do_sample = True,
    temp = 0.9,
    top_p = 0.9,
    top_k = None,
    output_scores = None,
    output_hidden_states = None,
    output_attentions = None,
    stop_sequence = None,
    return_response = True,
    **gen_kwargs
  ):
    """ __call__ overloader initialises the model.generate() function. We emphasise
    a lot more on most powerful arguments, but you can always pass custom kwargs through
    `gen_kwargs`. As you can see that we have not added many beam-search related arguments.
    Args:
      prompt (str): prompt string, tokens will be generated in continuation
      n (int, optional): number of tokens to return
      r (int, optional): number of sequences to return
      temp (float, optional): sampling temperature
      top_p (float, optional): tokens whose probability adds up to this are considered
      top_k (int, optional): top-k tokens to consider for each distribution
      output_scores (bool, optional): output scores for each generted token, returns shape `[r,n]`
      output_hidden_states (bool, optional): output the hidden states of the generation, returns shape `[r,n+1,...]`
      output_attentions (bool, optional): Whether or not to return the attentions tensors of all attention layers
      stop_sequence (str, optional): Stop generation once the first token of this string is achieved
      return_response (bool, optional): To parse the generated dictionary to `Response` class
      gen_kwargs (dict, optional): any extra arguments to pass to the model.generate() method
    Returns:
      if return_response:
       Response instance
      else:
       model.generate() output
    """
    t = self.tokenizer
    m = self.model
    
    # tokenize the input prompt and stop token if provided
    input_ids = t(prompt, return_tensors = "pt")["input_ids"].to(self.device)
    if stop_sequence is not None:
      eos_token_id = t(stop_sequence)["input_ids"][0]
    else:
      eos_token_id = self.eot_id
      
    # generate the items
    out = m.generate(
      input_ids,
      max_length = len(input_ids[0]) + n,
      temperature = temp,
      top_p=top_p,
      top_k=top_k,
      num_return_sequences=r,
      pad_token_id = self.eot_id,
      output_scores = output_scores,
      output_hidden_states = output_hidden_states,
      output_attentions = output_attentions,
      do_sample = do_sample,
      return_dict_in_generate = True,
      eos_token_id = eos_token_id,
      **gen_kwargs
    )
    
    # return items or 
    if return_response:
      return Response(out, t)
    else:
      return out

  def classify(
    self,
    prompt: str,
    labels: list,
    softmax_temp = 0.9,
    add_unknown = False,
    **gen_kwargs,
  ) -> dict:
    """Perform classification directly.
    NOTE: ensure that first tokens in labels are not the same.
    Args:
      prompt (str): prompt string to be given as input
      labels (list): list of strings that are labels
      gen_kwargs (dict, optional): extra arguments to be passed for generation
      softmax_temp (float, optional): temprature for scoring labels. Defaults to 0.9.
      add_unknown (bool, optional): adds an extra "Unknown" label. Defaults to False.
    Returns:
      dict: values are 0. if model returns 'nan'
    """
    # we will use the same format that OpenAI uses for GPT-3
    # read: https://beta.openai.com/docs/guides/classifications
    # We normalize all labels by `label.strip().lower().capitalize()` at the API
    # backend. Thus corresponding output labels are always capitalized.
    unq_options = set([x.strip().lower().capitalize() for x in labels])
    unq_options = sorted(list(unq_options))

    # each label must have a distinct first token, because classification
    # works by looking only one step ahead. Also encode the labels with extra
    # white space prepended.
    label_ids = [self.tokenizer.encode(" " + x)[0] for x in unq_options]
    
    out = self(prompt, n = 1, r = 1, output_scores = True, **gen_kwargs, return_response = False)
    logits = out.scores[0][0]
    logits = (logits / softmax_temp)[label_ids].softmax(-1).cpu()
    logits = logits.numpy()

    scores = {o:i for o,i in zip(unq_options, logits)}
    
    # naaaaan - check
    scores = {k: 0. if np.isnan(l) else l for k,l in scores.items()}

    if add_unknown:
      # fill the Probability for the special "Unknown" token
      scores["Unknown"] = 1 - sum(scores.values())
    
    return scores