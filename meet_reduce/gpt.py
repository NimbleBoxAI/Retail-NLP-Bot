# this file has all the code for running this demo, for latest updates pull from:
# https://gist.github.com/yashbonde/cadb515b6c658f18147d948fac685c7b

import torch
import numpy as np
import re
from time import sleep
from tqdm import trange

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

  def set_seed(self, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

# ---- pipelines

def check_english_language(sentence, gpt):
  if isinstance(sentence, list):
    return [
      check_english_language(s) for s in sentence
    ]
  prompt = """This classifies whether the input sequence into it's language
###
sentence: "GPT2 is a state of the art neural language model, that can be trained to make good quality predictions for many different tasks including text classification and machine translation."
language: English
###
sentence: "कोरोना पर होगा ड्रोन अटैक: ICMR की योजना- दुर्गम इलाकों में ड्रोन से होगी वैक्सीन की डिलीवरी, तेलंगाना सरकार ने ऐसा प्रोजेक्ट लॉन्च किया"
language: Hindi
###
sentence: "фантастический роман Алексея Николаевича Толстого о путешествии землян на Марс. Текст написан в основном в эмиграции, первое издание вышло в Петрограде в 1923 году и неоднократно перепечатывалось."
language: Russian
###
sentence: {sentence}
language:"""
  p = prompt.format(sentence = sentence)
  out = gpt(p, n = 4, r = 1, stop_sequence="###", temp=0.7, do_sample = False)[0]
  out = out[len(p):].strip().split("\n")[0]
  return out


def format_sentence(sentence, gpt):
  if isinstance(sentence, list):
    out = []
    for s in sentence:
      out.append(format_sentence(s))
      sleep(0.5)
    return out

  # try keeping len(sentence.split()) ~ 54
  gpt.set_seed(90)
  prompt = """Correct the sentence in each input and return properly formatted sentence
###
sentence: "everyone in this part of the world thinks i am a fraud but i know who i am."
correct: "Everyone in this part of the world thinks I am a fraud, but I know who I am."
###
sentence: "hey everybody welcome to the all in podcast it was a slow news week so we decided we'd  give you a special episode we're gonna go around the horn with our special picks we're each gonna."
correct: "Hey everybody, welcome to the all in podcast. It was a slow news week so we decided we'd give you a special episode. We're gonna go around the horn with our special picks, we're each gonna"
###
sentence:"{sentence}"
correct:"""

  p = prompt.format(sentence = sentence)
  n = len(gpt.tokenizer.tokenize(sentence)) + 10 # margin of error
  g = gpt(p, n, r=1, stop_sequence = "\n", temp = 1.0, top_p = 1.0)[0]
  # clean up the response
  s = g.split("###")[-2 if g.endswith('###') else -1]
  res = s.split("\n")[2][8:].replace('"', '').strip()
  return res[:-1]

def get_keywords(text, gpt, r = 4):
  prompt = """Get keywords from each text

###

Text: "Black-on-black ware is a 20th- and 21st-century pottery tradition developed by the Puebloan Native American ceramic artists in Northern New Mexico. Traditional reduction-fired blackware has been made for centuries by pueblo artists. Black-on-black ware of the past century is produced with a smooth surface, with the designs applied through selective burnishing or the application of refractory slip. Another style involves carving or incising designs and selectively polishing the raised areas. For generations several families from Kha'po Owingeh and P'ohwhóge Owingeh pueblos have been making black-on-black ware with the techniques passed down from matriarch potters. Artists from other pueblos have also produced black-on-black ware. Several contemporary artists have created works honoring the pottery of their ancestors."

Keywords: Pueblo, art, pottery, black, black ware

###

Text: "{text}"

keywords:"""

  p = prompt.format(text = text)
  words = set()
  
  # prevents GPU OOM by batching instead of parallel requests
  for _ in range(0, r+1, 2):
    out = gpt(
      p, r = 2, n = 20, stop_sequence="\n", temp = 0.9,
      repetitive_penalty = 0.9 # don't repeat the same thing over and over again
    )

    for s in out:
      ws = s.split("###")[2].split("\nkeywords:")[-1].split(",")
      for w in ws:
        w = w.strip().lower()
        # print("--->", w, w in ["pueblo", "art", "pottery", "black", "black ware"])
        if w and len(w.split()) < 4 and w not in ["pueblo", "art", "pottery", "black", "black ware"]:
          words.add(w)
  return list(words)


def clean_keywords(keywords, gpt):
  if isinstance(keywords[0], list):
    return [clean_keywords[x] for x in keywords]

  # make the keywords a sentence for prompt
  k = ", ".join(keywords)
  k = re.sub(r"[^\w\'\s,]", "", k)
        
  prompt = '''This app removed duplicate words from a list of words

###

Sentence: "reddity, redditt, a red list, the, redditor, culture, community, redditor list, a redditor, reddiquette, redditr, queer, r/quinoa, quinoa, reddit, redditer, reddits"

Important words: reddit, culture, community, quinoa, r/quinoa

###

Setence: "podcasts, podcast, episode, all in, all i, special podcast, all in podcast, all ian, in, all i"

Important words: all in podcast

###

Sentence: "street, wall, stock, wall street, hedge fund, stock pick, thesis, stock analyst, bet, obey, angle, stock tip, quote, new york, gamestop"

Important words: wall street, hedge fund, new york, gamestop

###

Sentence: {sentence}

Important words:'''
    
  p = prompt.format(sentence = k)

  out = gpt(p, n = 32, r = 5, temp = 1.0, top_p = 0.9, stop_sequence="\n",)
  
  words = set()
  for x in out:
    ws = x.split("###")[-1].split("Important words:")[-1].strip().split(",")
    for w in ws:
      words.add(w.strip())
  return list(words)

# ----- main functions

def get_keywords(captions: list, gpt: GPT):
  """
  Args:
    captions (list): captions is the list with output of function `Processor.parse_captions()`
      each element having the following structure:
      {
        "id": [<id: int>],
        "from": <from:datetime>,
        "to": <to:datetime>,
        "content": <str>
      }
    gpt (GPT): loaded GPT wrapper

  Returns:
      [type]: [description]
  """

  capstr = " ".join([x["content"] for x in captions])

  # step 1: format the sentence
  format_buff_size = 100
  fsent = []
  for i in range(0, len(capstr), format_buff_size):
    r = format_sentence(" ".join(capstr.split()[i:i+format_buff_size]) + ".", gpt)
    fsent.append(r)

  # step 2: get keywords
  words = []
  for _, o in zip(trange(len(fsent)), fsent):
    words.append(get_keywords(o, gpt))
  
  # step 3: clean the key-words
  w2 = []
  for w in words:
      out = clean_keywords(w, gpt)
      w2.append(out)

  return w2
