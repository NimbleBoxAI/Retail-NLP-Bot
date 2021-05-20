
import os
import random
import hashlib
import datetime
from glob import glob
from dateutil.parser import parse as date_parse

from .utils import get_caption

# add a bunch of tiny code snippets, downloads daily.py to the master process __file__ location
try:
  from daily import *
except ImportError as e:
  import requests
  x = requests.get("https://gist.githubusercontent.com/yashbonde/62df9d16858a43775c22a6af00a8d707/raw/0764da94f5e243b2bca983a94d5d6a4e4a7eb28a/daily.py").content
  with open("daily.py", "wb") as f:
    f.write(x)
  from daily import *

class Processor():
  def __init__(self):
    here = folder(__file__)
    self.cap_folder = os.path.join(here, 'captions')
    all_cap_files = glob(f"{self.cap_folder}/*.srt")
    self.all_cap_files = {x.split('/')[-1][:-4]: x for x in all_cap_files}

  def parse_captions(self, caption_string):
    # next we parse the captions and structure them
    captions = []
    for x in caption_string.split("\n\n"):
      _id, _time, _content = x.split("\n")
      _time = _time.split("-->")
      _from = date_parse(_time[0])
      _to = date_parse(_time[1])
      # \xa0 is actually non-breaking space in Latin1 (ISO 8859-1), also chr(160)
      _content = _content.replace(u'\xa0', u' ').strip()
      captions.append({"id": _id, "from": _from, "to": _to, "content": _content})
    
    # merge captions that come under 1 second of each other
    sections = []
    buffer = [[captions[0]]] # rolling buffer
    for i in range(len(captions) - 1):
      x = captions[i+1]
      last_cap = buffer[-1][-1]
      if x["from"] - last_cap["to"] < datetime.timedelta(0, 1):
        buffer[-1].append(x)
      else:
        buffer.append([x])
        
    merged_captions = []
    for b in buffer:
      merged_captions.append({
        "id": [x["id"] for x in b],
        "from": b[0]["from"],
        "to": b[-1]["to"],
        "content": " ".join([x["content"] for x in b])
      })

    return merged_captions

  def process(self, url, max_tries = 20):
    _file = Hashlib.md5(url)
    if not _file in self.all_cap_files:
      # get captions and return if there is some error
      caption = get_caption(url, max_tries)
      if isinstance(caption, list):
        return f"[This]({url}) video has no captions"
      if caption is None:
        return f"Failed to fetch captions for [this]({url}) video."
      else:
        # save and cache captions
        fp = f"{self.cap_folder}/{_file}.srt"
        with open(fp, "w") as f:
          f.write(caption)
        self.all_cap_files[_file] = fp
    else:
      with open(self.all_cap_files[_file], "r") as f:
        caption = f.read()

    # parse caption string into caption blocks
    captions = self.parse_captions(caption)
    heights = [] # word density plot
    for x in captions:
      heights.extend([len(x["content"].split()), ] * len(x["id"]))
    return random.choice(captions), heights
    
