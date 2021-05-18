import re
import os
import hashlib
from glob import glob
from dateutil.parser import parse as date_parse

from .utils import get_caption

sha256 = lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()
md5 = lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()

class Processor():
  def __init__(self):
    here = os.path.split(os.path.abspath(__file__))[0]
    cap_folder = os.path.join(here, 'captions')
    all_cap_files = glob(f"{cap_folder}/*.srt")
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
    return captions

  def process(self, url, max_tries = 20):
    if not md5(url) in self.all_cap_files:
      # get captions and return if there is some error
      caption = get_caption(url, max_tries)
      if isinstance(caption, list):
        return f"[This]({url}) video has no captions"
      if caption is None:
        return f"Failed to fetch captions for [this]({url}) video."
    else:
      with open(self.all_cap_files[md5(url)], "r") as f:
        caption = f.read()

    # parse caption string into caption blocks
    caption = self.parse_captions(caption)
    return caption[0]
