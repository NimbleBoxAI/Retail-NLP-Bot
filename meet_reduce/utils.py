# define the util functions
import time
import sys
from requests import HTTPError

from .daily import folder
sys.path.append(folder(folder(__file__)))

from pytube import YouTube

def get_caption(x, n, m = 0):
  # get caption for any YouTube video, if available
  # n = total number of retries
  # m = current try number 
  if m == n:
    return None
  source = YouTube(x)
  try:
    cap = source.captions
    cap_keys = list(cap.keys())
    if not cap_keys:
      # this video has no captions, simple list return
      return [None]
    cap = cap[cap_keys[0].code] # def: choose first
    cap_text = cap.generate_srt_captions()
  except HTTPError:
    print("sleeping for a 2 seconds due to connection Error")
    time.sleep(2)
    return get_caption(x, n, m+1)
  return cap_text
