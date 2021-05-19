import re
import os
# multithreading problems man!
# fixes the huggingface multithreading so it does not cause issues with streamlit
# multithreading
os.environ["TOKENIZERS_PARALLELISM"] = "0"

import streamlit as st

# this caches the output to store the output and not call this function again
# and again preventing time wastage. `allow_output_mutation = True` tells the
# function to not hash the output of this function and we can get away with it
# because no arguments are passed through this.
# https://docs.streamlit.io/en/stable/api.html#streamlit.cache
@st.cache(allow_output_mutation = True)
def get_processors():
  from address_tagger.model import Processor as AddressTaggerProcessor
  from meet_reduce.summarizer import Processor as MeetReduceSummarizerProcessor
  return {
    # "AddressTagger": AddressTaggerProcessor("distilbert-base-cased", "./address_tagger/params.npy"),
    "MeetReduce": MeetReduceSummarizerProcessor()
  }

# load all the processors before the app starts
PROCESSORS = get_processors()

st.write('''
# Retail NLP Bots

This is an interactive webapp to use the agents created by [NimbleBox.ai](https://nimblebox.ai/).
There are three different categories available:
- Address Tagger: this takes in the Indian addresses, which are highly unstructured compared to its
  western counterparts and returns the split data
- AutoInvoice [WIP]: To take the unstructured string for invoice and automatically structure it
  - Catalog orders: when the given order is already in a predefined catalog
  - Custom orders: when the input text is highly custom
- MeetReduce: **Tired to super long meetings, need quick summary?** Put any YouTube link and summarize
  captions in a few minutes.
''')

appname = st.sidebar.selectbox(
  'Which app do you want to use',
  ["None", "AddressTagger", "Catalog Order", "Custom Order", "MeetReduce"],
  help = "Select Demo"
)

# current bypass, to be removed at final delivery - @yashbonde
if appname not in ["MeetReduce", "AddressTagger"]:
  st.write("Other apps are WIP. Why don't you use `AddressTagger`, it's interactive 💥")
  appname = "MeetReduce"

if appname == "AddressTagger":
  # run the Address bot here
  st.write('''### `AddressTagger` Module
  Break your input text into correct entities''')
  processor = PROCESSORS["AddressTagger"]

  # add the text and control thingys
  default_ = "By: Ankush Agarwal, Fl. 4, Seaside Apartment, Airoli, Mumbai, Maharashtra \n\n Mobile : 9834529842"
  input_text = st.text_input("Input Address", value = default_, key = "input_text")
  if st.button("process"):
    data = processor.process(input_text)
    st.write(data)

elif appname == "MeetReduce":
  st.write('''### `MeetReduce` Module
  Summarize any YouTube video captions in minutes. Check top-right while it is processing!''')
  processor = PROCESSORS["MeetReduce"]

  default_ = "https://www.youtube.com/watch?v=rWEPSKkkdKQ"
  input_text = st.text_input("YouTube Link", value = default_, key = "input_text")
  if st.button("process"):
    if not re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", input_text):
      st.write(f"Please enter correct URL, got: `{input_text}`")
    else:
      summ, heights = processor.process(input_text)
      st.area_chart(heights)
      st.write(summ)
