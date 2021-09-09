# Retail-NLP-Bot

Simple NLP agents to solve everyday tasks in retail like AddressTagger. Find individual models and corresponding notebooks in each folder:
- [`retail_bot/`](./retail_bot/): Retail bot with the following items:
  - "AddressTagger"
  - "Catalog Order"
  - "Custom Order"
- [`meet_reduce/`](./meet_reduce/): App to search through meeting notes

Files:
- `run.py`: streamlit webapp for this example

### Usage

Two commands and you are good to go:
```
# install packages
$ pip install -r requirements.txt

# run the interactive webapp
$ streamlit run run.py

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.4:8501

```

If everything goes correctly, it should automatically open up a browser with the network URL. On the left you will see the build apps, select the one that you want to use.

<img src="./usage.gif">
