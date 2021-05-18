# MeetReducer

This takes the audio from meetings and automatically builds notes from it.

For this demo we are using the popular "All-In" podcast, but you can record your own clip using `record.py` file.

<a href="https://www.youtube.com/watch?v=rWEPSKkkdKQ">
  <img src="https://i.ytimg.com/vi/DprQR7jhd7A/maxresdefault.jpg" style="height: 400px">
</a>

### Requirements

At the top of `MeetReduce` notebook you'll see the packages need to be installed. For running on NBX platform, please install following packages in this order:

```bash
sudo apt update -y                  # always update OS
sudo apt install libsndfile1 -y     # system package for soundfile
sudo apt install ffmpeg -y          # ffmpeg is <3
pip install youtube_dl pytube      # youtube things
pip install transformers datasets  # models
pip install soundfile pydub        # audio files
```
