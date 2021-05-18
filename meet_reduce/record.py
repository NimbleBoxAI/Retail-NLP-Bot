# Inspired from:
# https://stackoverflow.com/questions/40704026/voice-recording-using-pyaudio

import pyaudio
import wave
 

device_index = 2
audio = pyaudio.PyAudio()

print("----------------------DEVICE LIST---------------------")
info = audio.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
  dev_info = audio.get_device_info_by_host_api_device_index(0, i)
  if (dev_info.get('maxInputChannels')) > 0:
    print(f"Input Device id {i} - {dev_info.get('name')}")
print("------------------------------------------------------")

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 512
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "recordedFile.wav"

index = int(input("Which device you want to use: "))
print(f"Recording via index: {index}")

stream = audio.open(
  format=FORMAT,
  channels=CHANNELS,
  rate=RATE,
  input=True,
  input_device_index = index,
  frames_per_buffer=CHUNK
)
print ("recording started")
Recordframes = []
 
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
  data = stream.read(CHUNK)
  Recordframes.append(data)
print ("recording stopped")
 
stream.stop_stream()
stream.close()
audio.terminate()
 
waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(Recordframes))
waveFile.close()
