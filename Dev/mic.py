import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100 #Sample rate
seconds = 5 #Duration of recording

recording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait() #Wait until recording is finished
write('output.wav',fs,recording) #Save as WAV file