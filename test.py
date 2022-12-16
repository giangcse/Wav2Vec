import numpy as np
from pydub import AudioSegment
import librosa

sound = AudioSegment.from_file("/home/pylab/Desktop/Wav2Vec/test.mp3")
sound = sound.set_frame_rate(16000)
channel_sounds = sound.split_to_mono()
samples = [s.get_array_of_samples() for s in channel_sounds]

fp_arr = np.array(samples).T.astype(np.float32)
fp_arr /= np.iinfo(samples[0].typecode).max

y_mono = librosa.to_mono(fp_arr.T)
print(y_mono.shape)