import numpy as np
from scipy.io.wavfile import write, read

norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 'int64':INT64_FAC,'float32':1.0,'float64':1.0}

fs, x = read('sample_audio.wav')

#scale down and convert audio into floating point number in range of -1 to 1
x_norm = np.float32(x)/norm_fact[x.dtype.name]
  
# x_norm *= INT16_FAC                               # scaling floating point -1 to 1 range signal to int16 range
# x_norm = np.int16(x)                              # converting to int16 type

write('sample_audio_normalized.wav', fs, x_norm)
