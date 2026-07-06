DURATION = 3.0 # do not change
WIDTH = HEIGHT = 224 # 300 was used for humans
FRAMERATE = 20 # do not change
NBFRAMES = int(DURATION * FRAMERATE)

# video encoding
FRAMES_DNN = [0.0, 0.2, 0.4, 0.6, 0.8,
              1.0, 1.2, 1.4, 1.6, 1.8,
              2.0, 2.2, 2.4, 2.6, 2.8,
              2.95] # last timestamp added after a shorter interval

# audio encoding
SAMPLERATE_DNN = 20000
NBCHANNELS_DNN = 1
BITRATE_DNN = SAMPLERATE_DNN * NBCHANNELS_DNN * 16 # given encoding_format = "pcm_s16le"