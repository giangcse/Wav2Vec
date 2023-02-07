import time

filename = 'abc.wav'

print(str(filename).split('.wav')[0] + '_' + str(int(time.time())) + '.wav')