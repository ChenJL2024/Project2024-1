import numpy as np
import random

behavType='normal'

samples=np.load(behavType+'.npy')
num=samples.shape[0]
samples=samples[random.sample(range(num),int(num*0.5416))]
np.save(behavType+'_randomSampled.npy',samples)
