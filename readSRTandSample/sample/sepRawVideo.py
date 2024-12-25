import numpy as np

file='0_series.npy'
totalTimeLength=300

data=np.load(file) #timeStepLength, numStudents, 10, 2
dataShape=data.shape
dataNew=[]
i=0
while True:
   if i+totalTimeLength <= dataShape[0]:
      temp=np.zeros((dataShape[1],2,300,10,1))
      temp[:,:,:totalTimeLength,:,:]=data[i:i+totalTimeLength].transpose(1,3,0,2)[:,:,:,:,None]
   else:
      temp=np.zeros((dataShape[1],2,300,10,1))
      T=dataShape[0]-i
      temp[:,:,:T,:,:]=data[i:dataShape[0],:,:,:].transpose(1,3,0,2)[:,:,:,:,None]
   dataNew.append(temp)
   i+=totalTimeLength
   if i>dataShape[0]-1:
      break
dataNew=np.concatenate(dataNew,axis=0)
np.save('eval_data.npy',dataNew)      
      