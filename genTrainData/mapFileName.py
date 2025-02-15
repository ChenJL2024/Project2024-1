import os
import glob

files = glob.glob(pathname='./*.mp4')
Note=open('./mapFileName.txt',mode='w')

i=-1
for file in files:
   i+=1
   os.rename(file,f'{i}.mp4')
   Note.write(f'{i}\n')
   Note.write(file+'\n')
Note.close()