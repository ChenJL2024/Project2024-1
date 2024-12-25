import os
import glob

files = glob.glob(pathname='./*.mp4')
Note=open('./mapFileName.txt',mode='r')
while True:
  content=Note.readline()[:-1]
  if content.strip()=='':
     break
  if '.\\'+content+'.mp4' in files:
     os.rename('.\\'+content+'.mp4',Note.readline()[2:-1])
Note.close()
  