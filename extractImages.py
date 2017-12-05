import glob
from shutil import copy2
import os

workingDir = os.getcwd()
imageDir = "tmp"

sampleNumbers = [1, 50, 100, 150]

finalLocation = workingDir + "/image_samples"
print finalLocation

objectTypeDirs = glob.glob(imageDir+"/*")
samples = []

for objectTypeDir in objectTypeDirs:
  objectDirs = glob.glob(objectTypeDir + "/*")
  for objectDir in objectDirs:
    for sampleNumber in sampleNumbers:
      newSamples = glob.glob(objectDir + "/*_*_" +str(sampleNumber) + ".pcd")
      samples.extend(newSamples)
      for newSample in newSamples:
        copy2(newSample,finalLocation)
      
    
