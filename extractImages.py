import glob
from shutil import copy2
import os

workingDir = os.getcwd()
imageDir = "/home/brad/data/rgbd-dataset"

labelFilename = "img_labels.txt"

sampleNumbers = [1, 50, 100, 150]

finalLocation = workingDir + "/image_samples_png/"
print finalLocation

objectTypeDirs = glob.glob(imageDir+"/*")

def extract_images():
  samples = []
  for objectTypeDir in objectTypeDirs:
    objectDirs = glob.glob(objectTypeDir + "/*")
    for objectDir in objectDirs:
      for sampleNumber in sampleNumbers:
        newSamples = glob.glob(objectDir + "/*_*_" +str(sampleNumber) + "_crop.png")
        samples.extend(newSamples)
        for newSample in newSamples:
          copy2(newSample,finalLocation)
  return samples

def store(filenames):
  of = open(labelFilename, "w")
  for fn in filenames:
    objfn = fn.split("/")[-1]
    label = objfn.split("_")[0]
    
    of.write(objfn + ", " + label + "\n")
          
  
      
if __name__ =="__main__": 
  filenames = extract_images()
  # store(filenames)
