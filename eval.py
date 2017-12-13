from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import glob
import random
import matplotlib.pyplot as plt
from threading import Thread
#from multiprocessing import Process
#evaluation plan

#gross evaluation:
#thresholded knn, done incrementally on the entire corpus, in a a rondom order
#for each ordering
#for each element : find nearest nighbor(s) if distance above a given threshold: the robot woul dneed to "ask" what the object is
#record number of "asks" as well as false positives
#compare each feature extraction method as well as combinations of methods


#the order of the clusters probably will matter so...
#for each category:
# calculate the cluster shilouette of the whole cluster
# for each oredering of the objects in the cluster
# incrementally add each element and calculate the cluter shilouette.
# average  shilouette size across orerings? and plot number of isntances vs total shilouette

#there will probably also be an effect of if the object category is accounted for in the training data
#do some analysis comparison - maybe just descriptive statistics of categories in the training data vs not in the training data


knn = 5
pcDir = "pcd_features/"
similarityThreshold = .5

def extractPCFeatures(filename):
  f = open(filename, "r")
  data = f.readlines()[-1].strip().split(" ")
  out = []
  for elem in data:
    out.append(float(elem))
  return out


# returns [label, data]
def imgNet():
  dataset = []
  results = dict()
  with open("features.txt") as f:
    newline = f.readline()
    while newline != "":
      line = newline.strip().split(" ")
      label = line[0].split("_")
      actualLabel = label[0]
      data = []
      if len(label) > 5:
        actualLabel += "_" + label[1]
      for elem in line[1:]:
        data.append(float(elem))
      if not results.has_key(actualLabel):
        results[actualLabel] = [np.array([0,0,0,0]), []]
        #results[actualLabel] = (np.array([0,0,0]))
      dataset.append([actualLabel,[data]])
      newline = f.readline()
    results["overall"] = [np.array([0,0,0,0]), []]
  return [dataset,results]

# returns [label, data]
def pc():
  pcFilenames = glob.glob(pcDir +"*")
  dataset = []
  results = dict()
  for fn in pcFilenames:
    splitFN = fn.strip().split("/")[1].split("_")
    actualLabel = splitFN[0]
    if len(splitFN) > 4:
      actualLabel = actualLabel + "_" + splitFN[1]
    if not results.has_key(actualLabel):
      results[actualLabel] = [np.array([0,0,0,0]), []]
      #results[actualLabel] = (np.array([0,0,0]))
    dataset.append([actualLabel, [extractPCFeatures(fn)]])
    results["overall"] = [np.array([0,0,0,0]), []]
  return [dataset,results]

def calculateNN(data,dataset):
  lenData = len(dataset)
  similarities = []
  ts = []
  for index,[elem,label] in enumerate(dataset):
    similarities.append([])
    #t = Thread(target=threadMethod, args=(data,elem,label,similarities,index), name = index)
    #t.start()
    #ts.append(t)
    threadMethod(data,elem,label,similarities,index)
  for t in ts:
    t.join()
  if len(similarities) > 0:
    similarities.sort()
    similarities.reverse()
  return similarities

def threadMethod(data,elem,label,similarities,index):
    similarities[index] = [cosine_similarity(data,elem),label]
  

def classify(similarity, actualLabel, label, results):    
  #print actualLabel, "is classified as", label
  overall = "overall"
  if similarity > similarityThreshold:
    if actualLabel == label:
      results[actualLabel][0][0] +=1
      results[overall][0][0] +=1
    else:
      results[actualLabel][0][1] +=1
      results[overall][0][1] +=1
  else:
    results[actualLabel][0][2] +=1
    results[overall][0][2] +=1

  results[actualLabel][0][3] +=1
  results[overall][0][3] +=1
  results[actualLabel][1].append(results[actualLabel][0][0] / float(results[actualLabel][0][3]))
  results[overall][1].append(results[overall][0][0] / float(results[overall][0][3]))
  return results

# returns results dict (label: [true, false, not classified]
def myNN(inputDataset, results):
  dataset = []
  for [actualLabel,data] in inputDataset[:]:
    nns = calculateNN(data,dataset)
    if len(nns) > 0:
      [similarity, guessLabel] = nns[0]
      results = classify(similarity, actualLabel, guessLabel, results)
      dataset.append([data,actualLabel])
    else:
      dataset.append([data,actualLabel])
      results[actualLabel][0][2] +=1
      results[actualLabel][0][3] +=1
      results[actualLabel][1].append(0)
      results["overall"][0][2] +=1
      results["overall"][0][3] +=1
      results["overall"][1].append(0)
  return results

def plotIncrementalResults(results,title):
  for result in results:
    #print results[result][0][3], len(results[result][1])
    plt.figure(1) 
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Number of samples")
    plt.title(title + " Individual Classification")
    if result == "overall":
      plt.figure(2)
      plt.title(title + " Overall Classification")
      plt.ylabel("Accuracy (%)")
      plt.xlabel("Number of samples")
    plt.plot(np.linspace(1,results[result][0][3],results[result][0][3]), results[result][1])
  plt.show()


def analyzeResults(results):
  for result in results:
    [stotalT,stotalF,stotalN,subtotal] = results[result][0]
    if not subtotal ==  0:
      print result, "total: ", subtotal
      print "\tcorrect classification: ", stotalT
      print "\tincorrect classificion: ", stotalF
      print "\tno classification: ", stotalN

if __name__== "__main__":
  [imgDataset, imgResults] = imgNet()
  [pcDataset, pcResults] = pc()

  #sort datasets into smae order
  imgDataset.sort()
  pcDataset.sort()

  #shuffle into same random order
  combined=list(zip(imgDataset, pcDataset))
  random.shuffle(combined)

  #join both features together
  fusedDataset=[]
  fusedResults=pcResults
  for pair in combined:
    #get label
    # fusion = [pair[0][0]]
    #append values
    feat1 = pair[0][1][0]
    expd = np.exp(feat1)
    denom = np.sum(expd)
    normed = np.divide(expd,denom)
    feat2 = pair[1][1][0]
    combo = feat1+normed
    fusion= [pair[0][0],[combo]]
    fusedDataset.append(fusion)

  # split back into orignals
  imgDataset, pcDataset = zip(*combined)

  # print imgDataset
  # print pcDataset
  # print fusedDataset

  pcResults = myNN(pcDataset,pcResults)
  print "pc\n",pcResults["overall"]
  analyzeResults(pcResults)

  plotIncrementalResults(pcResults, "Point Cloud")
  print "\n"

  imgResults = myNN(imgDataset,imgResults)
  #print "img\n",imgResults
  analyzeResults(imgResults)

  plotIncrementalResults(imgResults, "Image")

  print pcResults
  print imgResults
  #print "\n"

  #fusedResults = myNN(fusedDataset, fusedResults)
  #print  "fused\n", fusedResults
  #analyzeResults(fusedResults)
