from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import glob
import random
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

def calculateNN(data,dataset):
  similarities = []
  for [elem,label] in dataset:
    similarities.append([cosine_similarity(data,elem),label])
  if len(similarities) > 0:
    similarities.sort()
    similarities.reverse()
  return similarities

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
        results[actualLabel] = np.array([0,0,0])
      dataset.append([actualLabel,[data]])
      newline = f.readline()
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
      results[actualLabel] = np.array([0,0,0])
    dataset.append([actualLabel, [extractPCFeatures(fn)]])
  return [dataset,results]

def classify(similarity, actualLabel, label, results):    
  #print actualLabel, "is classified as", label
  if similarity > similarityThreshold:
    if actualLabel == label:
      results[actualLabel][0] +=1
    else:
      results[actualLabel][1] +=1
  else:
    results[actualLabel][2] +=1
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
      results[actualLabel][2] +=1
  return results


def analyzeResults(results):
  total = 0
  totalT = 0
  totalF = 0
  totalN = 0
  for result in results:
    subtotal = results[result].sum()
    [stotalT,stotalF,stotalN] = results[result]
    total += subtotal   
    totalT += stotalT
    totalF += stotalF
    totalN += stotalN
    if not subtotal ==  0:
      print result,"sub true / all : ", stotalT / (float(subtotal))

  print "total true / all : ", totalT / (float(total))
  print "total false / all : ", totalF / (float(total))
  print "total no class / all : ", totalN / (float(total))

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
  print "pc\n",pcResults
  analyzeResults(pcResults)

  print "\n"

  imgResults = myNN(imgDataset,imgResults)
  print "img\n",imgResults
  analyzeResults(imgResults)

  print "\n"

  fusedResults = myNN(fusedDataset, fusedResults)
  print  "fused\n", fusedResults
  analyzeResults(fusedResults)

