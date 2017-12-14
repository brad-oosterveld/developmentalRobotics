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
similarityThreshold = .7

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
  with open("features.txt") as f:
    newline = f.readline()
    while newline != "":
      line = newline.strip().split(" ")
      label = line[0].split(".")[0].split("_crop")
      actualLabel = label[0]
      #actualLabel = label[0]
      data = []
      #if len(label) > 5:
      #  actualLabel += "_" + label[1]
      for elem in line[1:]:
        data.append(float(elem))
      dataset.append([actualLabel,[data]])
      newline = f.readline()
  return dataset

# returns [label, data]
def pc():
  pcFilenames = glob.glob(pcDir +"*")
  dataset = []
  results = dict()
  for fn in pcFilenames:
    #splitFN = fn.strip().split("/")[1].split("_")
    splitFN = fn.strip().split("/")[1].split(".")
    actualLabel = splitFN[0]
    dataset.append([actualLabel, [extractPCFeatures(fn)]])
  return dataset

def calculateNN(data, dataset, data2, dataset2):
  lenData = len(dataset)
  similarities = [[],[],[]]
  for index in xrange(0,lenData):
    [elem,label] = dataset[index]
    [elem2,label2] = dataset2[index]
    pcSim = cosine_similarity(data,elem)
    imgSim = cosine_similarity(data2,elem2)
    similarities[0].append([pcSim, label])
    similarities[1].append([imgSim,label])
    similarities[2].append([(pcSim+imgSim) / 2.0, label])
  return similarities


def classify(sim, actualLabel, results):    
  #print actualLabel, "is classified as", label
  [similarity,label] = sim
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

def myNN(pcInput, imgInput, pcResults, imgResults, cResults):
  pcDataset = []
  imgDataset = []
  for i in xrange (0, len(pcInput)):
  #for i in xrange (0, 1000):
    [pcLabel, pcData] = pcInput[i]
    [imgLabel, imgData] = imgInput[i]
    cNN = calculateNN(pcData, pcDataset, imgData, imgDataset)
    if len(cNN[0]) > 0:
      [pcSim, imgSim, cSim] = cNN
      pcSim.sort()
      pcSim.reverse()
      imgSim.sort()
      imgSim.reverse()
      cSim.sort()
      cSim.reverse()
     # print pcSim[0]
     # print imgSim[0]
     # print cSim[0]
     # print "\n"
      pcResults = classify(pcSim[0], pcLabel, pcResults)
      imgResults = classify(imgSim[0], imgLabel, imgResults)
      cResults = classify(cSim[0], pcLabel, cResults)
      pcDataset.append([pcData,pcLabel])
      imgDataset.append([imgData,imgLabel])
    else:
      pcDataset.append([pcData,pcLabel])
      imgDataset.append([imgData,imgLabel])
      pcResults[pcLabel][0][2] +=1
      pcResults[pcLabel][0][3] +=1
      pcResults[pcLabel][1].append(0)
      pcResults["overall"][0][2] +=1
      pcResults["overall"][0][3] +=1
      pcResults["overall"][1].append(0)
      imgResults[pcLabel][0][2] +=1
      imgResults[pcLabel][0][3] +=1
      imgResults[pcLabel][1].append(0)
      imgResults["overall"][0][2] +=1
      imgResults["overall"][0][3] +=1
      imgResults["overall"][1].append(0)
      cResults[pcLabel][0][2] +=1
      cResults[pcLabel][0][3] +=1
      cResults[pcLabel][1].append(0)
      cResults["overall"][0][2] +=1
      cResults["overall"][0][3] +=1
      cResults["overall"][1].append(0)
  return [pcResults, imgResults, cResults]

def plotIncrementalResults(results,title):
  index = 0
  best = ["r",[0]]
  worst = ["r",[12000000]]
  for r in results:
    for result in r:
      #print results[result][0][3], len(results[result][1])
      if result != "overall":# and len(r[result][1]) >1:
        if r[result][1][-1] > best[1][-1]:
          best[1] = r[result][1]
          best[0] = result
        if r[result][1][-1] < worst[1][-1]:
          worst[1] = r[result][1]
          worst[0] = result
        plt.figure(1) 
        plt.ylabel("Accuracy")
        plt.axis([0,200,0,1])
        plt.xlabel("Number of samples")
        plt.title("Individual Classification")
    if index == 0:
      plt.plot(np.linspace(1,len(best[1]),len(best[1])), best[1], label ="VFH "+best[0], ls="-")
      plt.plot(np.linspace(1,len(worst[1]),len(worst[1])), worst[1], label ="VFH " + worst[0], ls ="-")
    elif index == 1:
      plt.plot(np.linspace(1,len(best[1]),len(best[1])), best[1],label = "Inception v3 "+best[0], ls ="--")
      plt.plot(np.linspace(1,len(worst[1]),len(worst[1])), worst[1],label = "Inception v3 " + worst[0], ls ="--")
    elif index == 2:
      plt.plot(np.linspace(1,len(best[1]),len(best[1])), best[1], label ="Combined "+best[0])
      plt.plot(np.linspace(1,len(worst[1]),len(worst[1])), worst[1],label = "Combined " + worst[0])
    index +=1
  plt.legend(loc=2)
  plt.show()

def plotIncrementalOverall(overallA, overallB, overallC):
  plt.figure(2) 
  plt.ylabel("Accuracy")
  plt.xlabel("Number of samples")
  plt.axis([0,4000,0,1])
  plt.title("Overall Classification")
  plt.plot(np.linspace(1,overallA[0][3],overallA[0][3]), overallA[1], label="VFH")
  plt.plot(np.linspace(1,overallB[0][3],overallB[0][3]), overallB[1], label="Inception v3")
  plt.plot(np.linspace(1,overallC[0][3],overallC[0][3]), overallC[1], label="Combination")
  plt.legend(loc=2)
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
  imgDataset = imgNet()
  pcDataset = pc()

  #sort datasets into smae order
  imgDataset.sort()
  pcDataset.sort()
  pcResults = dict()
  imgResults = dict()
  cResults = dict()
  for i in xrange(0,len(pcDataset)):
    pcLabels = pcDataset[i][0].split("_")
    imgLabels = imgDataset[i][0].split("_")
    pcLabel = pcLabels[0]
    imgLabel = imgLabels[0]
    if len(pcLabels) > 4:
      pcLabel += "_" + pcLabels[1]
      imgLabel += "_" + imgLabels[1]
    pcDataset[i][0] = pcLabel
    imgDataset[i][0] = imgLabel
    if not pcResults.has_key(pcLabel):
      pcResults[pcLabel] = [np.array([0,0,0,0]), []]
      imgResults[pcLabel] = [np.array([0,0,0,0]), []]
      cResults[pcLabel] = [np.array([0,0,0,0]), []]

  pcResults["overall"] = [np.array([0,0,0,0]), []]
  imgResults["overall"] = [np.array([0,0,0,0]), []]
  cResults["overall"] = [np.array([0,0,0,0]), []]


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
  [pcResults,imgResults,cResults] = myNN(pcDataset,imgDataset, pcResults, imgResults,cResults)
  #print "pc\n",pcResults["overall"][1][-1]
  print "pcd"
  analyzeResults(pcResults)
  print "\nimg"
  analyzeResults(imgResults)
  print "\ncombined"
  analyzeResults(cResults)

  #print "\n"

  #(imgSimilarities, imgResults) = myNN(imgDataset,imgResults)
  #print "img\n",imgResults["overall"][1][-1]

  plotIncrementalResults([pcResults,imgResults,cResults], "Best & Worst")
  #plotIncrementalResults(imgResults, "Image")
  #plotIncrementalResults(cResults, "Combined")
  #combinedNN(pcSimilarities,imgSimilarities,combinedResults)

  plotIncrementalOverall(pcResults["overall"], imgResults["overall"], cResults["overall"])
  #print "\n"

  #fusedResults = myNN(fusedDataset, fusedResults)
  #print  "fused\n", fusedResults
