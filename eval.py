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
