import os.path

imgnames = []
for fname in os.listdir("image_samples_png"):
    if fname.endswith("crop.png"):
        id= fname[:fname.rindex('_')]
        imgnames.append(id)

print ("imgnames: "+ str(len(imgnames)))



pcdnames = []
for fname in os.listdir("pcd_features"):
    if fname.endswith(".pcd"):
        id = fname[:fname.rindex('.')]
        pcdnames.append(id)


print ("pcdnames: "+ str(len(pcdnames)))

imgnames.sort()
pcdnames.sort()

diff = list(set(imgnames)-set(pcdnames))
print ("diff size"+str(len(diff)))
for d in diff:
    print(d)