from os import walk
import shutil
from math import sqrt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import random

### image processing filters ###

def binarizare(img):
  s=img.shape
  img2=np.zeros((s[0],s[1]))
  img=img.astype('float')
  if len(s)==3 and s[2]==3:
    img_gri=(0.299*img[:,:,0]+0.587*img[:,:,1]+0.114*img[:,:,2])
    img_gri=np.clip(img_gri,0,255)
  else:
    img_gri=img
  for i in range(s[0]):
    for j in range(s[1]):
      if img_gri[i,j]<50:
        img2[i,j]=(0/50)*img_gri[i,j]
      elif img_gri[i,j]>50 and img_gri[i,j]<150:
        img2[i,j]=0+((255-0)/(150-50))*(img_gri[i,j]-50)
      else:
        img2[i,j]=255+((255-255)/(255-150))*(img_gri[i,j]-150)
  return img2


def clip(img):
  s=img.shape
  img3=np.zeros((s[0],s[1]))
  img=img.astype('float')
  if len(s)==3 and s[2]==3:
    img_gri=(0.299*img[:,:,0]+0.587*img[:,:,1]+0.114*img[:,:,2])
    img_gri=np.clip(img_gri,0,255)
  else:
    img_gri=img
  for i in range(s[0]):
    for j in range(s[1]):
      if img_gri[i,j]<100 or img[i,j]>200:
        img3[i,j]=0
      else:
        img3[i,j]=40+((220-40)/(200-100))*(img_gri[i,j]-100)
  return img3


def exp(img):
  img1=255**(img/255)-1
  return img1

def modelTrain(tdata, tsdata, vdata, tlabels, tslabels, vlabels):

    print("model starts training ... \n")
    LR = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr', max_iter = 1000)
    LR.fit(tdata, tlabels)
    print ("training done ... \n")

    print("model starts predicting ... \n")
    vlabels_predicted = LR.predict(vdata)
    print("prediction done .... \n")
    vacc = accuracy_score(vlabels, vlabels_predicted)
    print ("Accuracy after validation = " + str(vacc))


    print("model starts predicting ... \n")
    tslabels_predicted = LR.predict(tsdata)
    print("prediction done .... \n")
    tsacc = accuracy_score(tslabels, tslabels_predicted)
    print ("Accuracy after testing = " + str(tsacc))
    print ("Confusion matrix :\n", confusion_matrix (np.array(tslabels), np.array(tslabels_predicted)))

if __name__ == "__main__":

### file parsing ###

    path_to_folder = "C:/Users/Iustin/Desktop/proiect_tsim"
    path_to_label_file = "C:/Users/Iustin/Desktop/labels.txt"

    labels = []
    images = []

    file = open(path_to_label_file, 'r')
    lines = file.readlines()
    file.close()

    for line in lines:
        list = line.split(' ')
        if '\n' in list[1]:
            list[1] = list[1][:-1]
        images.append(path_to_folder + "/label" + list[1] + "/" + list[0] + '.pgm')
        labels.append(list[1])

    for i in range(len(images)):
        print(images[i] + " with label " + labels[i])

### image reading and filter addition ###

    features = []
    for i in range (len (images)):
        img = plt.imread (images[i])
        img1 = exp (img)
        img2 = binarizare (img)
        img3 = clip (img)
        print ("Processing image " + images[i])
        features.append ([img, img1, img2, img3])

### shuffling the images ###

    vtemp = list (zip (features, labels))
    random.shuffle (vtemp)
    rimages, rlabels = zip (*vtemp)

### database splitting : train : 70 validation : 15 test : 15

    dim = len (images)

    start = 0
    end = int (0.7 * dim)
    train_img = rimages[start:end]
    train_lbl = rlabels[start:end]

    start = int (0.7 * dim)
    end = int (0.85 * dim)
    validare_img = rimages[start:end]
    validare_lbl = rlabels[start:end]

    start = int (0.85 * dim)
    test_img = rimages[start:]
    test_lbl = rlabels[start:]


    modelTrain(train_img, test_img, validare_img, train_lbl, test_lbl, validare_lbl)