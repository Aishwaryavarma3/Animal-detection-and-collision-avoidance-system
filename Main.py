from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pickle
import os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import seaborn as sns
import cv2
import math
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import winsound

main = Tk()
main.title("A Practical Animal Detection and Collision Avoidance System Using Computer Vision Technique")
main.geometry("1300x1200")

global filename
global dataset
global X, Y
global X_train, X_test, y_train, y_test, labels
global hog_model
labels = []
frozen_model = 'model/hog_cascade.pb'
config_file = 'model/labels.pbtxt'

for root, dirs, directory in os.walk("Dataset"):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name.strip())

def getLabel(name):
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

#fucntion to upload dataset
def uploadDataset():
    global filename, filename, X, Y
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".") #upload dataset file
    text.insert(END,filename+" loaded\n\n")
    if os.path.exists('model/X.txt.npy'):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (16,16))
                    X.append(img.ravel())
                    label = getLabel(name)
                    Y.append(label)
                    print(name+" "+str(label))
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Animals Found in Dataset : "+str(labels))    


def preprocess():
    text.delete('1.0', END)
    global X, Y, hog_model
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    text.insert(END,"Dataset Image Processing & Normalization Completed\n\n")
    unique, count = np.unique(Y, return_counts=True)
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel('Animal Type')
    plt.ylabel('Count')
    plt.title("Dataset Class Labels Graph")
    plt.show()
    

def splitDataset():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, X, Y, hog_model
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    hog_model = cv2.dnn_DetectionModel(frozen_model, config_file)
    hog_model.setInputSize(320, 320)
    hog_model.setInputScale(1.0/127.5) # 255 / 2 = 127.5
    hog_model.setInputMean((127.5, 127.5, 127.5)) # mobilenet => [-1, 1]
    hog_model.setInputSwapRB(True)
    text.insert(END,"Dataset Train & Test Split Completed\n\n")
    text.insert(END,"Total Images found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Total features found in each Image : "+str(X.shape[1])+"\n\n")
    text.insert(END,"80% dataset records used to train ML algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset records used to train ML algorithms : "+str(X_test.shape[0])+"\n")
    text.update_idletasks()
       

def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")
    text.update_idletasks()

    conf_matrix = confusion_matrix(y_test, predict)
    fig, axs = plt.subplots(1,2,figsize=(10, 4))
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g", ax=axs[0]);
    ax.set_ylim([0,len(labels)])
    axs[0].set_title(algorithm+" Confusion matrix") 

    random_probs = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='orange',label="True classes")
    ns_fpr, ns_tpr, _ = roc_curve(y_test, predict, pos_label=1)
    axs[1].plot(ns_fpr, ns_tpr, linestyle='--', label='Predicted Classes')
    axs[1].set_title(algorithm+" ROC AUC Curve")
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive rate')
    plt.show()    

def runBoostedClassifier():
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    boosted = GradientBoostingClassifier()
    boosted.fit(X_train, y_train)
    predict = boosted.predict(X_test)
    calculateMetrics("Boosted Cascade Classifier", predict, y_test)

def predict():
    global hog_model
    cars = []
    animals = []

    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename)

    height, width, channels = img.shape

    class_ids, confidences, boxes = hog_model.detect(img, confThreshold=0.5)

    if class_ids is None:
        text.insert(END,"No objects detected\n")
        return

    for cls, conf, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
        if cls == 3:  # car class
            cars.append(box)
        if cls == 21 or cls == 18:  # animal classes
            animals.append(box)

    end = 0

    for i in range(len(cars)):
        car_box = cars[i]
        x1, y1, w1, h1 = car_box

        cv2.rectangle(img, car_box, (255,0,0),2)
        cv2.putText(img,"Car"+str(i),(x1+10,y1+40),
                    cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)

        for j in range(len(animals)):
            animal_box = animals[j]
            x2, y2, w2, h2 = animal_box

            if end == 0:
                end = y2 + 40

            dist = round(math.sqrt((x1-width)**2 + (y1-height)**2),2) / 100

            cv2.rectangle(img, animal_box, (0,255,0),2)
            cv2.putText(img,"C="+str(i)+" D="+str(dist),
                        (x2+10,end),
                        cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)

            end += 50

    cv2.imshow("Animal Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def alarm():
    frequency = 2500  
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)

def getPredict():
    global hog_model
    cars = []
    animals = []
    img = cv2.imread("temp.png")
    height, width, channels = img.shape
    class_index, confidence, bbox = hog_model.detect(img, confThreshold=0.5)
    for class_index, conf, boxes in zip(class_index.flatten(), confidence.flatten(), bbox):
        if class_index == 3:
            cars.append(boxes)
        if class_index == 21 or class_index == 18:
            animals.append(boxes)
    end = 0
    for i in range(len(cars)):
        car_box = cars[i]
        x1, y1, x2, y2 = car_box
        cv2.rectangle(img, car_box, (255, 0, 0), 2)
        cv2.putText(img, "Car"+str(i), (car_box[0]+10, car_box[1]+40), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 255, 0), thickness=2)
        for j in range(len(animals)):
            animal_box = animals[j]
            x3, y3, x4, y4 = animal_box
            if end == 0:
                end = animal_box[1]+40
            dist = round(math.sqrt((x1 - width)**2 + (y1 - height)**2), 2) / 100
            if dist < 11:
                alarm()
            cv2.rectangle(img, animal_box, (255, 0, 0), 2)
            cv2.putText(img, "C="+str(i)+" D="+str(dist), (animal_box[0]+10, end), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 255, 0), thickness=2)
            end = end + 50
    return img 


def detectAnimal():
    global hog_model
    filename = filedialog.askopenfilename(initialdir="video")
    video = cv2.VideoCapture(filename)
    while(True):
        ret, frame = video.read()
        if ret == True:
            filename = "temp.png"
            cv2.imwrite("temp.png",frame)
            img = getPredict()
            cv2.imshow("Predicted Result", img)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break  
        else:
            break
    video.release()
    cv2.destroyAllWindows()


def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='A Practical Animal Detection and Collision Avoidance System Using Computer Vision Technique')
title.config(bg='gold2', fg='thistle1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload KTH-Animal Dataset", command=uploadDataset)
uploadButton.place(x=20,y=550)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset", command=preprocess)
processButton.place(x=300,y=550)
processButton.config(font=ff)

splitButton = Button(main, text="Split Dataset Train & Test", command=splitDataset)
splitButton.place(x=520,y=550)
splitButton.config(font=ff)

bcButton = Button(main, text="Run Boosted Classifier Algorithm", command=runBoostedClassifier)
bcButton.place(x=750,y=550)
bcButton.config(font=ff)

predictButton = Button(main, text="Animal Detection", command=predict)
predictButton.place(x=20,y=600)
predictButton.config(font=ff)

closeButton = Button(main, text="Animal Detection from Video", command=detectAnimal)
closeButton.place(x=300,y=600)
closeButton.config(font=ff)

font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)

main.config(bg='DarkSlateGray1')
main.mainloop()
