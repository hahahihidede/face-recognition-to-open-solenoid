#Face recognition to control relay
#   pastikan semua library sudah ter install ya bebeb
#   dan pin sudah terpasang dengan benar
#pin
#VCC relay to GPIO 17
#GND relay to GPIO 20
#S relay to anywhere except VCC and GND

#import library dan file yang dibutuhkan
import dlib         
import numpy as np  
import cv2         
import pandas as pd 
import os
import time
import RPi.GPIO as GPIO

#Define  GPIO pin untuk Relay
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)
GPIO.output(17, GPIO.LOW)

#Atur delay untuk relay
time.sleep(0.25)

#Gunakan Dlib sebagai model untuk mengenali wajah
#Turorial lengkap kunjungi http://dlib.net/python/index.html#dlib.face_recognition_model_v1
faceRecognizer = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


def returnEuclideanDistance(feature1, feature2):
    feature1 = np.array(feature1)
    feature2 = np.array(feature2)
    dist = np.sqrt(np.sum(np.square(feature1 - feature2)))
    return dist


#lokasi dataset csv
if os.path.exists("data/features_all.csv"):
    csvPath = "data/features_all.csv"
    readCSV = pd.read_csv(csvPath, header=None)


    
    featuresArr = []

    
    for i in range(readCSV.shape[0]):
        someoneFeature = [] #i know it's like adele's song wqwq
        for j in range(0, len(readCSV.ix[i, :])):
            someoneFeature.append(readCSV.ix[i, :][j])
        featuresArr.append(someoneFeature)
    print("Faces in Database：", len(featuresArr))

#dataset Dlib
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

    
    cap = cv2.VideoCapture(0)

    
    while cap.isOpened():

        flag, frame = cap.read()
        img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = detector(img_gray, 0)

        
        font = cv2.FONT_ITALIC

        
        nameListPosition = []
        nameList = []

        kk = cv2.waitKey(1)

        
        if kk == ord('q'):
            break
        else:

            if len(faces) != 0:
                
                featureCapArray = []
                for i in range(len(faces)):
                    shape = predictor(frame, faces[i])
                    featureCapArray.append(faceRecognizer.compute_face_descriptor(frame, shape))

               
                for k in range(len(faces)):
                    print("##### camera person", k+1, "#####")
                   
                    nameList.append("unknown")

                     
                    nameListPosition.append(tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top())/4)]))

                    
                    estDistanceList = []
                    for i in range(len(featuresArr)):
                        
                        if str(featuresArr[i][0]) != '0.0':
                            print("with person", str(i + 1), "the e distance: ", end='')
                            e_distance_tmp = returnEuclideanDistance(featureCapArray[k], featuresArr[i])
                            print(e_distance_tmp)
                            estDistanceList.append(e_distance_tmp)
                        else:
                            
                            estDistanceList.append(999999999)
                    
                    similarPersonList = estDistanceList.index(min(estDistanceList))
                    print("Minimum e distance with person", int(similarPersonList)+1)

                    if min(estDistanceList) < 0.4:
                        
                        nameList[k] = "Person "+str(int(similarPersonList)+1)
                        print("May be person "+str(int(similarPersonList)+1))
                        
                        GPIO.output(17, GPIO.LOW)
                    else:
                        print("Unknown person")
                        GPIO.output(17, GPIO.HIGH)

                    for kk, d in enumerate(faces):
                        
                        cv2.rectangle(frame, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255), 2)
                    print('\n')

                
                for i in range(len(faces)):
                    cv2.putText(frame, nameList[i], nameListPosition[i], font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

        print("Face In Cam:", nameList, "\n")

        # cv2.putText(frame, "Press 'q': Quit", (20, 450), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)
        cv2.putText(frame, "Face Recognition", (20, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "Faces: " + str(len(faces)), (20, 100), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow("cam", frame)

    cap.release()
    cv2.destroyAllWindows()

else:
    print('##### Warning #####')
