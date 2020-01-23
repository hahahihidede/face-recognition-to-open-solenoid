#Face recognition to control relay
#   pastikan semua library sudah ter install
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

#Define pin untuk Relay
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)
GPIO.output(17, GPIO.LOW)

#Atur delay untuk relay
time.sleep(0.25)

#Gunakan Dlib sebagai model untuk mengenali wajah
#Turorial lengkap kunjungi http://dlib.net/python/index.html#dlib.face_recognition_model_v1
facerec = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    return dist


#lokasi dataset csv
if os.path.exists("data/features_all.csv"):
    path_features_known_csv = "data/features_all.csv"
    csv_rd = pd.read_csv(path_features_known_csv, header=None)


    
    features_known_arr = []

    
    for i in range(csv_rd.shape[0]):
        features_someone_arr = []
        for j in range(0, len(csv_rd.ix[i, :])):
            features_someone_arr.append(csv_rd.ix[i, :][j])
        features_known_arr.append(features_someone_arr)
    print("Faces in Database：", len(features_known_arr))

#dataset Dlib
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

    
    cap = cv2.VideoCapture(0)

    
    while cap.isOpened():

        flag, img_rd = cap.read()
        img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
        faces = detector(img_gray, 0)

        
        font = cv2.FONT_ITALIC

        
        pos_namelist = []
        name_namelist = []

        kk = cv2.waitKey(1)

        
        if kk == ord('q'):
            break
        else:

            if len(faces) != 0:
                
                features_cap_arr = []
                for i in range(len(faces)):
                    shape = predictor(img_rd, faces[i])
                    features_cap_arr.append(facerec.compute_face_descriptor(img_rd, shape))

               
                for k in range(len(faces)):
                    print("##### camera person", k+1, "#####")
                   
                    name_namelist.append("unknown")

                     
                    pos_namelist.append(tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top())/4)]))

                    
                    e_distance_list = []
                    for i in range(len(features_known_arr)):
                        
                        if str(features_known_arr[i][0]) != '0.0':
                            print("with person", str(i + 1), "the e distance: ", end='')
                            e_distance_tmp = return_euclidean_distance(features_cap_arr[k], features_known_arr[i])
                            print(e_distance_tmp)
                            e_distance_list.append(e_distance_tmp)
                        else:
                            
                            e_distance_list.append(999999999)
                    
                    similar_person_num = e_distance_list.index(min(e_distance_list))
                    print("Minimum e distance with person", int(similar_person_num)+1)

                    if min(e_distance_list) < 0.4:
                        
                        name_namelist[k] = "Person "+str(int(similar_person_num)+1)
                        print("May be person "+str(int(similar_person_num)+1))
                        
                        GPIO.output(17, GPIO.LOW)
                    else:
                        print("Unknown person")
                        GPIO.output(17, GPIO.HIGH)

                    for kk, d in enumerate(faces):
                        
                        cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255), 2)
                    print('\n')

                
                for i in range(len(faces)):
                    cv2.putText(img_rd, name_namelist[i], pos_namelist[i], font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

        print("Faces in camera now:", name_namelist, "\n")

        cv2.putText(img_rd, "Press 'q': Quit", (20, 450), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Face Recognition", (20, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 100), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow("camera", img_rd)

    cap.release()
    cv2.destroyAllWindows()

else:
    print('##### Warning #####', '\n')
    print("'features_all.py' not found!")
    print("Please run 'get_faces_from_camera.py' and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'", '\n')
    print('##### Warning #####')
