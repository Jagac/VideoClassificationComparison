import os # https://www.tutorialspoint.com/python/os_walk.htm
import re
import cv2

data_folder = r"C:\Users\jagac\Downloads\Data"
sub_folder_vid = "hmdb51_org"
sub_folder_jpg = "hmdb51_jpg"
path2videos = os.path.join(data_folder, sub_folder_vid)

for path, subdirs, files in os.walk(path2videos):
    for name in files:
        vids = os.path.join(path, name)

        cap = cv2.VideoCapture(vids)
        i = 0
        # only takes a certain number of frames to save on complexity
        frame_skip = 10
        frame_count = 0

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            
            if i > frame_skip - 1:
                frame_count += 1
                
                path2store = r"C:\Users\jagac\OneDrive\Documents\CSC578\final\HumanActionClassifier\images"
                storage = os.path.join(path2store, name)
                os.makedirs(storage, exist_ok= True)
                path2img = os.path.join(storage, 'test_' + str(frame_count*frame_skip) + ".jpg")
                cv2.imwrite(path2img, frame)
                
                i = 0
                continue
            i += 1 
        cap.release()
        cv2.destroyAllWindows()    
      
                    





            
   
   





