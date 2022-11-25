import os # https://www.tutorialspoint.com/python/os_walk.htm
import re
import cv2

data_folder = "C:/Users/jagac/Downloads/Data"
sub_folder_vid = "hmdb51_org"
sub_folder_jpg = "hmdb51_jpg"
path2aCatgs = os.path.join(data_folder, sub_folder_vid)


cap = cv2.VideoCapture(file)
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
        
        cv2.imwrite('test_'+str(frame_count*frame_skip)+'.jpg', frame)
        i = 0
        continue
    i += 1 
cap.release()
cv2.destroyAllWindows()




            
   
   





