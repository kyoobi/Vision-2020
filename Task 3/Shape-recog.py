import cv2
import numpy as np

cap=cv2.VideoCapture("http://192.168.0.102:4747/video")

while True:

    #Reading Video Capture
    _,frame= cap.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    #Setting the making limits
    lower_vals = np.array([25,120,70])
    higher_vals = np.array([35,255,255])

    #Preparing the masks
    mask = cv2.inRange(hsv,lower_vals,higher_vals)
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel,iterations=1)
    mask = cv2.dilate(mask,kernel,iterations = 1)

    #Preparing colour filtered image
    fil_image = cv2.bitwise_and(frame,frame,mask=mask)

    #Preparing Mask for shape filtering
    maskc = np.zeros(frame.shape[:2], dtype="uint8")

    #Reading contours from the mask
    contours,hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    
    shape_contours=[]
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.08*cv2.arcLength(cnt,True),True)
        
        #Filtering Shape contours (Triangle)
        if cv2.contourArea(cnt)< 600:
            continue
        if len(approx)==3:
            cv2.drawContours(maskc,[cnt],-1,(255,255,255),-1)
            shape_contours.append(cnt)
    
    #Preparing shape filtered mask
    
    maskc = cv2.morphologyEx(maskc,cv2.MORPH_OPEN,kernel)

    #Preparing shape filtered image
    final_res = cv2.bitwise_and(frame,frame,mask=maskc)
    cv2.drawContours(final_res,shape_contours,-1,(255,0,0),1)

    #Showing results
    cv2.imshow("final",final_res)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    
