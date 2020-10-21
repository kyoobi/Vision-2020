import numpy as np
import cv2

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

for i in range(30):
  _,background = cap.read()

while(cap.isOpened()):
    
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #making masks
    lower_red = np.array([0,120,70])
    upper_red = np.array([20,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(hsv,lower_red,upper_red)

    mask1 = mask1+mask2
    mask2 = cv2.bitwise_not(mask1)

    #preparing the background
    res1 = cv2.bitwise_and(background,background,mask=mask1)

    #preparing the foreground
    res2 = cv2.bitwise_and(frame,frame,mask=mask2)

    #adding bg and fg
    final = cv2.add(res1,res2)
    
    cv2.imshow("final",final)
    out.write(final)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
