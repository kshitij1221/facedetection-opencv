import cv2
#here we are using pretrained haarcascade classifier  for face detection 
varriable=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
#using webcam 
cap = cv2.VideoCapture(0)
while True:
    #reading frame from web cam 
    ret,frame =cap.read() #ret is (Boolean varriable )  and frames conatin capture image 
    gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  # here we are converting frames into gryscale because it reduces complexity of image  
    faces=varriable.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5,minSize=(30,30))
    for (x,y,w,h) in faces:#x,y are used to provide corrdinate of top left corner w=width ,h=height of detected face 
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)#this line draw green rectangle 
    cv2.imshow('Face Detection',frame)
    if cv2.waitKey(1) & 0xFF == ord('k'):#and here if we press k it will close the window 
        break
cap.release()
cv2.destroyAllWindows()