import cv2
import sys

image_path='test_images/'
frame_saver=[]

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
	# Capture frame-by-frame
	ret, frame = video_capture.read()
	orig=frame.copy()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
	   # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
	)

	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		
    # Display the resulting frame
	r = 1350.0 / frame.shape[1]
	dim = (1350, int(frame.shape[0] * r))
	frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
	cv2.imshow('Video', frame)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
		
	if cv2.waitKey(1) & 0xFF == ord('s'):
		frame_saver.append(orig)
		

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

count=0
for frame in frame_saver:
	r = 1350.0 / frame.shape[1]
	dim = (1350, int(frame.shape[0] * r))
	frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
	cv2.imwrite(image_path+'test-'+str(count)+'.png',frame)
	count+=1
	cv2.imshow('image',frame)
	cv2.waitKey(0)
	
cv2.destroyAllWindows()	
	