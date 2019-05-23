# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 22:22:19 2018

@author: Reyan Baig
"""

import dlib # dlib for accurate face detection
import cv2 # opencv
import imutils # helper functions from pyimagesearch.com
from imutils import face_utils

# Grab video from your webcam
stream = cv2.VideoCapture(0)

id=0

# Face detector
detector = dlib.get_frontal_face_detector()

rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer\\trainingData.xml")

font = cv2.FONT_HERSHEY_SIMPLEX

# Fancy box drawing function by Dan Masek
def draw_border(img, pt1, pt2, color, thickness, r, d):
	x1, y1 = pt1
	x2, y2 = pt2

	# Top left drawing
	cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
	cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
	cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

	# Top right drawing
	cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
	cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
	cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

	# Bottom left drawing
	cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
	cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
	cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

	# Bottom right drawing
	cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
	cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
	cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
	# read frames from live web cam stream
	(grabbed, frame) = stream.read()
	
	# Flip the capturing video
	frame=cv2.flip(frame,1)
	
	# resize the frames to be smaller and switch to gray scale
	frame = imutils.resize(frame, width=700)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Make copies of the frame for transparency processing
	overlay = frame.copy()
	output = frame.copy()

	# set transparency value
	alpha  = 0.5

	# detect faces in the gray scale frame
	face_rects = detector(gray, 0)
	
	a=len(face_rects)
	print("Detected Number Of Faces : ",a,"\n")
	
	# loop over the face detections
	for i, d in enumerate(face_rects):
		x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
		(x, y, w, h) = face_utils.rect_to_bb(d)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
		
		id,conf=rec.predict(gray[y:y+h,x:x+w])
		if(id==1):
			name="Reyan"
			print(name)
			
		elif(id==2):
			name="Ahmed"
			print(name)
			
		else:
			id="Unkonwn"
				
		cv2.putText(frame,str(name),(x,h),font,1.5,(150,250,0),lineType=cv2.LINE_AA)
		   
	# show the frame
	cv2.imshow("Face Detection", frame)
	key = cv2.waitKey(1) & 0xFF

	# press q to break out of the loop
	if key == ord("q"):
		break
 
# cleanup
stream.release()
cv2.destroyAllWindows()