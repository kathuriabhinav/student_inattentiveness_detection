# student_inattentiveness_detection


This system is made for monitoring of students in online classes


In this system, it verifies if user is attentive or not based on the following features :
- if there is a face or not 
- if same person is sitting or not
- Head alignment
- lips movement
- expression change
- if spoofing is done or not
- eye lid movement
- yawn detection

This system is able to achieve all the above using following induvidual modules :
- face detection
- face recognition
- head pose detection
- estimation of mouth aspect ratio
- estimation of expression change
- spoof detection
- estimation of eye aspect ratio


All the output data of induvidual modules are fetched frame by frame

These data is estimated using segment wise analysis using sliding window method

The final segment wise analysis is further processed to find the inatentiveness detection 
