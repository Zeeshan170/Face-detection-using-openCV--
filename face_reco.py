import cv2

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start the webcam
webcam = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    _, img = webcam.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.5, 4)
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    # Display the frame with detected faces
    cv2.imshow("Face Recognition", img)
    
    # Exit the loop if the "ESC" key is pressed
    key = cv2.waitKey(10)
    if key == 27:  # 27 is the ASCII code for the ESC key
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
