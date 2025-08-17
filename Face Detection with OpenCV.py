import cv2

# Step 1: Load the image
img = cv2.imread("input.jpg")   # make sure input.jpg exists

# Step 2: Convert to grayscale (face detection works better on gray)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 3: Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Step 4: Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Step 5: Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Step 6: Save and show result
cv2.imwrite("faces_detected.jpg", img)
cv2.imshow("Face Detection", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
