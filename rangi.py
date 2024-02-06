import cv2
import numpy as np

# Load the pre-trained MobileNet SSD model
net = cv2.dnn.readNetFromCaffe('/home/max/Desktop/Rangi/MobileNet-SSD-master/deploy.prototxt', '/home/max/Desktop/Rangi/MobileNet-SSD-master/mobilenet_iter_73000.caffemodel')

color_labels = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
                "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
                "train", "tvmonitor", "black", "blue", "brown", "gray", "green", "orange", "pink", "purple",
                "red", "white", "yellow"]

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Resize the frame to match the input size expected by the model
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # Set the input to the model
    net.setInput(blob)

    # Perform inference
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.2:
            color_index = int(detections[0, 0, i, 1])

            # Get the predicted color label
            predicted_color = color_labels[color_index]

            # Display the predicted color label
            cv2.putText(frame, f'Predicted Color: {predicted_color}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Real-Time Color Identification', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
