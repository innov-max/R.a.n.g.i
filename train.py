import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f]

layer_names = net.getUnconnectedOutLayersNames()

# Function to get the color name based on RGB values
def get_color_name(rgb_values):
    # Add your color detection logic here
    # You can use color mapping libraries or create your own logic

    # Example: Just check for common colors like red, green, blue
    if rgb_values[0] > rgb_values[1] and rgb_values[0] > rgb_values[2]:
        return "red"
    elif rgb_values[1] > rgb_values[0] and rgb_values[1] > rgb_values[2]:
        return "green"
    elif rgb_values[2] > rgb_values[0] and rgb_values[2] > rgb_values[1]:
        return "blue"
    else:
        return "unknown"

# Open a connection to the webcam (0 represents the default webcam)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Preprocess frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Process the YOLO output to get object information
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Get the color of the detected object
                roi = frame[y:y + h, x:x + w]
                avg_color = np.mean(roi, axis=(0, 1))
                color_name = get_color_name(avg_color)

                # Draw the bounding box and label
                color = (0, 255, 0)  # Green color for bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{classes[class_id]} ({confidence:.2f}), Color: {color_name}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the result
    cv2.imshow("Object Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
