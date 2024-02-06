import numpy as np 
import cv2 

# Capturing video through webcam 
webcam = cv2.VideoCapture(0) 

# Set the coordinates for the region of interest (ROI)
roi_x, roi_y, roi_width, roi_height = 100, 100, 300, 200

while(1): 
    _, imageFrame = webcam.read() 

    # Define the region of interest (ROI)
    roi = imageFrame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]

    hsvFrame = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) 

    # Set range for red color and define mask 
    red_lower = np.array([136, 87, 111], np.uint8) 
    red_upper = np.array([180, 255, 255], np.uint8) 
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 

    # Set range for green color and define mask 
    green_lower = np.array([25, 52, 72], np.uint8) 
    green_upper = np.array([102, 255, 255], np.uint8) 
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper) 

    # Set range for blue color and define mask 
    blue_lower = np.array([94, 80, 2], np.uint8) 
    blue_upper = np.array([120, 255, 255], np.uint8) 
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper) 

    # Set range for yellow color and define mask 
    yellow_lower = np.array([20, 100, 100], np.uint8) 
    yellow_upper = np.array([30, 255, 255], np.uint8) 
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper) 

    # Set range for orange color and define mask 
    orange_lower = np.array([5, 50, 50], np.uint8) 
    orange_upper = np.array([15, 255, 255], np.uint8) 
    orange_mask = cv2.inRange(hsvFrame, orange_lower, orange_upper) 

    # Set range for purple color and define mask 
    purple_lower = np.array([125, 50, 40], np.uint8) 
    purple_upper = np.array([155, 255, 255], np.uint8) 
    purple_mask = cv2.inRange(hsvFrame, purple_lower, purple_upper) 

    # Set range for pink color and define mask 
    pink_lower = np.array([140, 50, 50], np.uint8) 
    pink_upper = np.array([170, 255, 255], np.uint8) 
    pink_mask = cv2.inRange(hsvFrame, pink_lower, pink_upper) 

    # Set range for cyan color and define mask 
    cyan_lower = np.array([85, 50, 50], np.uint8) 
    cyan_upper = np.array([100, 255, 255], np.uint8) 
    cyan_mask = cv2.inRange(hsvFrame, cyan_lower, cyan_upper) 

    # Set range for black color and define mask 
    black_lower = np.array([0, 0, 0], np.uint8)
    black_upper = np.array([180, 255, 30], np.uint8)
    black_mask = cv2.inRange(hsvFrame, black_lower, black_upper)

    # Set range for white color and define mask 
    white_lower = np.array([0, 0, 200], np.uint8)
    white_upper = np.array([180, 30, 255], np.uint8)
    white_mask = cv2.inRange(hsvFrame, white_lower, white_upper)

    # Morphological Transform, Dilation 
    kernel = np.ones((5, 5), "uint8") 

    red_mask = cv2.dilate(red_mask, kernel) 
    res_red = cv2.bitwise_and(roi, roi, mask=red_mask) 

    green_mask = cv2.dilate(green_mask, kernel) 
    res_green = cv2.bitwise_and(roi, roi, mask=green_mask) 

    blue_mask = cv2.dilate(blue_mask, kernel) 
    res_blue = cv2.bitwise_and(roi, roi, mask=blue_mask) 

    yellow_mask = cv2.dilate(yellow_mask, kernel) 
    res_yellow = cv2.bitwise_and(roi, roi, mask=yellow_mask) 

    orange_mask = cv2.dilate(orange_mask, kernel) 
    res_orange = cv2.bitwise_and(roi, roi, mask=orange_mask) 

    purple_mask = cv2.dilate(purple_mask, kernel) 
    res_purple = cv2.bitwise_and(roi, roi, mask=purple_mask) 

    pink_mask = cv2.dilate(pink_mask, kernel) 
    res_pink = cv2.bitwise_and(roi, roi, mask=pink_mask) 

    cyan_mask = cv2.dilate(cyan_mask, kernel) 
    res_cyan = cv2.bitwise_and(roi, roi, mask=cyan_mask) 

    black_mask = cv2.dilate(black_mask, kernel)
    res_black = cv2.bitwise_and(roi, roi, mask=black_mask)

    white_mask = cv2.dilate(white_mask, kernel)
    res_white = cv2.bitwise_and(roi, roi, mask=white_mask)

    # Creating contours for each color 
    colors = [
        (res_red, "Red Colour", (0, 0, 255)),
        (res_green, "Green Colour", (0, 255, 0)),
        (res_blue, "Blue Colour", (255, 0, 0)),
        (res_yellow, "Yellow Colour", (0, 255, 255)),
        (res_orange, "Orange Colour", (0, 165, 255)),
        (res_purple, "Purple Colour", (128, 0, 128)),
        (res_pink, "Pink Colour", (255, 182, 193)),
        (res_cyan, "Cyan Colour", (0, 255, 255)),
        (res_black, "Black Colour", (0, 0, 0)),
        (res_white, "White Colour", (255, 255, 255)),
    ]

    for res, text, color in colors:
        contours, hierarchy = cv2.findContours(
            cv2.inRange(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY), 1, 255),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 300:
                x, y, w, h = cv2.boundingRect(contour)
                # Adjust the coordinates based on the ROI
                x += roi_x
                y += roi_y
                imageFrame = cv2.rectangle(
                    imageFrame, (x, y), (x + w, y + h), color, 2
                )
                cv2.putText(
                    imageFrame,
                    text,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                )

    # Draw a frame around the ROI
    cv2.rectangle(imageFrame, (roi_x, roi_y), (roi_x+roi_width, roi_y+roi_height), (255, 255, 255), 2)

    cv2.imshow("Multiple Color Detection in ROI", imageFrame) 
    if cv2.waitKey(10) & 0xFF == ord('q'): 
        webcam.release() 
        cv2.destroyAllWindows() 
        break
