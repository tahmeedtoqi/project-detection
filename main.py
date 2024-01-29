#copyright claim by Tahmeed Thoky (C) 2024
#contact: tahmeedtoqi123@gmail.com
import cv2
import numpy as np


def detect_color(frame, lower_bound, upper_bound, color_name):
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask using the lower and upper bounds of the color
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around the detected objects
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # adjust this threshold based on your needs
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


def detect_cars(frame, car_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return frame


def main():
    # Open the video file
    cap = cv2.VideoCapture(input("enter the path of the vidoe:"))

    # Load the car cascade classifier
    car_cascade = cv2.CascadeClassifier('cars.xml')

    while True:
        # Read a frame from the video file
        ret, frame = cap.read()

        if not ret:
            # Break the loop if no more frames are available
            break

        # Define the lower and upper bounds of the colors you want to detect (in HSV)
        color_ranges = {
            "Blue": ([110, 50, 50], [130, 255, 255]),
            "Green": ([40, 40, 40], [80, 255, 255]),
            "Red": ([0, 100, 100], [10, 255, 255]),
            "Yellow": ([20, 100, 100], [30, 255, 255]),
            # Add more color ranges here
        }

        # Detect colors and display the result
        for color_name, (lower_bound, upper_bound) in color_ranges.items():
            frame = detect_color(frame, np.array(lower_bound), np.array(upper_bound), color_name)

        # Detect cars and display the result
        frame = detect_cars(frame, car_cascade)

        # Display the resulting frame
        cv2.imshow('Object Detection', frame)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video file and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
