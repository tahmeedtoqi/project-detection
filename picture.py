#copyright claim by Tahmeed Thoky (C) 2024
#contact: tahmeedtoqi123@gmail.com
import cv2
import numpy as np
import pytesseract

# Set the path to the YOLO configuration, weights, and coco names files
yolo_config = 'yolov3.cfg'
yolo_weights = 'yolov3.weights'
coco_names = 'coco.names'

# Load YOLO model and classes
net = cv2.dnn.readNet(yolo_weights, yolo_config)
with open(coco_names, 'r') as f:
    classes = [line.strip() for line in f]

# Function to perform object detection
def detect_objects(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Prepare the image for YOLO model
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get the output layer names
    layer_names = net.getUnconnectedOutLayersNames()

    # Run forward pass to get detections
    detections = net.forward(layer_names)

    objects_info = []

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                label = f"{classes[class_id]}: {confidence:.2f}"
                objects_info.append({'label': label, 'coordinates': (x, y, x+w, y+h)})

    return objects_info

# Function to perform OCR using pytesseract
def perform_ocr(image, coordinates):
    x, y, x_end, y_end = coordinates
    roi = image[y:y_end, x:x_end]
    text = pytesseract.image_to_string(roi)
    return text

# Example usage
image_path = 'test.jpg'
objects_info = detect_objects(image_path)

# Display detected objects
for obj_info in objects_info:
    label = obj_info['label']
    coordinates = obj_info['coordinates']
    print(f"Detected: {label}")

    # Perform OCR on the region of interest
    ocr_result = perform_ocr(cv2.imread(image_path), coordinates)
    print(f"OCR Result: {ocr_result}")

# Display the image with bounding boxes
image_with_boxes = cv2.imread(image_path)
for obj_info in objects_info:
    x, y, x_end, y_end = obj_info['coordinates']
    cv2.rectangle(image_with_boxes, (x, y), (x_end, y_end), (0, 255, 0), 2)

cv2.imshow('Object Detection', image_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()
