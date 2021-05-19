
import numpy as np
import cv2

from yolo.config import (
    coco_file,
    yolo_cfg_file,
    yolo_weights_file,
    img_size,
    conf_threshold,
    nms_threshold
)

class Yolo():

    def __init__(self):
        self.yolo = cv2.dnn.readNetFromDarknet(yolo_cfg_file, yolo_weights_file)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.class_names = open(coco_file, 'r').read().rstrip('\n').split('\n')
        self.class_colors = []
        for c in self.class_names:
            self.class_colors.append((np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)))       


    def predict(self, image, show_find=False):
        blob = cv2.dnn.blobFromImage(image, 1/255, img_size, [0,0,0],crop=False)

        self.yolo.setInput(blob)

        layer_names = self.yolo.getLayerNames()

        #Layers names start in 1 (wat a curse)
        ouput_names = [layer_names[i[0]-1] for i in self.yolo.getUnconnectedOutLayers()]

        #Output for each layer
        outputs = self.yolo.forward(ouput_names)
        bbox, classes_id, confidences, image = self.find_objects(outputs, image, show_find)
        
        return bbox, classes_id, confidences, image


    def find_objects(self, outputs, image, show_find=False):
        img_h, img_w, img_c = image.shape
        bbox = []
        classes_id = []
        confidences = []

        for output in outputs:
            for det in output:
                # cx, cy, w, h, confidence, classes...
                scores = det[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    b_w, b_h = int(det[2]*img_w), int(det[3]*img_h)
                    b_x, b_y = int((det[0]*img_w) - b_w/2), int((det[1]*img_h) - b_h/2)

                    bbox.append([b_x,b_y,b_w,b_h])
                    classes_id.append(class_id)
                    confidences.append(float(confidence))
        indices = cv2.dnn.NMSBoxes(bbox, confidences, conf_threshold, nms_threshold)
        bbox = [bbox[i[0]] for i in indices]
        classes_id = [classes_id[i[0]] for i in indices]
        confidences = [confidences[i[0]] for i in indices]

        aux = 0
        if show_find:
            for box in bbox:
                x, y, w, h = box
                cv2.rectangle(image, (x,y), (x+w, y+h), self.class_colors[aux], 2)
                cv2.putText(image, f"{self.class_names[aux]} {round(confidences[aux],2)}%", 
                    (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.class_colors[aux], 2
                )
                aux += 1


        return bbox, classes_id, confidences, image