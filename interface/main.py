import cv2
import interface.devices as dev
import classic.main as cmain
import classic.predictor as cpredictor
import yolo.yolo as yolo

def predict_webcam(device, classic_mode=False):
    video_capture = cv2.VideoCapture(device)
    predictor = None
    if classic_mode:
        predictor = cpredictor.ClassicPredictor()
    else:
        predictor = yolo.Yolo()
    while True:
        ret, image = video_capture.read()

        if not ret:
            break

        image = cv2.resize(image, (640,480), interpolation = cv2.INTER_AREA)
        if classic_mode:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            car_chunks, _, img = predictor.predict(img, show_find=True)
            for window in car_chunks:
                start, end = window
                cv2.rectangle(image, (start[1], start[0]), (end[1], end[0]), (255,0,0), 2)
        else:
            _, _, _, image = predictor.predict(image, show_find=True)
        
        cv2.imshow('Frame',image)
        #Waitig for q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    video_capture.release()
    cv2.destroyAllWindows()

def list():
    dev.list_ports()