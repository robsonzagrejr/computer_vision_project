import cv2
import interface.devices as dev
import classic.main as cmain
import classic.predictor as cpredictor

def predict_webcam(device, mode='classic'):
    video_capture = cv2.VideoCapture(device)
    predictor = None
    if mode=='classic':
        predictor = cpredictor.ClassicPredictor()
    while True:
        ret, image = video_capture.read()

        if not ret:
            break

        image = cv2.resize(image, (640,480), interpolation = cv2.INTER_AREA)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        car_chunks, _, _ = predictor.predict(img, show_find=False)
        for window in car_chunks:
            start, end = window
            cv2.rectangle(image, (start[1], start[0]), (end[1], end[0]), (255,0,0), 2)
        cv2.imshow('Frame',image)
        #Waitig for q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    video_capture.release()
    cv2.destroyAllWindows()

def list():
    dev.list_ports()