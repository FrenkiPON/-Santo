import torch

model =  YOLOWorld('yolov8s-world') 

def detect_and_manage_tracks(video_source=0):
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Не удалось открыть видеопоток")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр с камеры")
            break
        results = model(frame)

        tracks_status = analyze_detection(results)

        if tracks_status["track_occupied"]:
            print(f"Путь занят {tracks_status['number_of_cars']} вагонами. Переключаем на другой путь.")
            switch_to_free_track()
        else:
            print("Путь свободен.")

        annotated_frame = results[0].plot()
        cv2.imshow('YOLOv8 Real-Time Detection', annotated_frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def analyze_detection(results):
    tracks_status = {
        "track_occupied": False,
        "number_of_cars": 0
    }

    for result in results:
        for detection in result.boxes:
            label = detection.name  
            if label == 'train' or label == 'car':  
                tracks_status["track_occupied"] = True
                tracks_status["number_of_cars"] += 1

    return tracks_status
def switch_to_free_track():
    # Здесь будет логика для переключения на другой путь будем юзать API
    print("Переключение на другой свободный путь...")
    time.sleep(1)  
if __name__ == "__main__":
    detect_and_manage_tracks(0)
    
    
    
    
    
    
