from ultralytics import YOLO
import numpy as np
if __name__ == '__main__':
    model = YOLO("yolov8x-cls.pt")
    data_dir = "C:/Users/snk20/Downloads/Brain Tumor/cleardataset"
    results = model.train(
        data=data_dir,  
        epochs=20,      
        imgsz=256, 
        batch=32,
    )    
    
    print(results)
    



