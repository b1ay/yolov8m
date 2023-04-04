from ultralytics import YOLO
model = YOLO("yolov8m_71ep_helm.pt") 
results = model.predict("/home/user-7/yolov8/test_images", save=True, save_txt=True)
