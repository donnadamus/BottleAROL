from ultralytics import YOLO



def train(epochs, imgsz, device, datapath):
    # Load a model
    model = YOLO("../models/yolo11n.pt")
    
    # Train the model
    train_results = model.train(
        data=datapath,  # path to dataset YAML
        epochs=epochs,  # number of training epochs
        imgsz=imgsz,  # training image size
        device=device,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    )
    
    # Evaluate model performance on the validation set
    metrics = model.val()
    return metrics