from ultralytics import YOLO

# Load a model
model = YOLO('best.pt')

# Run batched inference on a list of images
results = model(['image_0041.jpg','image_0065.jpg','image_0129.jpg','image_0133.jpg'])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    result.save(filename='result.jpg')  # save to disk