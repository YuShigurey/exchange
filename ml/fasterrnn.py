import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw

# Load a pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Function to preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image)
    return image_tensor.unsqueeze(0), image

# Function to perform object detection
def detect_objects(image_tensor, model, threshold=0.5):
    with torch.no_grad():
        predictions = model(image_tensor)[0]
    boxes = predictions['boxes']
    labels = predictions['labels']
    scores = predictions['scores']
    
    # Filter out low-confidence detections
    high_conf_indices = [i for i, score in enumerate(scores) if score > threshold]
    boxes = boxes[high_conf_indices]
    labels = labels[high_conf_indices]
    scores = scores[high_conf_indices]
    
    return boxes, labels, scores

# Function to visualize the detection results
def visualize_detections(image, boxes, labels, scores):
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(boxes, labels, scores):
        draw.rectangle(box.tolist(), outline="red", width=3)
        draw.text((box[0], box[1]), f"{label}: {score:.2f}", fill="red")
    image.show()

# Load an image and perform detection
image_path = "path/to/your/image.jpg"
image_tensor, image = preprocess_image(image_path)
boxes, labels, scores = detect_objects(image_tensor, model)

# Visualize the results
visualize_detections(image, boxes, labels, scores)