# Object Detection with Hugging Face Transformers

## Overview
This Python script provides object detection functionality using Hugging Face transformers, specifically designed to work with image files (PNG, JPG, etc.) and return detected objects with bounding boxes, labels, and confidence scores.

## Features
- **Multiple Detection Methods**: Simple pipeline and advanced approaches
- **Visual Output**: Automatic generation of images with bounding boxes and labels
- **Flexible Input**: Supports local files and URLs
- **Multiple Models**: Easy switching between different DETR models
- **GPU Support**: Automatic GPU acceleration when available
- **Comprehensive Results**: Detailed detection summaries and statistics

## Installation

1. Install required dependencies:
```bash
pip install -r requirements_object_detection.txt
```

2. For GPU acceleration (optional):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Basic Usage
```python
from object_detection import detect_objects_pipeline

# Detect objects in an image
detections = detect_objects_pipeline("my_image.jpg")

# View results
for detection in detections:
    print(f"Found {detection['label']} with {detection['score']:.2f} confidence")
```

### Advanced Usage with Visualization
```python
from object_detection import detect_objects_advanced, visualize_detections

# Advanced detection
detections, image = detect_objects_advanced("my_image.jpg")

# Create visualization
visualize_detections(image, detections, "result.jpg")
```

### Command Line Usage
```bash
# Test with your own image
python test_detection.py my_image.jpg

# Interactive mode
python test_detection.py
```

## Available Functions

### `detect_objects_pipeline()`
Simple object detection using Hugging Face pipeline.
- **Input**: Image path or URL
- **Output**: List of detected objects
- **Best for**: Quick detection tasks

### `detect_objects_advanced()`
Advanced detection with more control and better performance.
- **Input**: Image path or URL
- **Output**: Detected objects + original image
- **Best for**: Production use, custom processing

### `visualize_detections()`
Creates visualization with bounding boxes and labels.
- **Input**: Image and detection results
- **Output**: Annotated image file
- **Best for**: Visual analysis, presentations

### `print_detection_summary()`
Prints formatted summary of detection results.
- **Input**: Detection results
- **Output**: Console summary with statistics

## Supported Models

### DETR (Detection Transformer) Models
- `facebook/detr-resnet-50` - Standard model, good balance of speed/accuracy
- `facebook/detr-resnet-101` - Higher accuracy, slower inference
- `microsoft/conditional-detr-resnet-50` - Conditional DETR variant

### Model Selection
```python
# Standard model (recommended)
detections = detect_objects_pipeline("image.jpg", model_name="facebook/detr-resnet-50")

# High accuracy model
detections = detect_objects_pipeline("image.jpg", model_name="facebook/detr-resnet-101")
```

## Configuration Options

### Confidence Threshold
```python
# Only show high-confidence detections
detections = detect_objects_pipeline("image.jpg", confidence_threshold=0.8)

# Show more detections (including uncertain ones)
detections = detect_objects_pipeline("image.jpg", confidence_threshold=0.3)
```

### Device Selection
```python
# Force CPU usage
detections, image = detect_objects_advanced("image.jpg", device="cpu")

# Force GPU usage (if available)
detections, image = detect_objects_advanced("image.jpg", device="cuda")
```

## Output Format

### Detection Object Structure
```python
{
    "label": "person",           # Object class name
    "score": 0.95,              # Confidence score (0.0 to 1.0)
    "box": {                    # Bounding box coordinates
        "xmin": 100,            # Left edge
        "ymin": 150,            # Top edge  
        "xmax": 300,            # Right edge
        "ymax": 450             # Bottom edge
    }
}
```

### Supported Object Classes
The DETR model can detect 80+ object classes including:
- **People**: person
- **Vehicles**: car, truck, bus, motorcycle, bicycle
- **Animals**: dog, cat, horse, cow, sheep, bird
- **Objects**: chair, table, laptop, phone, book
- **Food**: apple, banana, pizza, sandwich
- And many more...

## Example Results

### Sample Detection Output
```
==============================================================
OBJECT DETECTION RESULTS
==============================================================
Total objects detected: 4

Detected objects:
--------------------------------------------------------------
 1. person          | Confidence: 0.998 | Box: (123, 45) to (456, 678)
 2. car             | Confidence: 0.892 | Box: (567, 234) to (890, 456)
 3. dog             | Confidence: 0.756 | Box: (234, 567) to (345, 789)
 4. bicycle         | Confidence: 0.654 | Box: (678, 123) to (789, 345)

Object counts by category:
------------------------------
bicycle        : 1
car            : 1
dog            : 1
person         : 1
==============================================================
```

## Performance Tips

### For Better Speed
- Use `facebook/detr-resnet-50` model
- Enable GPU acceleration
- Resize large images before detection
- Use higher confidence thresholds

### For Better Accuracy
- Use `facebook/detr-resnet-101` model
- Lower confidence threshold (0.3-0.5)
- Ensure good image quality
- Use original image resolution

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce image size
   - Use CPU instead of GPU
   - Close other applications

2. **No Objects Detected**
   - Lower confidence threshold
   - Check image quality
   - Try different model

3. **Slow Performance**
   - Enable GPU acceleration
   - Use smaller model
   - Resize input images

### Error Messages
- `"Model not found"` → Check internet connection, model will download automatically
- `"Image not found"` → Verify file path and format
- `"CUDA out of memory"` → Switch to CPU or reduce image size

## Example Scripts

### Batch Processing
```python
import os
from object_detection import detect_objects_advanced, visualize_detections

# Process all images in a folder
image_folder = "images/"
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_folder, filename)
        detections, image = detect_objects_advanced(image_path)
        output_path = f"detected_{filename}"
        visualize_detections(image, detections, output_path)
```

### Custom Analysis
```python
from object_detection import detect_objects_advanced

# Analyze specific object types
detections, image = detect_objects_advanced("street_scene.jpg")

# Count vehicles
vehicles = ['car', 'truck', 'bus', 'motorcycle']
vehicle_count = sum(1 for d in detections if d['label'] in vehicles)
print(f"Found {vehicle_count} vehicles in the image")

# Find largest object
if detections:
    largest = max(detections, key=lambda d: 
        (d['box']['xmax'] - d['box']['xmin']) * 
        (d['box']['ymax'] - d['box']['ymin'])
    )
    print(f"Largest object: {largest['label']}")
```

## Next Steps
- Try different models for comparison
- Implement real-time video detection
- Add custom object classes through fine-tuning
- Integrate with web applications or APIs
- Explore other computer vision tasks (segmentation, pose estimation)
