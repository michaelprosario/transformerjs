"""
Object Detection using Hugging Face Transformers
Detects objects in images (PNG, JPG, etc.) and returns bounding boxes with labels and confidence scores
"""

import torch
from transformers import pipeline, DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
import requests
from io import BytesIO

def detect_objects_pipeline(
    image_path: str,
    model_name: str = "facebook/detr-resnet-50",
    confidence_threshold: float = 0.5
) -> List[Dict]:
    """
    Detect objects in an image using Hugging Face pipeline (simple approach).
    
    Args:
        image_path (str): Path to the image file or URL
        model_name (str): Hugging Face model identifier for object detection
        confidence_threshold (float): Minimum confidence score for detections
    
    Returns:
        List[Dict]: List of detected objects with labels, scores, and bounding boxes
    
    Example:
        >>> detections = detect_objects_pipeline("image.jpg")
        >>> print(detections[0])
        {'label': 'person', 'score': 0.99, 'box': {'xmin': 100, 'ymin': 200, 'xmax': 300, 'ymax': 400}}
    """
    try:
        print(f"Loading object detection pipeline with model: {model_name}")
        
        # Create object detection pipeline
        detector = pipeline("object-detection", model=model_name)
        
        # Load image
        if image_path.startswith(('http://', 'https://')):
            print(f"Loading image from URL: {image_path}")
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content))
        else:
            print(f"Loading image from file: {image_path}")
            image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        print("Running object detection...")
        # Perform object detection
        detections = detector(image)
        
        # Filter by confidence threshold
        filtered_detections = [
            detection for detection in detections 
            if detection['score'] >= confidence_threshold
        ]
        
        print(f"Found {len(filtered_detections)} objects above confidence threshold {confidence_threshold}")
        
        return filtered_detections
        
    except Exception as e:
        print(f"Error in object detection: {str(e)}")
        raise


def detect_objects_advanced(
    image_path: str,
    model_name: str = "facebook/detr-resnet-50",
    confidence_threshold: float = 0.5,
    device: Optional[str] = None
) -> Tuple[List[Dict], Image.Image]:
    """
    Advanced object detection with more control over the process.
    
    Args:
        image_path (str): Path to the image file or URL
        model_name (str): Hugging Face model identifier
        confidence_threshold (float): Minimum confidence score for detections
        device (str): Device to run inference on ('cpu', 'cuda', or None for auto)
    
    Returns:
        Tuple[List[Dict], Image.Image]: Detected objects and the original image
    """
    try:
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        print(f"Loading model and processor: {model_name}")
        # Load model and processor
        processor = DetrImageProcessor.from_pretrained(
            model_name,
            do_resize=True,
            size=800,  # Default DETR size
            max_size=1333,  # Default DETR max size
            do_pad=True,    # Enable padding
            pad_size_divisor=32  # DETR's default padding divisor
        )
        model = DetrForObjectDetection.from_pretrained(model_name)
        model.to(device)
        
        # Load image
        if image_path.startswith(('http://', 'https://')):
            print(f"Loading image from URL: {image_path}")
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content))
        else:
            print(f"Loading image from file: {image_path}")
            image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        print("Processing image...")
        # Process image
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        print("Running inference...")
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process outputs
        target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
        results = processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes, 
            threshold=confidence_threshold
        )[0]
        
        # Format results
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections.append({
                "label": model.config.id2label[label.item()],
                "score": score.item(),
                "box": {
                    "xmin": int(box[0].item()),
                    "ymin": int(box[1].item()),
                    "xmax": int(box[2].item()),
                    "ymax": int(box[3].item())
                }
            })
        
        print(f"Found {len(detections)} objects above confidence threshold {confidence_threshold}")
        
        return detections, image
        
    except Exception as e:
        print(f"Error in advanced object detection: {str(e)}")
        raise


def visualize_detections(
    image: Image.Image,
    detections: List[Dict],
    output_path: str = "detected_objects.jpg",
    font_size: int = 20
) -> str:
    """
    Visualize object detections on the image with bounding boxes and labels.
    
    Args:
        image (Image.Image): PIL Image object
        detections (List[Dict]): List of detected objects
        output_path (str): Path to save the visualization
        font_size (int): Font size for labels
    
    Returns:
        str: Path to the saved visualization
    """
    print(f"Creating visualization with {len(detections)} detections...")
    
    # Create a copy of the image for drawing
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()
    
    # Color palette for different objects
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57",
        "#FF9FF3", "#54A0FF", "#5F27CD", "#00D2D3", "#FF9F43",
        "#FFB142", "#FF6348", "#1DD1A1", "#FECA57", "#48DBFB"
    ]
    
    for i, detection in enumerate(detections):
        box = detection["box"]
        label = detection["label"]
        score = detection["score"]
        
        # Get color for this detection
        color = colors[i % len(colors)]
        
        # Draw bounding box
        draw.rectangle(
            [(box["xmin"], box["ymin"]), (box["xmax"], box["ymax"])],
            outline=color,
            width=3
        )
        
        # Prepare label text
        label_text = f"{label}: {score:.2f}"
        
        # Get text bounding box for background
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Draw label background
        draw.rectangle(
            [(box["xmin"], box["ymin"] - text_height - 5), 
             (box["xmin"] + text_width + 10, box["ymin"])],
            fill=color
        )
        
        # Draw label text
        draw.text(
            (box["xmin"] + 5, box["ymin"] - text_height - 2),
            label_text,
            fill="white",
            font=font
        )
    
    # Save the visualization
    draw_image.save(output_path)
    print(f"Visualization saved to: {output_path}")
    
    return output_path


def print_detection_summary(detections: List[Dict]) -> None:
    """
    Print a formatted summary of the detections.
    
    Args:
        detections (List[Dict]): List of detected objects
    """
    print("\n" + "="*60)
    print("OBJECT DETECTION RESULTS")
    print("="*60)
    
    if not detections:
        print("No objects detected.")
        return
    
    print(f"Total objects detected: {len(detections)}")
    print("\nDetected objects:")
    print("-" * 60)
    
    for i, detection in enumerate(detections, 1):
        label = detection["label"]
        score = detection["score"]
        box = detection["box"]
        
        print(f"{i:2d}. {label:<15} | Confidence: {score:.3f} | "
              f"Box: ({box['xmin']}, {box['ymin']}) to ({box['xmax']}, {box['ymax']})")
    
    # Count objects by category
    object_counts = {}
    for detection in detections:
        label = detection["label"]
        object_counts[label] = object_counts.get(label, 0) + 1
    
    print("\nObject counts by category:")
    print("-" * 30)
    for label, count in sorted(object_counts.items()):
        print(f"{label:<15}: {count}")
    
    print("="*60)


# Sample program demonstrating the object detection functions
def main():
    """
    Sample program demonstrating object detection functionality.
    """
    print("🔍 Object Detection with Hugging Face Transformers")
    print("="*60)
    
    # Test images (you can replace these with your own images)
    test_images = [
        # Local image path (if you have one)
        # "test_image.jpg",
        
        # Online test images
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png",
        "https://images.unsplash.com/photo-1551963831-b3b1ca40c98e?w=800&h=600&fit=crop",  # Breakfast
    ]
    
    # Available models (you can try different ones)
    models = [
        "facebook/detr-resnet-50",           # Standard DETR
        # "facebook/detr-resnet-101",        # Larger DETR (more accurate, slower)
        # "microsoft/conditional-detr-resnet-50",  # Conditional DETR
    ]
    
    model_name = models[0]  # Use the first model
    confidence_threshold = 0.7
    
    for i, image_path in enumerate(test_images, 1):
        print(f"\n🖼️  Processing Image {i}: {image_path}")
        print("-" * 50)
        
        try:
            # Method 1: Simple pipeline approach
            print("Method 1: Using pipeline approach...")
            detections_simple = detect_objects_pipeline(
                image_path=image_path,
                model_name=model_name,
                confidence_threshold=confidence_threshold
            )
            
            print_detection_summary(detections_simple)
            
            # Method 2: Advanced approach with visualization
            print("\nMethod 2: Using advanced approach with visualization...")
            detections_advanced, image = detect_objects_advanced(
                image_path=image_path,
                model_name=model_name,
                confidence_threshold=confidence_threshold
            )
            
            # Create visualization
            output_filename = f"detected_objects_image_{i}.jpg"
            visualize_detections(
                image=image,
                detections=detections_advanced,
                output_path=output_filename
            )
            
            print(f"\n✅ Results saved to: {output_filename}")
            
        except Exception as e:
            print(f"❌ Error processing image {i}: {str(e)}")
            continue
    
    print("\n🎉 Object detection demo completed!")
    print("\nTips:")
    print("- Try different confidence thresholds (0.1 to 0.9)")
    print("- Experiment with different models for better accuracy")
    print("- Use GPU for faster inference if available")
    print("- Check the generated visualization images!")


if __name__ == "__main__":
    main()
