#!/usr/bin/env python3
"""
Simple Object Detection Test Script
Quick test for object detection functionality with your own images
"""

from object_detection import detect_objects_pipeline, detect_objects_advanced, visualize_detections, print_detection_summary
import sys
import os

def test_single_image(image_path: str):
    """Test object detection on a single image."""
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' not found!")
        return
    
    print(f"🔍 Analyzing image: {image_path}")
    print("-" * 50)
    
    try:
        # Run object detection
        detections, image = detect_objects_advanced(
            image_path=image_path,
            confidence_threshold=0.5
        )
        
        # Print results
        print_detection_summary(detections)
        
        # Create visualization
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{base_name}_detected.jpg"
        
        visualize_detections(
            image=image,
            detections=detections,
            output_path=output_path
        )
        
        print(f"\n✅ Analysis complete! Check '{output_path}' for visualization.")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def main():
    """Main function for command line usage."""
    
    print("🔍 Object Detection Test Script")
    print("="*40)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        # Use image path from command line
        image_path = sys.argv[1]
        test_single_image(image_path)
    else:
        # Interactive mode
        print("Enter the path to your image file:")
        print("(Supported formats: JPG, PNG, BMP, TIFF)")
        print("(Or press Enter to use demo URL)")
        
        image_path = input("\nImage path: ").strip()
        
        if not image_path:
            # Use demo image
            image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
            print(f"Using demo image: {image_path}")
        
        test_single_image(image_path)

if __name__ == "__main__":
    main()
