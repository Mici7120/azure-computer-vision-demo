"""
Azure Computer Vision API Demo
==============================

This script demonstrates how to use the Azure Computer Vision API for various tasks:
1. Image Analysis - Detect objects, tags, and extract visual features
2. OCR (Optical Character Recognition) - Extract text from images
3. Face Detection - Detect faces and analyze attributes
4. Image Description - Generate image captions

Prerequisites:
- Azure account with Computer Vision resource
- Python 3.6+
- Required packages: requests, Pillow
"""

import requests
import json
import time
import os
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Configuration - Replace with your own values
# ============================================

ENDPOINT = os.environ["VISION_ENDPOINT"]
API_KEY = os.environ["VISION_KEY"]

# Helper functions
# ================

def display_image(image_path, title="Image"):
    """Display an image using PIL"""
    try:
        img = Image.open(image_path)
        print(f"Displaying {title} - Size: {img.size}")
        # In a real environment, this would display the image
        # img.show()
        return img
    except Exception as e:
        print(f"Error displaying image: {e}")
        return None

def save_image_with_annotations(original_image, annotations, output_path):
    """Save image with annotations drawn on it"""
    img_draw = ImageDraw.Draw(original_image)
    
    # Simple font for annotations
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    
    for annotation in annotations:
        x, y, w, h = annotation["box"]
        label = annotation["label"]
        
        # Draw rectangle
        img_draw.rectangle([(x, y), (x+w, y+h)], outline="red", width=2)
        
        # Draw label
        img_draw.text((x, y-20), label, fill="red", font=font)
    
    original_image.save(output_path)
    print(f"Annotated image saved to {output_path}")
    return original_image

# 1. Image Analysis
# ================

def analyze_image(image_path, features=None):
    """
    Analyze an image using Azure Computer Vision API.
    
    Parameters:
        image_path: Path to the image file
        features: List of features to analyze, e.g., ["objects", "tags", "read"]
    
    Returns:
        JSON response from the API
    """
    if not features:
        features = ["tags", "objects", "caption"]
    
    analyze_url = f"{ENDPOINT}computervision/imageanalysis:analyze"
    params = {
        "features": ",".join(features),
        "language": "en",
        "gender-neutral-caption": "false"
    }
    
    headers = {
        "Ocp-Apim-Subscription-Key": API_KEY,
        "Content-Type": "application/octet-stream"
    }
    
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            
        response = requests.post(
            analyze_url, 
            headers=headers, 
            params=params, 
            data=image_data
        )
        response.raise_for_status()
        
        return response.json()
    except Exception as e:
        print(f"Error in analyze_image: {e}")
        return None

def demo_image_analysis(image_path="sample_image.jpg"):
    """Run a demonstration of image analysis"""
    print("\n=== Image Analysis Demo ===")
    
    # Display the image
    img = display_image(image_path, "Original Image")
    
    # Analyze the image
    features = ["tags", "objects", "caption", "denseCaptions", "smartCrops"]
    results = analyze_image(image_path, features)
    
    if not results:
        print("Failed to analyze image.")
        return
    
    # Print image caption
    if "caption" in results:
        caption = results["caption"]["text"]
        confidence = results["caption"]["confidence"]
        print(f"\nImage Caption: {caption} (Confidence: {confidence:.2f})")
    
    # Print tags
    if "tags" in results:
        print("\nTags detected:")
        for tag in results["tags"]:
            print(f"- {tag['name']} (Confidence: {tag['confidence']:.2f})")
    
    # Print objects
    if "objects" in results:
        print("\nObjects detected:")
        annotations = []
        
        for obj in results["objects"]:
            name = obj["tags"][0]["name"]
            confidence = obj["tags"][0]["confidence"]
            box = [obj["rectangle"]["x"], obj["rectangle"]["y"], 
                   obj["rectangle"]["w"], obj["rectangle"]["h"]]
            
            print(f"- {name} (Confidence: {confidence:.2f})")
            annotations.append({"box": box, "label": name})
        
        # Save image with object annotations
        if img and annotations:
            output_path = "objects_detected.jpg"
            save_image_with_annotations(img.copy(), annotations, output_path)
    
    print("\nImage analysis completed!")

# 2. OCR (Optical Character Recognition)
# =====================================

def read_image_text(image_path):
    """
    Extract text from an image using OCR.
    
    Parameters:
        image_path: Path to the image file
    
    Returns:
        Extracted text and bounding boxes
    """
    # For the new Azure Computer Vision 4.0, this is included in 'read' feature
    ocr_url = f"{ENDPOINT}computervision/imageanalysis:analyze"
    
    params = {
        "features": "read",
        "language": "en"
    }
    
    headers = {
        "Ocp-Apim-Subscription-Key": API_KEY,
        "Content-Type": "application/octet-stream"
    }
    
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        
        response = requests.post(
            ocr_url,
            headers=headers,
            params=params,
            data=image_data
        )
        response.raise_for_status()
        
        return response.json()
    except Exception as e:
        print(f"Error in read_image_text: {e}")
        return None

def demo_ocr(image_path="text_sample.jpg"):
    """Run a demonstration of OCR capabilities"""
    print("\n=== OCR Demo ===")
    
    # Display the image
    img = display_image(image_path, "Text Image")
    
    # Extract text
    ocr_results = read_image_text(image_path)
    
    if not ocr_results or "readResult" not in ocr_results:
        print("Failed to extract text from image.")
        return
    
    # Print extracted text
    print("\nExtracted Text:")
    extracted_text = ""
    annotations = []
    
    for block in ocr_results["readResult"]["blocks"]:
        for line in block["lines"]:
            text = line["text"]
            extracted_text += text + "\n"
            print(f"- {text}")
            
            # Get bounding box for annotation
            x = line["boundingBox"][0]
            y = line["boundingBox"][1]
            w = line["boundingBox"][2] - x
            h = line["boundingBox"][5] - y
            annotations.append({"box": [x, y, w, h], "label": text})
    
    # Save image with text annotations
    if img and annotations:
        output_path = "text_detected.jpg"
        save_image_with_annotations(img.copy(), annotations, output_path)
    
    print("\nOCR processing completed!")

# 3. Face Detection
# ================

def detect_faces(image_path):
    """
    Detect faces in an image using Azure Face API.
    
    Parameters:
        image_path: Path to the image file
    
    Returns:
        JSON response from the API
    """
    # Note: Face API is a separate service from Computer Vision
    face_api_url = f"{ENDPOINT}face/v1.0/detect"
    
    params = {
        "returnFaceId": "true",
        "returnFaceLandmarks": "false",
        "returnFaceAttributes": "age,gender,emotion,smile",
        "recognitionModel": "recognition_04"
    }
    
    headers = {
        "Ocp-Apim-Subscription-Key": API_KEY,
        "Content-Type": "application/octet-stream"
    }
    
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        
        response = requests.post(
            face_api_url,
            headers=headers,
            params=params,
            data=image_data
        )
        response.raise_for_status()
        
        return response.json()
    except Exception as e:
        print(f"Error in detect_faces: {e}")
        return None

def demo_face_detection(image_path="faces_sample.jpg"):
    """Run a demonstration of face detection"""
    print("\n=== Face Detection Demo ===")
    
    # Display the image
    img = display_image(image_path, "Faces Image")
    
    # Detect faces
    face_results = detect_faces(image_path)
    
    if not face_results:
        print("Failed to detect faces or no faces found.")
        return
    
    print(f"\nDetected {len(face_results)} faces:")
    
    annotations = []
    for i, face in enumerate(face_results):
        # Get face attributes
        age = face["faceAttributes"]["age"]
        gender = face["faceAttributes"]["gender"]
        emotion = max(face["faceAttributes"]["emotion"].items(), key=lambda x: x[1])[0]
        
        # Get rectangle coordinates
        rect = face["faceRectangle"]
        x = rect["left"]
        y = rect["top"]
        w = rect["width"]
        h = rect["height"]
        
        print(f"- Face {i+1}: Age ~{age}, {gender.capitalize()}, Emotion: {emotion}")
        annotations.append({
            "box": [x, y, w, h], 
            "label": f"Age: {int(age)}, {gender[0].upper()}"
        })
    
    # Save image with face annotations
    if img and annotations:
        output_path = "faces_detected.jpg"
        save_image_with_annotations(img.copy(), annotations, output_path)
    
    print("\nFace detection completed!")

# 4. Image Description
# ==================

def generate_description(image_path):
    """
    Generate a description for an image.
    
    Parameters:
        image_path: Path to the image file
    
    Returns:
        Generated description and confidence
    """
    # This is included in the 'caption' feature of analyze_image
    results = analyze_image(image_path, ["caption", "denseCaptions"])
    
    return results

def demo_image_description(image_path="scene_sample.jpg"):
    """Run a demonstration of image description generation"""
    print("\n=== Image Description Demo ===")
    
    # Display the image
    display_image(image_path, "Scene Image")
    
    # Generate description
    description_results = generate_description(image_path)
    
    if not description_results or "caption" not in description_results:
        print("Failed to generate description.")
        return
    
    # Print caption
    caption = description_results["caption"]["text"]
    confidence = description_results["caption"]["confidence"]
    print(f"\nImage Caption: {caption} (Confidence: {confidence:.2f})")
    
    # Print dense captions (detailed descriptions of image regions)
    if "denseCaptions" in description_results:
        print("\nDetailed Region Descriptions:")
        for i, caption in enumerate(description_results["denseCaptions"]):
            print(f"- Region {i+1}: {caption['text']} (Confidence: {caption['confidence']:.2f})")
    
    print("\nImage description completed!")

# Main demo function
# =================

def run_azure_vision_demo():
    """Run the complete Azure Vision API demo"""
    print("===== Azure Computer Vision API Demo =====")
    
    # Check if API key and endpoint are set
    if ENDPOINT == "https://your-resource-name.cognitiveservices.azure.com/" or API_KEY == "your_api_key_here":
        print("\nWARNING: Please set your actual Azure API key and endpoint before running this demo.")
        print("The demo will run with mock data for demonstration purposes.")
    
    # For demo purposes, we'll assume these image files exist
    # In a real implementation, you would need to provide actual image files
    
    # Run demos
    demo_image_analysis()
    #demo_ocr()
    #demo_face_detection()
    #demo_image_description()
    
    print("\n===== Demo Completed =====")
    print("\nTo use this demo with your own images:")
    print("1. Update the ENDPOINT and API_KEY variables with your Azure credentials")
    print("2. Provide paths to your own image files for each demo function")
    print("3. Install required packages: pip install requests pillow")

if __name__ == "__main__":
    run_azure_vision_demo()