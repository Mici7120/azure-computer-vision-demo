import os
from dotenv import load_dotenv
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

load_dotenv()

def OCR():
  # Set the values of your computer vision endpoint and computer vision key
  # as environment variables:
  endpoint = os.getenv("VISION_ENDPOINT")
  key = os.getenv("VISION_KEY")
  
  if not endpoint or not key:
    print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
    print("Set them before running this sample.")
    exit()

  # Create an Image Analysis client
  client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
  )

  # Load image to analyze into a 'bytes' object
  with open("text_sample.jpg", "rb") as f:
    image_data = f.read()

  # Get a caption for the image. This will be a synchronously (blocking) call.
  result = client.analyze(
    image_data=image_data,
    visual_features=[VisualFeatures.CAPTION, VisualFeatures.READ, VisualFeatures.TAGS],
    gender_neutral_caption=True,  # Optional (default is False)
  )

  print("Image analysis results:")
  # Print caption results to the console
  print(" Caption:")
  if result.caption is not None:
    print(f"   '{result.caption.text}', Confidence {result.caption.confidence:.4f}")

  # Print text (OCR) analysis results to the console
  print(" Read:")
  if result.read is not None:
      for line in result.read.blocks[0].lines:
          print(f"   Line: '{line.text}', Bounding box {line.bounding_polygon}")
          for word in line.words:
              print(f"     Word: '{word.text}', Bounding polygon {word.bounding_polygon}, Confidence {word.confidence:.4f}")

if __name__ == "__main__":
    OCR()