import os
from dotenv import load_dotenv
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

load_dotenv()

def caption():
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
  with open("scene_sample.jpg", "rb") as f:
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

if __name__ == "__main__":
  caption()