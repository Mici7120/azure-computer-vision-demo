import os
from dotenv import load_dotenv
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

load_dotenv()

def sample_objects_image_file():
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
  with open("sample.jpg", "rb") as f:
    image_data = f.read()

  # Detect objects in an image stream. This will be a synchronously (blocking) call.
  result = client.analyze(
    image_data=image_data,
    visual_features=[VisualFeatures.OBJECTS]
  )

  # Print Objects analysis results to the console
  print("Image analysis results:")
  print(" Objects:")
  if result.objects is not None:
    for object in result.objects.list:
      print(f"   '{object.tags[0].name}', {object.bounding_box}, Confidence: {object.tags[0].confidence:.4f}")
  print(f" Image height: {result.metadata.height}")
  print(f" Image width: {result.metadata.width}")
  print(f" Model version: {result.model_version}")

if __name__ == "__main__":
  sample_objects_image_file()