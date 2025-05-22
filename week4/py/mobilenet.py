from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests

url = "https://encrypted-tbn2.gstatic.com/licensed-image?q=tbn:ANd9GcSgT_X5joXfhoCsRZ88aPNAAywP5acTcYomcKuM09o2qcTSvrYhM8afvZbOdQSKybyncWbBAS1F-RdOCLY"
image = Image.open(requests.get(url, stream=True).raw)

# initialize processor and model
preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
model = AutoModelForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")

# preprocess the inputs
inputs = preprocessor(images=image, return_tensors="pt")

# get the output and the class labels
outputs = model(**inputs)
logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
