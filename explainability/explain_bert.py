
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer

# 1. Select the available device
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)

# 2. Load your saved BERT model and tokenizer
model_path = "models/bert_fake_news"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# 3. Get the class names from the saved model
class_names = [
    model.config.id2label[i]
    for i in range(model.config.num_labels)
]

print("Classes:", class_names)

# 4. Define a function that returns class probabilities
def predict_probabilities(texts):
    inputs = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)

    return probabilities.cpu().numpy()

# 5. Create the LIME explainer
explainer = LimeTextExplainer(class_names=class_names)

# 6. Ask the user for a tweet
tweet = input("\nEnter a tweet to explain: ")

# 7. Generate an explanation
prediction = predict_probabilities([tweet])[0]
predicted_index = int(np.argmax(prediction))

print("\nPredicted label:", class_names[predicted_index])
print("Model confidence scores:")

for label, probability in zip(class_names, prediction):
    print(f"{label}: {probability:.2%}")

print("\nGenerating LIME explanation...")
explanation = explainer.explain_instance(
    tweet,
    predict_probabilities,
    num_features=10,
    num_samples=500,
    labels=[predicted_index],
)

print("\nWords influencing the predicted class:")
for word, weight in explanation.as_list(label=predicted_index):
    print(f"{word}: {weight:+.4f}")

# 8. Save a visual explanation as an HTML file
explanation.save_to_file("explainability/lime_explanation.html")

print("\nExplanation saved to explainability/lime_explanation.html")
