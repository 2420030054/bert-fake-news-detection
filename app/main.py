
from pathlib import Path

import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Paths
ROOT = Path(__file__).resolve().parent.parent
APP_DIR = ROOT / "app"
MODEL_DIR = ROOT / "models" / "bert_fake_news"

# FastAPI app
app = FastAPI(title="True or False")
app.mount("/static", StaticFiles(directory=APP_DIR), name="static")

# Select device
device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Load BERT model and tokenizer
tokenizer = None
model = None
lime_explainer = None
class_names = []

if MODEL_DIR.exists():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR)
    )
    model.to(device)
    model.eval()

    # Get the class names in the model's output order
    class_names = [
        model.config.id2label.get(i, str(i))
        for i in range(model.config.num_labels)
    ]

    # Initialize LIME
    lime_explainer = LimeTextExplainer(
        class_names=class_names
    )

    print("BERT model loaded successfully.")
    print("Classes:", class_names)
    print("LIME explainer initialized.")


class PostRequest(BaseModel):
    text: str


def predict_probabilities(texts):
    """
    Return class probabilities for a list of texts.
    LIME calls this function with many slightly modified texts.
    """
    if model is None or tokenizer is None:
        raise RuntimeError("BERT model is not loaded.")

    inputs = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)

    return probabilities.cpu().numpy()


@app.get("/")
def home():
    return FileResponse(APP_DIR / "index.html")


@app.get("/result")
def result_page():
    return FileResponse(APP_DIR / "result.html")


@app.post("/predict")
def predict(request: PostRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Please enter a post."
        )

    if len(text) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Post must be 1000 characters or fewer."
        )

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=500,
            detail="BERT model not found. Check models/bert_fake_news."
        )

    # Get the original prediction
    probabilities_array = predict_probabilities([text])[0]
    probabilities = probabilities_array.tolist()

    predicted_index = int(np.argmax(probabilities))
    label = class_names[predicted_index]

    # Generate LIME explanation for the predicted class
    explanation_words = []

    if lime_explainer is not None:
        try:
            lime_result = lime_explainer.explain_instance(
                text_instance=text,
                classifier_fn=predict_probabilities,
                labels=[predicted_index],
                num_features=8,
                num_samples=500
            )

            explanation_words = [
                {
                    "word": word,
                    "weight": float(weight)
                }
                for word, weight in lime_result.as_list(
                    label=predicted_index
                )
            ]

        except Exception as error:
            print("LIME explanation error:", repr(error))

    # Return prediction, probabilities and explanation
    return {
        "label": label,
        "probabilities": {
            class_names[i]: float(probability)
            for i, probability in enumerate(probabilities)
        },
        "explanation": explanation_words
    }
