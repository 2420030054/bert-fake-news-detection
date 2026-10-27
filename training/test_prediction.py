
@app.post("/predict")
def predict(request: PostRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Please enter a post.")

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

    # Function for getting BERT probabilities for one or more texts
    def predict_probabilities(texts):
        inputs = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)

        return probabilities.cpu().numpy()

    # Predict the submitted post
    probs = predict_probabilities([text])[0]
    index = int(np.argmax(probs))
    label = model.config.id2label.get(index, str(index))

    # Generate LIME explanation for the predicted class
    explanation_words = []

    if lime_explainer is not None:
        try:
            lime_result = lime_explainer.explain_instance(
                text,
                predict_probabilities,
                labels=[index],
                num_features=8,
                num_samples=500
            )

            explanation_words = [
                {
                    "word": word,
                    "weight": float(weight)
                }
                for word, weight in lime_result.as_list(label=index)
            ]

        except Exception as error:
            print("LIME explanation error:", error)

    return {
        "label": label,
        "probabilities": {
            model.config.id2label.get(i, str(i)): float(prob)
            for i, prob in enumerate(probs)
        },
        "explanation": explanation_words
    }
