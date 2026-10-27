
const form = document.getElementById("post-form");
const textarea = document.getElementById("post-text");
const charCount = document.getElementById("char-count");
const errorMessage = document.getElementById("error-message");
const checkButton = document.getElementById("check-button");

if (textarea) {
    textarea.addEventListener("input", () => {
        charCount.textContent = `${textarea.value.length}/1000`;
        errorMessage.textContent = "";
    });
}

if (form) {
    form.addEventListener("submit", async (event) => {
        event.preventDefault();

        const tweet = textarea.value.trim();
        if (!tweet) {
            errorMessage.textContent = "Please paste a post first.";
            return;
        }

        checkButton.disabled = true;
        checkButton.textContent = "Analyzing...";
        errorMessage.textContent = "";

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: tweet })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || "Could not analyze this post.");
            }

            const result = {
                label: data.label,
                text: tweet,
                probabilities: data.probabilities || {},
                explanation: data.explanation || []
            };

            sessionStorage.setItem("trueOrFalseResult", JSON.stringify(result));
            window.location.href = "/result";
        } catch (error) {
            errorMessage.textContent =
                error.message === "Failed to fetch"
                    ? "Cannot reach the server. Check that the FastAPI app is running."
                    : error.message;
        } finally {
            checkButton.disabled = false;
            checkButton.innerHTML = 'Check post <span>→</span>';
        }
    });
}

/* Render the results page */
const resultRoot = document.getElementById("result-root");

if (resultRoot) {
    const saved = sessionStorage.getItem("trueOrFalseResult");

    if (!saved) {
        window.location.replace("/");
    } else {
        const result = JSON.parse(saved);
        const label = String(result.label || "").toLowerCase();

        let state = "uncertain";
        let badge = "Uncertain";
        let heading = "This post needs more context.";
        let description = "The AI could not confidently categorize this post as true or false.";
        let symbol = "?";

        if (label === "true-positive" || label === "true" || label === "positive") {
            state = "positive";
            badge = "Positive";
            heading = "This post looks reliable.";
            description = "The AI classified this post as likely accurate based on patterns it learned.";
            symbol = "✓";
        } else if (label === "false-negative" || label === "false" || label === "negative") {
            state = "negative";
            badge = "Negative";
            heading = "This post looks suspicious.";
            description = "The AI classified this post as likely misinformation based on patterns it learned.";
            symbol = "×";
        }

        document.body.className = `result-page ${state}`;

        const resultLabel = document.getElementById("result-label");
        const resultHeading = document.getElementById("result-heading");
        const resultDescription = document.getElementById("result-description");
        const resultSymbol = document.getElementById("result-symbol");
        const submittedText = document.getElementById("submitted-text");
        const probabilities = document.getElementById("probabilities");
        const explanation = document.getElementById("explanation");

        resultLabel.textContent = badge;
        resultHeading.textContent = heading;
        resultDescription.textContent = description;
        resultSymbol.textContent = symbol;
        submittedText.textContent = result.text;

        for (const [name, value] of Object.entries(result.probabilities)) {
            const row = document.createElement("p");
            const numericValue = Number(value);
            row.textContent = `${name}: ${Number.isFinite(numericValue) ? (numericValue * 100).toFixed(2) + "%" : value}`;
            probabilities.appendChild(row);
        }

       if (Array.isArray(result.explanation) && result.explanation.length) {
    result.explanation.forEach(item => {
        const row = document.createElement("div");
        row.className = "lime-word";

        const word = document.createElement("span");
        word.className = "lime-token";
        word.textContent = item.word;

        const weight = Number(item.weight);
        const weightText = document.createElement("span");
        weightText.className =
            "lime-weight " + (weight >= 0 ? "supports" : "opposes");

        weightText.textContent =
            `${weight >= 0 ? "+" : ""}${weight.toFixed(4)}`;

        row.appendChild(word);
        row.appendChild(weightText);
        explanation.appendChild(row);
    });
} else {
    explanation.textContent = "No word-level explanation is available.";
}
    }
}

document.getElementById("check-again")?.addEventListener("click", () => {
    sessionStorage.removeItem("trueOrFalseResult");
    window.location.href = "/";
});

function showLimeExplanation(items) {
    const container = document.getElementById("explanation");

    if (!container) return;

    container.replaceChildren();

    if (!Array.isArray(items) || items.length === 0) {
        container.textContent = "No explanation available.";
        return;
    }

    items.forEach(item => {
        const row = document.createElement("div");
        row.className = "lime-word";

        const word = document.createElement("span");
        word.textContent = item.word;

        const weight = document.createElement("span");
        const value = Number(item.weight);

        weight.textContent =
            `${value >= 0 ? "+" : ""}${value.toFixed(4)}`;

        weight.className =
            value >= 0 ? "supports" : "opposes";

        row.append(word, weight);
        container.appendChild(row);
    });
}
