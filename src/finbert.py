import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_finbert_model(
    model_name: str = "ProsusAI/finbert"
):
    """
    Load FinBERT tokenizer and model.

    FinBERT classifies financial text into:
        positive, negative, neutral
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    model.eval()

    return tokenizer, model


def score_texts_finbert(
    texts,
    tokenizer,
    model,
    batch_size: int = 16,
    max_length: int = 128
) -> pd.DataFrame:
    """
    Score a list of texts using FinBERT.

    Returns:
        finbert_positive
        finbert_negative
        finbert_neutral
        signal_finbert

    signal_finbert = P(positive) - P(negative)
    """

    all_rows = []

    for start in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[start:start + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

        # ProsusAI/finbert labels are usually:
        # 0 = positive, 1 = negative, 2 = neutral
        for p in probs:
            pos = float(p[0])
            neg = float(p[1])
            neu = float(p[2])

            all_rows.append({
                "finbert_positive": pos,
                "finbert_negative": neg,
                "finbert_neutral": neu,
                "signal_finbert": pos - neg,
            })

    return pd.DataFrame(all_rows)


def add_finbert_signal(
    df: pd.DataFrame,
    text_col: str = "text",
    batch_size: int = 16,
    max_length: int = 128
) -> pd.DataFrame:
    """
    Add FinBERT sentiment columns to a dataframe.
    """

    out = df.copy()

    tokenizer, model = load_finbert_model()

    texts = out[text_col].fillna("").astype(str).tolist()

    scores = score_texts_finbert(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        batch_size=batch_size,
        max_length=max_length
    )

    out = pd.concat(
        [out.reset_index(drop=True), scores.reset_index(drop=True)],
        axis=1
    )

    return out
