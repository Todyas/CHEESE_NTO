"""
Feature engineering and embeddings for engagement prediction.
"""
import re
import numpy as np
import pandas as pd
from io import BytesIO
import base64
from PIL import Image

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # type: ignore[misc, assignment]
    print("BeautifulSoup not found")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[misc, assignment]
    print("sentence-transformers not found")

try:
    import torch
    from torchvision import models, transforms
except ImportError:
    torch = None  # type: ignore[assignment]
    models = None  # type: ignore[assignment]
    transforms = None  # type: ignore[assignment]
    print("torch and torchvision not found")


# --- Text preprocessing ---

def decode_html_entities(text: str) -> str:
    """Decode HTML numeric entities like &#10024; to Unicode."""
    if not text:
        return ""
    def replace_entity(m):
        try:
            return chr(int(m.group(1)))
        except (ValueError, OverflowError):
            return m.group(0)
    return re.sub(r"&#(\d+);", replace_entity, str(text))


def strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    if not text:
        return ""
    if BeautifulSoup is not None:
        return BeautifulSoup(str(text), "html.parser").get_text(separator=" ")
    return re.sub(r"<[^>]+>", " ", str(text))


def preprocess_text(text: str) -> str:
    """Full text preprocessing: strip HTML, decode emoji entities, normalize whitespace."""
    if pd.isna(text) or text == "":
        return ""
    t = decode_html_entities(str(text))
    t = strip_html(t)
    return " ".join(t.split())


# --- Text feature engineering ---

def count_emoji_entities(text: str) -> int:
    """Count HTML emoji entities like &#10024;"""
    if not text:
        return 0
    return len(re.findall(r"&#\d+;", str(text)))


def extract_text_features(series: pd.Series) -> np.ndarray:
    """
    Extract handcrafted text features.
    Returns array of shape (n_samples, 6): text_len, word_count, emoji_count, has_links, br_count, html_clean_len
    """
    rows = []
    for t in series:
        t = "" if pd.isna(t) else str(t)
        text_len = len(t)
        word_count = len(t.split()) if t else 0
        emoji_count = count_emoji_entities(t)
        has_links = 1.0 if ("http" in t or "www." in t) else 0.0
        br_count = t.lower().count("<br>") + t.lower().count("<br/>") + t.lower().count("<br />")
        clean = preprocess_text(t)
        html_clean_len = len(clean)
        rows.append([text_len, word_count, emoji_count, has_links, br_count, html_clean_len])
    return np.array(rows, dtype=np.float64)


# --- Image feature engineering ---

def img_vectorizer(photo_base64: str) -> np.ndarray:
    """
    Extract handcrafted image features from base64 photo.
    Fixes: reshape instead of resize, proper grayscale expansion.
    Adds: aspect_ratio, pixel_count.
    """
    try:
        img = np.array(Image.open(BytesIO(base64.b64decode(photo_base64))))
    except Exception:
        return np.zeros(22, dtype=np.float64)  # fallback for decode failure

    s = img.shape
    if len(s) == 2:
        img = np.stack([img, img, img], axis=-1)

    h, w = s[0], s[1]
    img = img.reshape(-1, 3)

    aspect_ratio = w / h if h > 0 else 0.0
    pixel_count = h * w

    stats = [
        np.array([h, w], dtype=np.float64),
        np.array([aspect_ratio, pixel_count], dtype=np.float64),
        img.min(axis=0).astype(np.float64),
        img.max(axis=0).astype(np.float64),
        img.mean(axis=0),
        img.std(axis=0),
        np.median(img, axis=0),
    ]
    cm = np.corrcoef(img.T)
    if np.any(np.isnan(cm)):
        cm = np.zeros((3, 3))
    stats.append(cm[np.triu_indices(len(cm), k=1)].astype(np.float64))
    return np.concatenate(stats)


def extract_img_features(photo_series: pd.Series) -> np.ndarray:
    """Extract handcrafted image features for all photos."""
    return np.vstack(list(photo_series.map(img_vectorizer)))


# --- Text embeddings ---

def get_text_embeddings(
    texts: pd.Series,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    model=None,
) -> tuple[np.ndarray, object]:
    """
    Get sentence embeddings for texts.
    Returns (embeddings, model). Pass model for inference (transform only).
    """
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is required. pip install sentence-transformers")

    if model is None:
        model = SentenceTransformer(model_name)

    cleaned = texts.fillna("").astype(str).map(preprocess_text)
    emb = model.encode(cleaned.tolist(), show_progress_bar=len(cleaned) > 1000)
    return np.array(emb, dtype=np.float64), model


# --- Image embeddings ---

def _get_image_transform():
    if transforms is None:
        raise ImportError("torchvision is required")
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _base64_to_tensor(photo_base64: str):
    """Decode base64 to tensor (3, 224, 224)."""
    try:
        img = Image.open(BytesIO(base64.b64decode(photo_base64))).convert("RGB")
    except Exception:
        return None
    return _get_image_transform()(img)


def get_image_embeddings(
    photo_series: pd.Series,
    model=None,
    batch_size: int = 32,
    device: str = 'cpu',
) -> tuple[np.ndarray, object]:
    """
    Extract ResNet18 penultimate layer features (512-dim) from images.
    Returns (embeddings, model).
    """
    if torch is None or models is None:
        raise ImportError("torch and torchvision are required")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model is None:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()  # type: ignore[assignment]
        model.eval()
        model.to(device)

    tensors = []
    for p in photo_series:
        t = _base64_to_tensor(p)
        if t is None:
            t = torch.zeros(3, 224, 224)
        tensors.append(t)

    stack = torch.stack(tensors).to(device)
    embs = []
    with torch.no_grad():
        for i in range(0, len(stack), batch_size):
            batch = stack[i : i + batch_size]
            out = model(batch)
            embs.append(out.cpu().numpy())

    return np.vstack(embs).astype(np.float64), model


# --- Load saved artifacts ---

def load_text_embedder(path: str = "text_embedder"):
    """Load saved sentence-transformers model."""
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers required")
    return SentenceTransformer(path)


def load_image_backbone(path: str = "image_backbone.pt"):
    """Load saved ResNet18 backbone."""
    if torch is None or models is None:
        raise ImportError("torch and torchvision required")
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Identity()  # type: ignore[assignment]
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model


# --- Combined pipeline ---

def build_features(
    df: pd.DataFrame,
    text_embedder=None,
    image_backbone=None,
    tf_vectorizer=None,
    fit: bool = False,
    text_embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
    use_text_embeddings: bool = True,
    use_text_handcrafted: bool = True,
    use_image_embeddings: bool = True,
    use_image_handcrafted: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Build full feature matrix.
    Returns (X, artifacts) where artifacts = {text_embedder, image_backbone, tf_vectorizer, ...}
    """
    parts = []
    artifacts = {}

    if use_text_embeddings and SentenceTransformer is not None:
        texts = df["text"].fillna("")
        X_te, model = get_text_embeddings(texts,  # type: ignore[arg-type]
            model_name=text_embedding_model,
            model=text_embedder,
        )
        parts.append(X_te)
        artifacts["text_embedder"] = model

    if use_text_handcrafted:
        X_tf = extract_text_features(df["text"])  # type: ignore[arg-type]
        parts.append(X_tf)

    if use_image_embeddings and torch is not None:
        X_ie, backbone = get_image_embeddings(df["photo"], model=image_backbone)  # type: ignore[arg-type]
        parts.append(X_ie)
        artifacts["image_backbone"] = backbone

    if use_image_handcrafted:
        X_img = extract_img_features(df["photo"])  # type: ignore[arg-type]
        parts.append(X_img)

    X = np.hstack(parts)
    return X, artifacts
