import numpy as np
import pandas as pd
import pickle

import features

test = pd.read_csv("test.csv")
test["text"] = test["text"].fillna("")

# Load saved artifacts (must match fit.ipynb config)
text_embedder = features.load_text_embedder("text_embedder")

# Build features - flags must match training (fit.ipynb)
X, _ = features.build_features(
    test,
    text_embedder=text_embedder,
    use_text_embeddings=True,
    use_text_handcrafted=True,
    use_image_embeddings=False,
    use_image_handcrafted=False,
)

target = ["like", "comment", "hide", "expand", "open_photo", "open", "share_to_message"]

with open("model.pickle", "rb") as f:
    model = pickle.load(f)

prediction = pd.DataFrame()
for column in target:
    reg = model[column]
    y = reg.predict(X)
    prediction[column] = y * test["view"]

prediction.to_csv("submission.csv", index=False)
