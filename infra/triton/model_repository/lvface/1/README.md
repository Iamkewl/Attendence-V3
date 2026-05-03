# lvface model artifact layout

Put the multi-task face model artifact in this folder as:

- `model.onnx`

Expected outputs from this model:

- `LIVENESS__0` with per-face logits/probabilities for binary liveness
- `EMBEDDING__1` with 512-dimensional face embeddings
