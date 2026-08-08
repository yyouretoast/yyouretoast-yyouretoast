# Yassin Yasser
Applied AI Engineer & CS Student | Cairo, Egypt

[LinkedIn](https://www.linkedin.com/in/yassinyasser/) | [Email](mailto:yyasso2005@gmail.com) | [Hugging Face Space](https://huggingface.co/spaces/yyouretoast/deepfake-detector)

---

## Tech Stack
PyTorch, OpenCV, SciPy, LangGraph, ChromaDB, SQLite, Docker, GCP, Streamlit

---

## Projects

### [Dual-Stream Deepfake Detector](https://github.com/yyouretoast/deepfake-detection) | [Live Demo](https://huggingface.co/spaces/yyouretoast/deepfake-detector)
- Fuses ConvNeXt-Small spatial embeddings with SRM + Bayar 2D Real FFT spectral embeddings (0.9987 test AUC).
- Graph connected-component identity partitioning (`networkx.Graph`) to enforce 0% identity leakage across splits.
- SciPy L-BFGS-B temperature scaling (0.0093 ECE) with 4-panel Grad-CAM diagnostics and a 54-test `pytest` suite.

### [FreightIQ](https://github.com/yyouretoast/freightiq)
- LangGraph multi-tool agentic RAG system for carrier dispatch and freight class calculations.
- Two-stage retrieval combining ChromaDB vector search with a custom PyTorch `CarrierReRanker` MLP model.
- Read-only SQLite connection isolation, SELECT-only SQL validation, and exponential backoff retry wrappers.

---

## Open Source

### [huggingface/transformers](https://github.com/huggingface/transformers/pull/46770)
- Fixed documentation bug miscategorizing `SmolLM3` under Multimodal models. Merged into `huggingface:main`.
