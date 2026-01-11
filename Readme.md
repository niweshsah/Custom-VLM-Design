Here is the technical design documentation formatted as a professional `README.md` file.

---

# Industrial PCB Inspection VLM: Low-Latency, Structured Output System

## Overview

This project implements a custom Vision-Language Model (VLM) designed specifically for offline, high-precision Printed Circuit Board (PCB) inspection. Unlike general-purpose VLMs (LLaVA, BLIP-2), this architecture is engineered to handle dense, small-object industrial metrology.

The system processes natural language queries regarding defects and returns structured, deterministic JSON outputs containing bounding box locations and confidence scores. It is optimized for inference times under 2 seconds on edge hardware (e.g., RTX A4000, Orin) and is trained on a dataset of 50,000 images with bounding box annotations (no native QA pairs).

## Key System Constraints & Objectives

* **Inference Latency:** < 2 seconds (offline).
* **Input Data:** 50k images with bounding box annotations.
* **Output Format:** Structured JSON (Location + Confidence).
* **Hallucination Rate:** Near-zero (Strict architectural constraints).
* **Deployment:** Offline, localized hardware.

---

## 1. Architecture Design

The system rejects the standard "Global Image Token" approach used in models like LLaVA, which leads to poor spatial grounding for small defects. Instead, it utilizes a **Detection-First, Language-Second** architecture.

### High-Level Pipeline

1. **Input:** High-resolution PCB Image (1024x1024).
2. **Vision Encoder:** Detection-aware backbone (ViT-Det or ConvNeXt).
3. **Region Proposal:** Extracts Region of Interest (ROI) features.
4. **Feature Flattening:** Converts ROIs into multimodal tokens: `[visual_features | spatial_coords]`.
5. **Fusion Layer:** Cross-attention between Text Queries and ROI tokens.
6. **LLM Decoder:** Small decoder-only model (1-3B parameters).
7. **Output:** JSON structure strictly defined by a schema.

### Component Selection

| Component | Choice | Rationale |
| --- | --- | --- |
| **Vision Backbone** | **ViT-Det / ConvNeXt-Det** | Pre-trained on detection tasks to preserve fine spatial details essential for micro-defects. |
| **Language Model** | **Small Decoder (1-3B)** | (e.g., Qwen-1.5, Phi, Mistral) Sufficient for reasoning without the overhead of 7B+ models. |
| **Fusion Mechanism** | **Region-Level Grounding** | Replaces global CLIP embeddings. The LLM only "sees" cropped regions, preventing hallucinations of non-existent areas. |

---

## 2. Model Selection Strategy

### Why Custom Architecture?

Standard VLMs are unsuitable for industrial inspection due to the following limitations:

* **LLaVA:** Uses global vision tokens, resulting in poor localization and high hallucination rates for small objects.
* **BLIP-2:** Query-based compression loses fine-grained defect texture details.
* **Qwen-VL:** Too heavy for edge inference; localization via text tokens is imprecise.

### Parameter Budget

To ensure the < 2s latency requirement:

* **Vision:** < 100M parameters.
* **Language:** 1B - 3B parameters.
* **Total:**  4B parameters.

### Licensing

All selected base models adhere to Apache 2.0 or MIT licenses to ensure commercial viability without legal ambiguity.

---

## 3. Implementation of Localization

Localization is enforced architecturally, not learned implicitly.

**Token Structure:**
Instead of a single `[CLS]` token, the visual input to the LLM is a sequence of Region of Interest tokens:
`[ROI_1, ROI_2, ..., ROI_N]`

Each token contains concatenated vectors:


**Output Schema:**
The model is constrained to output strict JSON. Free-form text generation is disabled during production.

```json
{
  "defect_type": "solder_bridge",
  "bbox": [102, 450, 115, 463],
  "confidence": 0.93
}

```

---

## 4. Optimization Strategy (< 2s Inference)

### Compression Techniques

1. **Quantization:**
* Vision Encoder: INT8.
* LLM: INT4 (using GPTQ or AWQ).


2. **Pruning:**
* Removal of attention heads not involved in Region-Text interaction.


3. **Distillation:**
* Teacher: Large VLM.
* Student: Compact VLM focused on ROI selection and confidence calibration.



### Pipeline Optimization

* **ROI Caching:** Run the vision detector once per image; answer multiple natural language queries against the cached ROI features.
* **LoRA (Low-Rank Adaptation):** Fine-tuning is applied only to cross-attention layers and the final decoder heads, keeping the bulk of the LLM frozen.

---

## 5. Hallucination Mitigation

Hallucination is the primary risk in industrial AI. We mitigate this through three layers of defense.

### Architectural Safeguards

* **No Global Tokens:** The LLM cannot access the entire image context vaguely; it can only process specific regions proposed by the detector.
* **Constrained Decoding:** Enforced JSON schema preventing the generation of unstructured text.

### Defect Ontology

The model is restricted to a defined set of classes (e.g., `solder_bridge`, `missing_pad`, `short`, `open_trace`). It cannot generate novel defect names.

### Training Strategies

1. **Negative QA Training:** Explicitly training on clean boards where the correct answer is `no_defect` or an empty JSON list.
2. **Abstention Training:** Training the model to output low confidence scores or "unknown" states when visual evidence is ambiguous.
3. **Confidence Calibration:** Optimization using Brier score and Expected Calibration Error (ECE) to penalize high-confidence errors.

**Loss Function Formulation:**



*Where  penalizes answers referencing non-existent ROIs.*

---

## 6. Training Plan

Since the dataset consists of 50k images with bounding boxes but no QA pairs, a multi-stage synthetic data strategy is required.

**Stage 1: Vision Pretraining**

* Train the detector backbone.
* **Target:** Recall > 95% (It is better to propose false positives than miss defects).

**Stage 2: Synthetic QA Generation**

* Procedurally generate questions based on ground truth bounding boxes.
* *Example:* "Where are the solder bridges?" -> Map to coordinates of class ID `solder_bridge`.

**Stage 3: Vision-Language Alignment**

* Freeze vision encoder.
* Train cross-attention layers to align text queries with ROI tokens.

**Stage 4: Stress Testing**

* Train on empty boards, partial crops, and noise-injected images to force the model to predict "safe" outputs (no defects).

---

## 7. Validation & Metrics

Evaluation focuses on metrology and reliability rather than linguistic fluency.

1. **Localization Precision:**
* Intersection over Union (IoU) @ 0.5 and 0.75.
* Mean Localization Error (pixels).


2. **Counting Accuracy:**
* Mean Absolute Error (MAE) on defect counts.
* Exact Match Accuracy.


3. **Hallucination Rate:**
* **Metric:** False Positive Rate (Model detects defect where Ground Truth is empty).
* **Target:** < 0.1% for critical defects.


4. **Operational Metrics:**
* JSON Validity Rate: Must be 100%.
* Latency: Average time per query (Target < 2s).



---

## Design Philosophy

**Do not ask a general VLM to "understand" a PCB.**

This system treats the Language Model not as a visionary, but as a logic processor for a deterministic vision system. The LLM's role is strictly to interpret the user's query and format the Vision Encoder's findings into a structured response.
