# Greek–English Translation Experiments

This repository contains a set of **controlled experiments for Greek → English machine translation**, comparing three different approaches:

1. **From-scratch Transformer**
2. **Pretrained sequence-to-sequence model** 
3. **Large Language Model (LLM)**

The goal is to **understand performance and trade-offs** across classical neural machine translation, transfer learning, and modern LLM-based translation.

We use the **OPUS Books** dataset (`Helsinki-NLP/opus_books`) from Hugging Face for Greek (`el`) → English (`en`) translation.

**Note:** This repository focuses on comparative behavior and trade-offs, not on achieving state-of-the-art MT performance.

---

## Contents

- `src/` - Core source code for all translation experiments, including model implementations, training scripts, evaluation logic, and utilities.
- `auxiliary/` - Exploratory code, small experiments, and supporting scripts and files used during development.
- `requirements.txt` - Project dependencies.
- `dataset` - Train and validation splits used in all experiments.
- `results` - Experiment outputs, including trained model weights, TensorBoard logs and graphs, evaluation logs, and JSON files with computed metrics for all models.
- `README.md` - Project documentation.

---

## Installation

### Prerequisites
- Python **3.10** recommended
- (Optional) NVIDIA GPU with CUDA for faster training

### Create a virtual environment

**macOS / Linux**
```
python -m venv nmt_venv
source nmt_venv/bin/activate
pip install -r requirements.txt
```

**Windows**
```
python -m venv nmt_venv
.\nmt_venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Quickstart

Clone the repository and move into the project directory.

Make sure the virtual environment is activated and dependencies are installed before running the commands below.

Train and evaluate the from-scratch Transformer with:
```
python src\train.py
```

Monitor training with TensorBoard:
```
tensorboard --logdir results/runs
```

Translate with the trained model (custom input or dataset index):
```
python src\translate.py "Τραβάει ένα χέρι καλοθρεμμένο από το κρεββάτι"
python src\translate.py 42
```

Evaluate the pretrained Transformer on the same validation split:
```
python src\pretrained_model.py
```

Translate with the pretrained model:
```
python src\translate_pretrained.py "Τραβάει ένα χέρι καλοθρεμμένο από το κρεββάτι"
python src\translate_pretrained.py 42
```

Evaluate the LLM-based model on the same validation split:
```
python src\llm_model.py
```

Translate with the LLM model:
```
python src\translate_llm.py "Τραβάει ένα χέρι καλοθρεμμένο από το κρεββάτι"
python src\translate_llm.py 42
```

---

## Results

Evaluation is performed using standard machine translation metrics:

- BLEU
- Word Error Rate (WER)
- Character Error Rate (CER)

| Method                   | BLEU   | WER    | CER    |
|--------------------------|--------|--------|--------|
| From-scratch Transformer | 0.0354 | 1.2337 | 0.8179 |
| Pretrained Transformer   | 0.1140 | 0.8496 | 0.6532 |
| LLM Approach             | 0.1338 | 0.8268 | 0.6379 |

All metrics are computed on the same validation split with identical preprocessing.

### Example Translation Results

We present qualitative translation examples to complement the quantitative evaluation. 

---

#### Example 1: Custom Input Sentence
- **SOURCE (Greek)**: Τραβάει ένα χέρι καλοθρεμμένο από το κρεβάτι
- **From-scratch Transformer Prediction**: "But, sir," replied Candide, "I have a so much from her mother on the old woman, and so."
- **Pretrained Transformer Prediction**: He pulls a well-fed hand out of bed
- **LLM Prediction**: It pulls a well-fed hand from the bed.

#### Example 2: Dataset Sample (OPUS Books, ID = 42)
- **SOURCE (Greek)**: Έχετε δίκαιο, είπεν ο Αγαθούλης· αυτό μούλεγε πάντα ο κύριος Παγγλώσσης και παρατηρώ καλά, πως όλα είναι άριστα.
- **TARGET (Reference English)**: "You are perfectly right, gentlemen," said Candide, "this is precisely the doctrine of Master Pangloss; and I am convinced that everything is for the best.
- **From-scratch Transformer Prediction**: But, Reverend Father," said Candide, "this is the best of this world is true, for I am convinced that everything is for the best.
- **Pretrained Transformer Prediction**: You are right," said the Master. "That is what Mr. Pangloss always said, and I see that everything is excellent.
- **LLM Prediction**: You are right," said Agathoulis; "this is what Mr. Pangloss always told me, and I observe well that everything is excellent.

#### Discussion

The following observations summarize qualitative differences between the three approaches:

- From-scratch Transformer
  - Frequently fails to preserve meaning, especially on short or literal inputs
  - Exhibits hallucinations, grammatical errors, and stylistic drift
  - Demonstrates the limitations of training NMT models from scratch with limited data
- Pretrained Transformer
  - Produces fluent and semantically accurate translations
  - Performs reliably on both literal and literary text, outperforming the custom transformer
  - Occasional paraphrasing is expected and acceptable for literary translation
- LLM-based Approach
  - Matches or slightly outperforms the pretrained Transformer in fluency and style
  - Produces near-identical outputs to the pretrained model on short sentences
  - Better preserves tone, proper nouns, and narrative style without task-specific fine-tuning
- Overall Observations
  - Qualitative results are consistent with quantitative metrics
  - Pretrained and LLM models significantly outperform the from-scratch model in BLEU, WER, and CER
  - The LLM achieves the strongest overall performance across our experiments

---

## Main Takeaways

- Training from scratch is not competitive without large-scale data and compute, even when the architecture is correct.
- Pretraining provides the largest performance gain, dramatically improving fluency and semantic accuracy.
- LLM prompting achieves comparable or superior translation quality without any task-specific training.
- LLMs are particularly strong for low-resource or rapid-prototyping scenarios, where training or fine-tuning is impractical.
- For controlled and efficient deployment, pretrained NMT models remain a strong baseline, while LLMs offer maximum flexibility and quality at higher inference cost.

---

## Practical Considerations

- Performance vs. Cost Trade-off
  - LLMs achieve the best translation quality with minimal engineering effort
  - This comes at the cost of higher inference latency and monetary cost compared to local NMT models
- Deployment Considerations
  - Pretrained Transformers are better suited for low-latency, high-throughput, and cost-sensitive deployments
  - LLM-based translation is ideal for on-demand, low-volume, or quality-critical use cases
- Agentic Workflow Potential
  - LLM translation can be embedded into agentic pipelines (e.g., routing, quality estimation, post-editing)
  - Agents can dynamically choose between from-scratch, pretrained, or LLM translation based on context, cost, and quality requirements

---

## Contributing

Contributions and feedback are welcome! Feel free to open issues or submit pull requests.

---

