# Image Captioning: Practical Research & Evaluation
**DS235 - Generative AI | Final Project** **American University of Armenia** **Authors:** Sona Barseghyan, Manuk Manukyan, Davit Hakobyan  

---

## 1. Project Description
This repository contains an empirical quantitative and qualitative evaluation of advanced generative AI models for the task of **Automatic Image Captioning**. The primary objective is to evaluate out-of-the-box, zero-shot generalization across different architectural approaches to understand how they bridge the modality gap between computer vision and natural language processing. 

All evaluations are conducted using the Hugging Face `transformers` library (Wolf et al., 2020) and are benchmarked against human consensus using standardized scoring algorithms.

---

## 2. Selected Methods & Justifications
We selected three state-of-the-art, open-source multimodal networks published in the 2020s. Each represents a distinct milestone in vision-language architecture.

### 1. BLIP (Bootstrapping Language-Image Pre-training)
* **Summary:** BLIP utilizes a Multimodal Mixture of Encoder-Decoder (MED) architecture. It addresses messy web data through a bootstrapping phase (CapFilt) that uses a synthetic captioner and a filter to clean the training dataset (Li et al., 2022; Hugging Face BLIP Documentation, n.d.).
* **Justification:** BLIP serves as our strong baseline model for unified vision-language understanding. 

### 2. BLIP-2 (with OPT-2.7b)
* **Summary:** Training massive models end-to-end is computationally prohibitive. BLIP-2 introduces the **Q-Former** (Querying Transformer) to extract visual features from a frozen image encoder and feed them as "soft prompts" into a frozen Large Language Model (Li et al., 2023; Hugging Face BLIP-2 Documentation, n.d.). Our configuration utilizes the 2.7-billion parameter OPT model (Zhang et al., 2022).
* **Justification:** It demonstrates how to leverage massive LLMs for image captioning with extremely high computational efficiency, serving as our state-of-the-art benchmark.

### 3. GIT (Generative Image-to-text Transformer)
* **Summary:** GIT completely removes the need for external object detectors or complex multi-task learning. It uses a single vision transformer to flatten image patches and passes them directly into a single text decoder (Wang et al., 2022; Hugging Face GIT Documentation, n.d.).
* **Justification:** GIT is included to represent a simplified, minimal, and highly scalable architectural approach to generative image-to-text tasks.

---

## 3. Test Dataset & Evaluation Metrics
### Dataset
All models were evaluated on a zero-shot basis using a subset of the **Flickr8k** dataset (Hodosh et al., 2013). This dataset contains 1,000 test images, each paired with 5 distinct human-written reference captions to ensure robust scoring and human consensus alignment.

### Quantitative Metrics
We utilized the COCO Evaluation Library (`pycocoevalcap`) (Chen et al., 2015; COCO Evaluation Library, n.d.) to compute the following metrics:
* **BLEU-4:** Measures exact 4-gram sequence matches to evaluate grammatical precision (Papineni et al., 2002).
* **METEOR:** Evaluates semantic alignment, penalizing fragmentation and accounting for synonyms/root words (Banerjee & Lavie, 2005).
* **ROUGE-L:** Measures recall based on the longest common subsequence (Lin, 2004).
* **CIDEr:** Consensus-based Image Description Evaluation; heavily weights descriptive importance and TF-IDF to match human judgment (Vedantam et al., 2015).

---

## 4. Installation & Setup Instructions
Because our evaluation uses the official MS COCO scoring tools, **Java is strictly required** to run the METEOR and CIDEr algorithms.

### Prerequisites
1. **Python 3.9+**
2. **Java Runtime Environment (JRE):** OpenJDK 11 or higher must be installed and added to your system path.

### Cross-Platform Setup

**1. Clone the repository:**
```bash
git clone [https://github.com/sonabarseghyan04/DS235_image_captioning.git](https://github.com/sonabarseghyan04/DS235_image_captioning.git)
cd DS235_image_captioning

```

**2. Install Java (If not already installed):**

* **macOS (Homebrew):** `brew install openjdk@11`
* **Linux (Ubuntu/Debian):** `sudo apt update && sudo apt install openjdk-11-jre`
* **Windows:** Download and install the OpenJDK from [Adoptium](https://adoptium.net/). Ensure you check the box to "Add to PATH" during installation.

**3. Create Virtual Environment & Install Python Dependencies:**

* **macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

*(Note for macOS Apple Silicon users: Ensure PyTorch is installed with MPS support for hardware acceleration).*

* **Windows:**

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

```

---

## 5. Reproduction Steps

To fully replicate our evaluation pipeline from scratch, follow these steps in order:

**Step 1: Data Preparation**
Ensure the Flickr8k sample dataset is correctly structured within the repository before running any code.

* Place the test images in the `data/flickr8k_sample/images/` directory.
* Place the human reference annotations file at `data/flickr8k_sample/raw_annotations.json`.

**Step 2: Execute Zero-Shot Inference & Metric Evaluation**
Run the core evaluation script. This script will automatically load the pre-trained weights for GIT, BLIP, and BLIP-2 via the Hugging Face library, process the test images to generate predictions, and calculate the COCO metrics (BLEU, METEOR, ROUGE, CIDEr) against the human annotations.

```bash
python src/evaluate_metrics.py

```

*(Note: Depending on your hardware, running inference for all three models—especially the 2.7B parameter BLIP-2—may take considerable time. Output predictions and final scores will be automatically saved to the `results/` directory as JSON files).*

**Step 3: Qualitative Analysis & Visualizations**
To view side-by-side qualitative image comparisons and generate the final performance bar charts, launch Jupyter Notebook:

```bash
jupyter notebook

```

Navigate to the `notebooks/` directory and open `qualitative_analysis.ipynb`. Run all cells to process the generated JSON outputs and reproduce the project's analytical findings and plots.

---

## 6. Results Summary

| Model | BLEU-4 | METEOR | ROUGE-L | CIDEr |
| --- | --- | --- | --- | --- |
| **GIT Base** | 0.137 | 0.164 | 0.374 | 0.426 |
| **BLIP Base** | 0.281 | 0.248 | 0.530 | 0.819 |
| **BLIP-2 (OPT-2.7b)** | **0.331** | **0.291** | **0.581** | **1.062** |

*Conclusion:* BLIP-2 vastly outperforms the baseline architectures, particularly in CIDEr, proving that the Q-Former successfully bridges the modality gap, allowing the LLM to generate highly accurate, human-consensus captions.

---

## 7. References

Banerjee, S., & Lavie, A. (2005). METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. *Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization*.

Chen, X., Fang, H., Lin, T.-Y., Vedantam, R., Gupta, S., Dollár, P., & Zitnick, C. L. (2015). *Microsoft COCO Captions: Data collection and evaluation server*. arXiv:1504.00325.

COCO Evaluation Library (pycocoevalcap). (n.d.). GitHub repository. https://github.com/salaniz/pycocoevalcap

Hodosh, M., Young, P., & Hockenmaier, J. (2013). Framing image description as a ranking task: Data, models and evaluation metrics. *Journal of Artificial Intelligence Research, 47*, 853–899.

Hugging Face BLIP Documentation. (n.d.). https://huggingface.co/docs/transformers/model_doc/blip

Hugging Face BLIP-2 Documentation. (n.d.). https://huggingface.co/docs/transformers/model_doc/blip-2

Hugging Face GIT Documentation. (n.d.). https://huggingface.co/docs/transformers/model_doc/git

Li, J., Li, D., Savarese, S., & Hoi, S. (2023). *BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models*. arXiv:2301.12597.

Li, J., Li, D., Xiong, C., & Hoi, S. (2022). *BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation*. arXiv:2201.12086.

Lin, C.-Y. (2004). ROUGE: A package for automatic evaluation of summaries. *Text Summarization Branches Out*.

Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). BLEU: A method for automatic evaluation of machine translation. *Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics*.

Vedantam, R., Zitnick, C. L., & Parikh, D. (2015). CIDEr: Consensus-based image description evaluation. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

Wang, J., Yang, Z., Hu, X., Li, L., Lin, K., Gan, Z., Liu, Z., Liu, C., & Wang, L. (2022). *GIT: A generative image-to-text transformer for vision and language*. arXiv:2205.14100.

Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., & Brew, J. (2020). Transformers: State-of-the-art natural language processing. *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*.

Zhang, S., Roller, S., Goyal, N., Artetxe, M., Moya, M., Lin, X., ... & Zettlemoyer, L. (2022). *OPT: Open pre-trained transformer language models*. arXiv:2205.01068.
