# 🧠 MLLMEraser 2026

Code for **MLLMEraser: Achieving Test-Time Unlearning in Multimodal Large Language Models through Activation Steering** (NeurIPS 2026).

After installing dependencies and preparing the model and datasets below, run the complete LLaVA experiment with:

```bash
bash llava.sh
```

`llava.sh` contains the full workflow directly. For Qwen, use `bash qwen.sh`.

### ⚙️ **Environment**

- Use **`requirements-llava.txt`** for **LLaVA-1.5-7B**.
- Use **`requirements-qwen.txt`** for **Qwen-2.5-VL-7B**. **`requirements-qwen-flash-attn.txt`** is an optional second step for FlashAttention.

---

### 📦 **Prepare the Data and the Vanilla Models**

Prepare the following models and datasets:

1. **Vanilla models**  
   - **LLaVA-1.5-7B (Vanilla)**  
     https://huggingface.co/oceanoceanna/LLaVA_Vanilla  
   - **Qwen-2.5-VL-7B (Vanilla)**  
     Train this model yourself and save the resulting checkpoint and processor/tokenizer files in `Qwen_Vanilla/`.

2. **Training split used in baseline comparisons**  
   https://huggingface.co/datasets/oceanoceanna/baseline_train_split

3. **Evaluation benchmark (MLLMU-Bench)**  
   https://huggingface.co/datasets/oceanoceanna/MLLMU-Bench

After downloading the datasets, place **MLLMU-Bench** into the project’s `data/` directory:
If the `data/` folder does not exist, create it manually:

```bash
mkdir data
```

### **Model and Data Paths**

Model paths are command-line arguments; no source edits are needed. Each local checkpoint should include its model, tokenizer, and processor files.

```bash
python -m newcode.embed_llava --dataset forget --model_path /path/to/LLaVA_Vanilla
python -m newcode.embed_qwen --dataset retain --model_path /path/to/Qwen_Vanilla
```

The full pipelines accept environment variables:

```bash
CUDA_VISIBLE_DEVICES=0 TRAIN_CUDA_VISIBLE_DEVICES=0,1 \
MODEL_PATH=/path/to/LLaVA_Vanilla DATA_DIR=/path/to/MLLMU-Bench \
TRAIN_DATA_DIR=/path/to/baseline_train_split bash llava.sh
```

---

### 📁 **Final Directory Structure**

After preparing the vanilla models and datasets, the project directory should look like:

```text
PHASE_3/
├── LLaVA_Vanilla/              # Vanilla LLaVA-1.5-7B model
├── Qwen_Vanilla/               # Your trained Qwen-2.5-VL-7B model
├── baseline_train_split/       
├── data/
│   └── MLLMU-Bench/
├── newcode/
├── refusal_direction/
├── llava.sh                   # Complete LLaVA experiment
├── qwen.sh                    # Complete Qwen experiment
├── requirements-llava.txt
├── requirements-qwen.txt
├── requirements-qwen-flash-attn.txt
└── ...
```


This structure ensures that all training, evaluation, and steering scripts can correctly locate the required models and datasets.

---

### 🚀 **Experiments**

#### **1. Run LLaVA Experiments**

```bash
python3.10 -m venv .venv/llava
source .venv/llava/bin/activate
python -m pip install pip==25.1 setuptools==78.1.1 wheel==0.45.1
python -m pip install -r requirements-llava.txt
python -m pip check
```

After installing dependencies and preparing `LLaVA_Vanilla/`, `data/MLLMU-Bench/`, and `baseline_train_split/`, run:

```bash
bash llava.sh
```

#### **2. Run Qwen Experiments** 

```bash
python3.10 -m venv .venv/qwen
source .venv/qwen/bin/activate
python -m pip install pip==25.1 setuptools==78.1.1 wheel==0.45.1 ninja==1.11.1.4
python -m pip install -r requirements-qwen.txt
# Optional, only when using --attn_implementation flash_attention_2:
# MAX_JOBS=4 python -m pip install --no-build-isolation -r requirements-qwen-flash-attn.txt
python -m pip check
```

Install FlashAttention **after** the main Qwen dependencies: its build imports PyTorch. The default scripts use the Transformers attention backend; evaluation can explicitly enable FlashAttention with `--attn_implementation flash_attention_2`. Only that backend requires the extra installation. Follow the [FlashAttention installation requirements](https://github.com/Dao-AILab/flash-attention/tree/v2.7.4.post1#installation-and-features): use a supported NVIDIA GPU and a CUDA toolkit with `nvcc` (CUDA 12.4 matches this environment), plus a C++ compiler if a source build is needed. The CUDA runtime installed by pip does not include the compiler toolchain.

Qwen's `click` pin is changed from 8.2.2 to 8.2.1 because [8.2.2 was yanked from PyPI](https://pypi.org/project/click/8.2.2/) for a regression. Other original Python package versions are preserved.

After installing dependencies and preparing `Qwen_Vanilla/`, `data/MLLMU-Bench/`, and `baseline_train_split/`, run:

```bash
bash qwen.sh
```

Both scripts will perform **test-time unlearning** and evaluate model behavior on:

- **Forget Set**
- **Retain Set**
- **Test Set**
- **Real-world Set**

---

### 📊 **Evaluation Metrics Explanation**

After running the experiments, the output will appear in the following format:

```
************************IMAGE_TEXTUAL RESULT*****************:
#######################CLASSIFICATION RESULTS################:
Forget Set Accuracy: ...
Test Set Accuracy: ...
Retain Set Accuracy: ...
Real Set Accuracy: ...

#######################GENERATION RESULTS########################:
Forget Set ROUGE-L: ...
Test Set ROUGE-L: ...
Retain Set ROUGE-L: ...
Real Set ROUGE-L: ...

#######################CLOZE RESULTS########################:
Forget Set Cloze Accuracy: ...
Test Set Cloze Accuracy: ...
Retain Set Cloze Accuracy: ...
Real Set Cloze Accuracy: ...
```

Below explains what each task measures and which direction is desirable.

------

#### 1. Classification Task

**Measures:**
 Image–text multiple-choice question answering accuracy.

**Performance Direction:**

- **Forget/Test Set** → **Lower is better**
  (model should forget harmful or undesired knowledge)
- **Retain/Real Sets** → **Higher is better**
  (model should preserve general reasoning ability)

------

#### 2. Generation Task (ROUGE-L)

**Measures:**
 Similarity between the model-generated response and the ground-truth answer (longest common subsequence).

**Performance Direction:**

- **Forget/ Test Set ROUGE-L** → **Lower is better**
  (the model should not reproduce forgotten content)
- **Retain/Real ROUGE-L** → **Higher is better**
  (the model should maintain generative quality)

------

#### 3. Cloze Task

**Measures:**
 Fill-in-the-blank accuracy for image–text cloze questions.

**Performance Direction:**

- **Forget/Test Set Accuracy** → **Lower is better**
  (model should no longer recall forgotten information)
- **Retain/Real Sets** → **Higher is better**
  (model should maintain factual and contextual reasoning)
