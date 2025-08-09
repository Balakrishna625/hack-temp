---
library_name: peft
base_model:
- unsloth/Llama-3.2-11B-Vision-Instruct
datasets:
- eltorio/ROCOv2-radiology
---

# Model Card – Llama-3.2 11B Vision for Medical Imaging

<img src="https://i5.walmartimages.com/seo/DolliBu-Beige-Llama-Doctor-Plush-Toy-Super-Soft-Stuffed-Animal-Dress-Up-Cute-Scrub-Uniform-Cap-Outfit-Fluffy-Gift-11-Inches_e78392b2-71ef-4e26-a23f-8bb0b0e2043a.70c3b5988d390cf43d799758a826f2a5.jpeg" alt="Llama Doctor Plush" width="400"/>

<font color="FF0000" size="5"><b>
A vision–language model fine-tuned for interpreting radiographic images</b></font><br>

**Base Model:** [unsloth/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/unsloth/Llama-3.2-11B-Vision-Instruct)  
**Dataset:** [ROCOv2-radiology](https://huggingface.co/datasets/eltorio/ROCOv2-radiology)  

The model was trained using **Google Colab Pro with an NVIDIA A100 GPU** and later pushed to the Hugging Face Hub for public access.

---

##  Model Overview

Built on top of *unsloth/Llama-3.2-11B-Vision-Instruct*, this version has been fine-tuned using **Supervised Fine-Tuning (SFT)** and **PEFT LoRA** for efficient adaptation to vision–language medical tasks.  
It is optimized for radiology image captioning and diagnostic-style descriptions.

### Core Libraries
- `unsloth`
- `transformers`
- `torch`
- `datasets`
- `trl`
- `peft`

---

##  Bias, Risks & Limitations

The fine-tuning process used **only 1/7th** of the ROCOv2 dataset to reduce compute requirements.  
<font color="FF0000">
Outputs should be treated as **assistive** rather than authoritative.  
Medical professionals must review any AI-generated interpretation before it is acted upon.  
Model predictions may be incorrect or misleading if the input data differs significantly from the training set.
</font>

---

##  Training Configuration

**Training Parameters**
- `per_device_train_batch_size`: 2
- `gradient_accumulation_steps`: 16
- `num_train_epochs`: 3
- `learning_rate`: 5e-5
- `weight_decay`: 0.02
- `lr_scheduler_type`: "linear"
- `max_seq_length`: 2048

**LoRA Settings**
- `r`: 32
- `lora_alpha`: 32
- `lora_dropout`: 0
- `bias`: "none"

**Hardware**
- Platform: Google Colab Pro  
- GPU: NVIDIA A100 (40GB) CUDA-enabled

**Training Stats**
- Duration: ~40,989 seconds (~683 minutes)
- Peak reserved memory: 12.8 GB (32.3% of max)
- Peak memory for training: 3.975 GB (10.1% of max)

---

##  Training Data

Dataset: **ROCOv2-radiology** – a collection of radiographic images paired with medical descriptions.  
Only a subset (2/7th) was used for computational efficiency.

---

##  Usage Example

```python
instruction = "You are an expert radiographer. Provide an accurate and detailed description of the medical image."
```

---

## Model Availability

Model on Hugging Face: [Bala9669/unsloth_finetune](https://huggingface.co/Bala9669/unsloth_finetune)  
