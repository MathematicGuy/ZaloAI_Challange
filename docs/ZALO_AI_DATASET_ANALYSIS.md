# Zalo AI Traffic Dataset Analysis & Tailored Implementation Plan

## 📊 Dataset Overview

### Dataset Statistics
- **Training Set**: 1,490 samples
- **Public Test Set**: 405 samples
- **Language**: Vietnamese (Tiếng Việt)
- **Format**: Video-based multiple choice questions
- **Video Count**: ~550 unique videos (based on video file count)

### Data Structure

#### Training Data Format
```json
{
    "id": "train_0001",
    "question": "Nếu xe ô tô đang chạy ở làn ngoài cùng bên phải trong video này thì xe đó chỉ được phép rẽ phải?",
    "choices": [ 
        "A. Đúng",
        "B. Sai"
    ],
    "answer": "B. Sai",
    "support_frames": [4.427402],
    "video_path": "train/videos/2b840c67_386_clip_002_0008_0018_Y.mp4"
}
```

#### Public Test Data Format
```json
{
    "id": "testa_0001",
    "question": "Theo trong video, nếu ô tô đi hướng chếch sang phải là hướng vào đường nào?",
    "choices": [
        "A. Không có thông tin",
        "B. Dầu Giây Long Thành",
        "C. Đường Đỗ Xuân Hợp",
        "D. Xa Lộ Hà Nội"
    ],
    "video_path": "public_test/videos/efc9909e_908_clip_001_0000_0009_Y.mp4"
}
```

**Key Difference**: Training data has `answer` and `support_frames`, test data doesn't.

### Question Types Analysis

Based on the sample questions, your dataset includes:

1. **Lane/Direction Questions**
   - "Làn đường xe đang chạy có được phép rẽ trái hay không?"
   - "Nếu xe ô tô đang chạy ở làn ngoài cùng bên phải..."

2. **Traffic Sign Recognition**
   - "Biển chỉ dẫn 3 hướng di chuyển chính tiếp theo, đúng hay sai?"
   - "Theo biển báo trong video, muốn đi đến đường Lương Đình Của thì phải đi hướng nào?"

3. **Navigation/Route Questions**
   - "Muốn đi Cầu Rạch Chiếc thì cần rẽ phải, đúng hay sai?"
   - Street names and direction questions

4. **True/False Questions**
   - Binary choice (Đúng/Sai)

5. **Multiple Choice Questions**
   - 2-4 options (A, B, C, D)

### Video Characteristics

**Naming Convention**: `{hash}_{id}_clip_{clip_num}_{start}_{end}_{label}.mp4`
- Example: `2b840c67_386_clip_002_0008_0018_Y.mp4`
- `Y` likely indicates positive/yes label
- `N` likely indicates negative/no label

**Support Frames**:
- Timestamps indicating key frames for answering questions
- Single timestamp per question in training data

---

## 🎯 Tailored Implementation Plan for Zalo AI Dataset

### Phase 1: Data Preparation & Understanding (Week 1)

#### 1.1 Dataset Exploration
```python
# Create data exploration notebook
import json
import cv2
from collections import Counter

# Load and analyze data
with open('train/train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# Statistics to gather:
# - Question length distribution
# - Answer distribution (A, B, C, D)
# - Video duration analysis
# - Support frame timestamp patterns
# - Question type categorization
```

#### 1.2 Video Processing Pipeline
```python
# Video preprocessing for Vietnamese traffic videos
def process_video(video_path, support_frames=None):
    """
    Extract frames from video, focus on support frames

    Args:
        video_path: Path to video file
        support_frames: List of timestamps (in seconds)

    Returns:
        frames: List of key frames as numpy arrays
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []

    if support_frames:
        # Extract frames at specific timestamps
        for timestamp in support_frames:
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
    else:
        # Uniform sampling (e.g., 8 frames)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, 8, dtype=int)

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

    cap.release()
    return frames
```

#### 1.3 Vietnamese Language Considerations
```python
# Install Vietnamese language support
# pip install underthesea  # Vietnamese NLP toolkit
# pip install pyvi  # Vietnamese word segmentation

from underthesea import word_tokenize

# Example Vietnamese text processing
def preprocess_vietnamese_text(text):
    """
    Preprocess Vietnamese traffic questions
    """
    # Word tokenization for Vietnamese
    tokens = word_tokenize(text, format="text")
    return tokens
```

### Phase 2: Model Selection & Setup (Week 2)

#### 2.1 Base Model Options

**Option 1: Qwen2-VL (Recommended for Vietnamese)**
```bash
# Qwen2-VL has good multilingual support including Vietnamese
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model_name = "Qwen/Qwen2-VL-7B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)
```

**Option 2: ViLT + Vietnamese LLM**
```python
# Combine vision model with Vietnamese language model
# ViLT for vision + PhoBERT/viBERT for Vietnamese language understanding
```

#### 2.2 Data Formatting for Training
```python
def format_training_sample(sample):
    """
    Format Zalo AI sample for model training
    """
    question = sample['question']
    choices = '\n'.join(sample['choices'])
    answer = sample['answer']

    # Create instruction format
    instruction = f"""Bạn là một trợ lý AI chuyên về giao thông Việt Nam.
Hãy xem video và trả lời câu hỏi sau:

Câu hỏi: {question}

Các lựa chọn:
{choices}

Hãy phân tích từng bước và chọn đáp án đúng."""

    response = f"""Phân tích:
[Mô tả các yếu tố quan trọng trong video]

Đáp án: {answer}"""

    return {
        'instruction': instruction,
        'video_path': sample['video_path'],
        'support_frames': sample.get('support_frames', []),
        'response': response
    }
```

### Phase 3: Fine-tuning Strategy (Weeks 3-4)

#### 3.1 LoRA Configuration for Vietnamese Traffic
```python
from peft import LoraConfig, get_peft_model

# LoRA configuration optimized for your dataset
lora_config = LoraConfig(
    r=16,                      # Rank
    lora_alpha=32,             # Scaling factor
    target_modules=[           # Target attention layers
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to model
model = get_peft_model(base_model, lora_config)
```

#### 3.2 Training Configuration
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./traffic-mllm-vietnamese",
    num_train_epochs=5,                    # Adjusted for 1,490 samples
    per_device_train_batch_size=2,         # Small batch for video
    gradient_accumulation_steps=8,         # Effective batch = 16
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    evaluation_strategy="steps",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,                             # Mixed precision training
    dataloader_num_workers=4,
    remove_unused_columns=False,
)
```

#### 3.3 Custom Dataset Class
```python
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class ZaloTrafficDataset(Dataset):
    def __init__(self, json_path, video_root, processor, max_frames=8):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.samples = data['data']
        self.video_root = video_root
        self.processor = processor
        self.max_frames = max_frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load video frames
        video_path = os.path.join(self.video_root, sample['video_path'])
        frames = self.extract_frames(
            video_path,
            sample.get('support_frames', [])
        )

        # Format instruction
        instruction = self.format_instruction(sample)

        # Process with model processor
        inputs = self.processor(
            text=instruction,
            images=frames,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Add labels if training data
        if 'answer' in sample:
            inputs['labels'] = self.processor(
                text=sample['answer'],
                return_tensors="pt",
                padding=True,
                truncation=True
            )['input_ids']

        return inputs

    def extract_frames(self, video_path, support_frames):
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []

        if support_frames and len(support_frames) > 0:
            # Use support frames + context frames
            for timestamp in support_frames[:3]:  # Max 3 support frames
                frame_num = int(timestamp * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)

            # Add context frames
            remaining = self.max_frames - len(frames)
            if remaining > 0:
                indices = np.linspace(0, total_frames-1, remaining, dtype=int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)
        else:
            # Uniform sampling
            indices = np.linspace(0, total_frames-1, self.max_frames, dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)

        cap.release()
        return frames[:self.max_frames]

    def format_instruction(self, sample):
        """Format Vietnamese traffic question"""
        question = sample['question']
        choices = '\n'.join(sample['choices'])

        instruction = f"""Bạn là trợ lý AI chuyên về giao thông Việt Nam. Xem video và trả lời câu hỏi.

Câu hỏi: {question}

Lựa chọn:
{choices}

Trả lời:"""
        return instruction
```

### Phase 4: Vietnamese Traffic Knowledge Base (Week 5)

#### 4.1 Build Traffic Rules Knowledge Base
```python
# Create Vietnamese traffic rules knowledge base
vietnamese_traffic_rules = {
    "lane_rules": [
        "Làn đường bên phải nhất có thể đi thẳng hoặc rẽ phải",
        "Làn đường giữa chỉ được đi thẳng",
        "Làn đường bên trái có thể đi thẳng hoặc rẽ trái",
        "Không được đổi làn khi có vạch liền",
    ],
    "traffic_signs": [
        "Biển báo cấm rẽ trái: không được rẽ trái",
        "Biển báo cấm rẽ phải: không được rẽ phải",
        "Biển chỉ dẫn hướng đi: chỉ các hướng được phép",
        "Biển cảnh báo: cảnh báo nguy hiểm phía trước",
    ],
    "road_markings": [
        "Vạch liền: không được vượt hoặc đổi làn",
        "Vạch đứt: được phép đổi làn",
        "Mũi tên chỉ hướng: chỉ hướng đi bắt buộc của làn",
    ]
}

# Save as JSON for RAG
import json
with open('knowledge_base/vietnamese_traffic_rules.json', 'w', encoding='utf-8') as f:
    json.dump(vietnamese_traffic_rules, f, ensure_ascii=False, indent=2)
```

#### 4.2 RAG Implementation for Vietnamese
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Use Vietnamese-friendly embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="keepitreal/vietnamese-sbert",  # Vietnamese sentence embeddings
    model_kwargs={'device': 'cuda'}
)

# Load and process traffic rules
def build_knowledge_base():
    """Build Vietnamese traffic knowledge base"""

    # Load rules
    with open('knowledge_base/vietnamese_traffic_rules.json', 'r', encoding='utf-8') as f:
        rules = json.load(f)

    # Flatten rules into text documents
    documents = []
    for category, rule_list in rules.items():
        for rule in rule_list:
            documents.append(rule)

    # Create vector store
    vectorstore = FAISS.from_texts(
        documents,
        embeddings,
        metadatas=[{"category": cat} for cat in rules.keys() for _ in rules[cat]]
    )

    return vectorstore

# Retrieve relevant rules
def retrieve_rules(question, vectorstore, k=3):
    """Retrieve relevant traffic rules for question"""
    docs = vectorstore.similarity_search(question, k=k)
    return "\n".join([doc.page_content for doc in docs])
```

### Phase 5: Chain-of-Thought for Vietnamese (Week 6)

#### 5.1 Vietnamese CoT Prompt Template
```python
COT_TEMPLATE_VIETNAMESE = """Bạn là chuyên gia giao thông Việt Nam. Phân tích video và trả lời theo các bước:

Video: {video_description}
Quy tắc liên quan: {retrieved_rules}

Câu hỏi: {question}
Lựa chọn: {choices}

Hãy suy nghĩ từng bước:

Bước 1 - Phân tích cảnh quay:
- Mô tả những gì thấy trong video
- Xác định các biển báo, làn đường, phương tiện

Bước 2 - Áp dụng quy tắc giao thông:
- Quy tắc nào liên quan đến tình huống này?
- Quy định của luật giao thông Việt Nam

Bước 3 - Đưa ra kết luận:
- Phân tích từng lựa chọn
- Chọn đáp án đúng và giải thích

Đáp án cuối cùng: [A/B/C/D]
"""

def format_cot_prompt(sample, retrieved_rules=""):
    """Format Chain-of-Thought prompt for Vietnamese"""
    return COT_TEMPLATE_VIETNAMESE.format(
        video_description="[Mô tả từ model vision]",
        retrieved_rules=retrieved_rules,
        question=sample['question'],
        choices='\n'.join(sample['choices'])
    )
```

### Phase 6: Training & Evaluation (Weeks 7-8)

#### 6.1 Training Script
```python
# train.py
import torch
from transformers import Trainer
from peft import prepare_model_for_kbit_training

# Load base model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Prepare for training
model = prepare_model_for_kbit_training(model)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Load datasets
train_dataset = ZaloTrafficDataset(
    'train/train.json',
    'train/videos',
    processor
)

# 80-20 split for train/val
from sklearn.model_selection import train_test_split
train_indices, val_indices = train_test_split(
    range(len(train_dataset)),
    test_size=0.2,
    random_state=42
)

train_subset = torch.utils.data.Subset(train_dataset, train_indices)
val_subset = torch.utils.data.Subset(train_dataset, val_indices)

# Define metrics
def compute_metrics(eval_pred):
    """Compute accuracy for multiple choice"""
    predictions, labels = eval_pred
    # Extract predicted answer (A, B, C, or D)
    # Compare with ground truth
    # Return accuracy
    pass

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_subset,
    eval_dataset=val_subset,
    compute_metrics=compute_metrics,
)

trainer.train()
```

#### 6.2 Evaluation on Public Test
```python
# evaluate.py
def evaluate_on_public_test(model, processor, test_json_path):
    """Evaluate model on Zalo AI public test set"""

    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    predictions = []

    for sample in test_data['data']:
        # Load video
        frames = extract_frames(sample['video_path'])

        # Format prompt
        prompt = format_instruction(sample)

        # Generate prediction
        inputs = processor(text=prompt, images=frames, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=100)
        answer = processor.decode(outputs[0], skip_special_tokens=True)

        # Extract choice (A, B, C, or D)
        predicted_choice = extract_choice(answer)

        predictions.append({
            'id': sample['id'],
            'predicted_answer': predicted_choice
        })

    # Save predictions
    save_predictions(predictions, 'predictions.json')

    return predictions
```

#### 6.3 Submission Format
```python
def create_submission(predictions):
    """Create submission file for Zalo AI"""
    submission = []

    for pred in predictions:
        submission.append({
            'id': pred['id'],
            'answer': pred['predicted_answer']  # e.g., "A", "B", "C", "D"
        })

    with open('submission.csv', 'w', encoding='utf-8') as f:
        f.write('id,answer\n')
        for item in submission:
            f.write(f"{item['id']},{item['answer']}\n")
```

---

## 📁 Tailored Project Structure

```
zalo-traffic-mllm/
├── config/
│   ├── model_config.yaml
│   └── training_config.yaml
├── data/
│   ├── train/
│   │   ├── train.json              # Your 1,490 training samples
│   │   └── videos/                 # Training videos
│   ├── public_test/
│   │   ├── public_test.json        # Your 405 test samples
│   │   └── videos/                 # Test videos
│   └── knowledge_base/
│       └── vietnamese_traffic_rules.json
├── src/
│   ├── data/
│   │   ├── dataset.py              # ZaloTrafficDataset class
│   │   ├── video_processor.py      # Frame extraction
│   │   └── vietnamese_processor.py # Vietnamese text processing
│   ├── model/
│   │   ├── base_model.py           # Load Qwen2-VL
│   │   ├── lora_config.py          # LoRA setup
│   │   └── traffic_mllm.py         # Main model
│   ├── rag/
│   │   ├── knowledge_base.py       # Vietnamese traffic rules
│   │   └── retriever.py            # RAG implementation
│   ├── training/
│   │   ├── trainer.py              # Training loop
│   │   ├── metrics.py              # Evaluation metrics
│   │   └── utils.py                # Helper functions
│   └── inference/
│       ├── predictor.py            # Prediction pipeline
│       └── submission.py           # Create submission file
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Analyze your dataset
│   ├── 02_video_analysis.ipynb     # Video processing tests
│   ├── 03_model_testing.ipynb      # Test base model
│   └── 04_evaluation.ipynb         # Evaluate results
├── scripts/
│   ├── train.sh                    # Training script
│   ├── evaluate.sh                 # Evaluation script
│   └── predict.sh                  # Inference script
├── knowledge_base/
│   └── vietnamese_traffic_rules.json
├── requirements.txt
└── README.md
```

---

## 🎯 Simplified MVP for Your Dataset

### Option 1: Quick Start (2-3 weeks)

**Use Pre-trained Qwen2-VL with Few-Shot Learning**

```python
# No fine-tuning, just few-shot prompting
def few_shot_inference(model, processor, sample, examples):
    """
    Use few-shot learning with your training examples
    """

    # Format few-shot examples
    few_shot_prompt = "Đây là một số ví dụ:\n\n"

    for ex in examples[:3]:  # Use 3 examples
        few_shot_prompt += f"""
Video: [Mô tả video]
Câu hỏi: {ex['question']}
Lựa chọn: {', '.join(ex['choices'])}
Đáp án: {ex['answer']}

"""

    # Add current question
    few_shot_prompt += f"""
Bây giờ hãy trả lời câu hỏi mới:
Video: [Mô tả từ video hiện tại]
Câu hỏi: {sample['question']}
Lựa chọn: {', '.join(sample['choices'])}
Đáp án:
"""

    # Generate answer
    # ... inference code
```

### Option 2: Fine-tuning Approach (6-8 weeks)

1. **Week 1-2**: Data preparation + exploration
2. **Week 3-4**: LoRA fine-tuning
3. **Week 5**: RAG knowledge base
4. **Week 6**: CoT integration
5. **Week 7-8**: Evaluation + optimization

---

## 💡 Key Insights for Your Dataset

### 1. Support Frames are Gold
- Your training data has `support_frames` timestamps
- These indicate KEY moments for answering
- **Priority**: Extract frames at these timestamps first

### 2. Vietnamese Language Handling
- Use Vietnamese-friendly models (Qwen2-VL has good multilingual support)
- Consider Vietnamese embeddings for RAG
- Vietnamese traffic terminology is domain-specific

### 3. Multiple Choice Format
- Output needs to be exact: "A. Đúng", "B. Sai", etc.
- Can simplify to just letter extraction: "A", "B", "C", "D"

### 4. Dataset Size Considerations
- 1,490 samples is moderate (not too small, not too large)
- Perfect for LoRA fine-tuning (don't need full fine-tuning)
- Can do 80-20 train/val split (1,192 train / 298 val)

---

## 📊 Expected Performance Targets

| Metric | Baseline | Target | Stretch Goal |
|--------|----------|--------|--------------|
| Accuracy | 40-50% | 70-75% | 80%+ |
| Training Time | - | 4-6 hours | 2-3 hours |
| Inference Speed | - | <3s/video | <1s/video |

---

## 🚀 Quick Start Commands

```bash
# 1. Setup environment
conda create -n zalo-traffic python=3.10
conda activate zalo-traffic
pip install -r requirements.txt

# 2. Explore data
python scripts/explore_data.py

# 3. Test base model
python scripts/test_base_model.py

# 4. Start training
python scripts/train.py --config config/training_config.yaml

# 5. Evaluate
python scripts/evaluate.py --checkpoint ./best_model

# 6. Create submission
python scripts/create_submission.py --output submission.csv
```

---

## 🎓 Next Steps

1. **Immediate**: Run data exploration notebook
2. **This Week**: Set up base model and test on few samples
3. **Next Week**: Start LoRA fine-tuning
4. **Following Weeks**: Add RAG + CoT, optimize performance

Would you like me to generate starter code for any of these components?
