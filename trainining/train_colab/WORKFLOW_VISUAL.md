# 🎨 Colab Notebook Workflow

```
╔═══════════════════════════════════════════════════════════════════╗
║         OPTIMIZED VIDEO QA - SINGLE COLAB NOTEBOOK                ║
╚═══════════════════════════════════════════════════════════════════╝

┌───────────────────────────────────────────────────────────────────┐
│ STEP 1: Setup & Configuration                                    │
├───────────────────────────────────────────────────────────────────┤
│ Cell 1:  Mount Google Drive                                      │
│          ↓                                                        │
│ Cell 2:  Install dependencies                                    │
│          • transformers, qwen-vl-utils                           │
│          • flash-attn, opencv-python                             │
│          ↓                                                        │
│ Cell 3:  Configuration                                           │
│          • DATA_ROOT path                                        │
│          • BATCH_SIZE = 4                                        │
│          • NUM_WORKERS = 8                                       │
│          ↓                                                        │
│ Cell 4:  Load utility functions                                 │
└───────────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────┐
│ PART 1: Preprocessing (Cells 5-7)                                │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│ Cell 5:  Load preprocessing functions                            │
│          • extract_and_cache_frames()                            │
│          • preprocess_dataset()                                  │
│          • create_index_file()                                   │
│          ↓                                                        │
│ Cell 6:  Run preprocessing ⚡                                     │
│          ┌─────────────────────────────────┐                     │
│          │ Parallel Processing (8 workers) │                     │
│          │                                 │                     │
│          │  Worker 1 → Video 1, 9, 17...  │                     │
│          │  Worker 2 → Video 2, 10, 18... │                     │
│          │  Worker 3 → Video 3, 11, 19... │                     │
│          │  Worker 4 → Video 4, 12, 20... │                     │
│          │  Worker 5 → Video 5, 13, 21... │                     │
│          │  Worker 6 → Video 6, 14, 22... │                     │
│          │  Worker 7 → Video 7, 15, 23... │                     │
│          │  Worker 8 → Video 8, 16, 24... │                     │
│          └─────────────────────────────────┘                     │
│                      ↓                                            │
│          Each worker:                                            │
│            1. Read video                                         │
│            2. Extract frames at timestamps                       │
│            3. Convert to PIL Images                              │
│            4. Save to Google Drive cache                         │
│                      ↓                                            │
│          Output:                                                 │
│            • Cached frames (.pkl files)                          │
│            • frames_index.json                                   │
│                                                                   │
│ 💾 Cache Location:                                               │
│    /content/drive/MyDrive/zalo_ai_cache/extracted_frames/        │
│                                                                   │
│ ⏱️  Time: ~2-5 minutes for 1490 videos                          │
└───────────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────┐
│ PART 2: Model Inference (Cells 8-11)                             │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│ Cell 8:  Load inference functions                                │
│          • CachedFramesDataset class                             │
│          • load_model_and_processor()                            │
│          • inference_batch()                                     │
│          ↓                                                        │
│ Cell 9:  Load Qwen3-VL-4B model                                  │
│          • HuggingFace login                                     │
│          • Load to GPU (A100/V100)                               │
│          • Enable Flash Attention 2                              │
│          ↓                                                        │
│ Cell 10: Run batch inference 🚀                                  │
│          ┌─────────────────────────────────┐                     │
│          │ Batch Processing (size=4)       │                     │
│          │                                 │                     │
│          │ Batch 1: Q1, Q2, Q3, Q4        │                     │
│          │   ↓                             │                     │
│          │ Load frames from cache          │                     │
│          │   ↓                             │                     │
│          │ Create prompts                  │                     │
│          │   ↓                             │                     │
│          │ Process in parallel on GPU      │                     │
│          │   ↓                             │                     │
│          │ Get 4 answers                   │                     │
│          │                                 │                     │
│          │ Batch 2: Q5, Q6, Q7, Q8        │                     │
│          │   ... (repeat)                  │                     │
│          └─────────────────────────────────┘                     │
│                      ↓                                            │
│          For each answer:                                        │
│            • Parse QUAN SÁT (observation)                        │
│            • Parse SUY LUẬN (reasoning)                          │
│            • Parse ĐÁP ÁN (A/B/C/D)                              │
│            • Compare with ground truth                           │
│                      ↓                                            │
│          Calculate accuracy                                      │
│                      ↓                                            │
│ Cell 11: Save results                                            │
│          → /content/drive/MyDrive/zalo_ai_cache/results/         │
│                                                                   │
│ ⏱️  Time: ~15-30 minutes for 1490 questions                     │
└───────────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────┐
│ Results Analysis (Cell 12)                                       │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│ • Total accuracy                                                 │
│ • Show incorrect answers                                         │
│ • Detailed error analysis                                        │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘


╔═══════════════════════════════════════════════════════════════════╗
║                        DATA FLOW                                  ║
╚═══════════════════════════════════════════════════════════════════╝

Google Drive:
   traffic_buddy_train+public_test/train/
      ↓
   Load train.json
      ↓
   Extract video paths & timestamps
      ↓
┌─────────────────────────────────────┐
│ PREPROCESSING (Parallel)            │
│   Video 1 → Frames → Cache          │
│   Video 2 → Frames → Cache          │
│   ...                                │
│   Video N → Frames → Cache          │
└─────────────────────────────────────┘
      ↓
   frames_index.json created
      ↓
┌─────────────────────────────────────┐
│ INFERENCE (Batch GPU)               │
│   Load cached frames                │
│   ↓                                 │
│   Process batch → Answers           │
│   ↓                                 │
│   Compare with ground truth         │
└─────────────────────────────────────┘
      ↓
   inference_results.json saved
      ↓
   Done! ✅


╔═══════════════════════════════════════════════════════════════════╗
║                    RESOURCE USAGE                                 ║
╚═══════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────┐
│ PART 1: Preprocessing                                          │
├─────────────────────────────────────────────────────────────────┤
│ CPU:     8 cores @ 100% (parallel workers)                     │
│ RAM:     4-8 GB                                                │
│ Storage: ~2-5 GB (cached frames)                               │
│ GPU:     Not used                                              │
│ Time:    ~2-5 minutes                                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ PART 2: Inference                                              │
├─────────────────────────────────────────────────────────────────┤
│ CPU:     Minimal (data loading)                                │
│ RAM:     8-16 GB                                               │
│ GPU:     12-20 GB VRAM (A100)                                  │
│          • Model: ~8 GB                                        │
│          • Batch processing: ~4-12 GB                          │
│ Time:    ~15-30 minutes                                        │
└─────────────────────────────────────────────────────────────────┘


╔═══════════════════════════════════════════════════════════════════╗
║                  KEY ADVANTAGES                                   ║
╚═══════════════════════════════════════════════════════════════════╝

✅ Single notebook - easy to use
✅ Parallel processing - 8x faster frame extraction
✅ Caching - reuse frames across runs
✅ Batch inference - 3-4x faster inference
✅ Google Drive storage - persistent cache
✅ Progress tracking - see what's happening
✅ Error handling - robust processing
✅ Auto-resume - skip cached frames

Total speedup: 5-8x faster than sequential! 🚀
```
