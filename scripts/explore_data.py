"""
Data exploration notebook for Zalo AI Traffic Dataset
Run this first to understand your dataset structure
"""

import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import pandas as pd

# Paths
TRAIN_JSON = "d:/ZALO_AI/ZaloAI CODE/train/train.json"
TEST_JSON = "d:/ZALO_AI/ZaloAI CODE/public_test/public_test.json"
TRAIN_VIDEOS = "d:/ZALO_AI/ZaloAI CODE/train/videos/"

# Load data
def load_data():
    """Load training and test data"""
    with open(TRAIN_JSON, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    with open(TEST_JSON, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    return train_data, test_data

# Analyze dataset
def analyze_dataset(data, dataset_name="Dataset"):
    """Analyze dataset statistics"""

    print(f"\n{'='*60}")
    print(f"{dataset_name} Analysis")
    print(f"{'='*60}")

    samples = data['data']

    print(f"\n📊 Basic Statistics:")
    print(f"Total samples: {len(samples)}")

    # Question length analysis
    question_lengths = [len(s['question']) for s in samples]
    print(f"\nQuestion Length Statistics:")
    print(f"  Min: {min(question_lengths)} characters")
    print(f"  Max: {max(question_lengths)} characters")
    print(f"  Mean: {np.mean(question_lengths):.1f} characters")
    print(f"  Median: {np.median(question_lengths):.1f} characters")

    # Number of choices analysis
    choice_counts = Counter([len(s['choices']) for s in samples])
    print(f"\nNumber of Choices:")
    for num_choices, count in sorted(choice_counts.items()):
        print(f"  {num_choices} choices: {count} samples ({count/len(samples)*100:.1f}%)")

    # Answer distribution (only for training data)
    if 'answer' in samples[0]:
        answers = [s['answer'][0] for s in samples if s.get('answer')]  # Get first char (A/B/C/D)
        answer_dist = Counter(answers)
        print(f"\nAnswer Distribution:")
        for answer, count in sorted(answer_dist.items()):
            print(f"  {answer}: {count} samples ({count/len(samples)*100:.1f}%)")

    # Support frames analysis (training data)
    if 'support_frames' in samples[0]:
        has_support = sum(1 for s in samples if s.get('support_frames'))
        print(f"\nSupport Frames:")
        print(f"  Samples with support frames: {has_support} ({has_support/len(samples)*100:.1f}%)")

        frame_counts = [len(s.get('support_frames', [])) for s in samples]
        print(f"  Frames per sample (avg): {np.mean(frame_counts):.2f}")

    # Video analysis
    video_paths = [s['video_path'] for s in samples]
    unique_videos = len(set(video_paths))
    print(f"\nVideo Statistics:")
    print(f"  Unique videos: {unique_videos}")
    print(f"  Avg samples per video: {len(samples)/unique_videos:.2f}")

    # Question type analysis
    print(f"\nQuestion Pattern Analysis:")
    patterns = analyze_question_patterns(samples)
    for pattern, count in patterns.most_common(10):
        print(f"  {pattern}: {count}")

    return samples

def analyze_question_patterns(samples):
    """Analyze common question patterns"""
    patterns = Counter()

    for sample in samples:
        question = sample['question'].lower()

        # Common keywords
        if 'rẽ phải' in question:
            patterns['Rẽ phải (Turn right)'] += 1
        if 'rẽ trái' in question:
            patterns['Rẽ trái (Turn left)'] += 1
        if 'đi thẳng' in question:
            patterns['Đi thẳng (Go straight)'] += 1
        if 'biển báo' in question or 'biển chỉ dẫn' in question:
            patterns['Biển báo (Traffic sign)'] += 1
        if 'làn đường' in question or 'làn' in question:
            patterns['Làn đường (Lane)'] += 1
        if 'đúng hay sai' in question or 'đúng' in question:
            patterns['True/False question'] += 1
        if 'hướng' in question:
            patterns['Direction question'] += 1

    return patterns

def analyze_videos(samples, video_root, num_samples=5):
    """Analyze video properties"""

    print(f"\n{'='*60}")
    print(f"Video Analysis (sampling {num_samples} videos)")
    print(f"{'='*60}")

    video_stats = []

    for sample in samples[:num_samples]:
        video_path = os.path.join(video_root, sample['video_path'])

        if not os.path.exists(video_path):
            print(f"⚠️  Video not found: {video_path}")
            continue

        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0

        video_stats.append({
            'video_path': sample['video_path'],
            'fps': fps,
            'frames': frame_count,
            'width': width,
            'height': height,
            'duration': duration
        })

        cap.release()

    if video_stats:
        df = pd.DataFrame(video_stats)
        print(f"\nVideo Statistics:")
        print(f"  Average FPS: {df['fps'].mean():.1f}")
        print(f"  Average Duration: {df['duration'].mean():.2f} seconds")
        print(f"  Average Frames: {df['frames'].mean():.0f}")
        print(f"  Resolution: {df['width'].iloc[0]}x{df['height'].iloc[0]}")

    return video_stats

def visualize_sample(sample, video_root):
    """Visualize a sample with video frames"""

    video_path = os.path.join(video_root, sample['video_path'])

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    print(f"\n{'='*60}")
    print(f"Sample Visualization")
    print(f"{'='*60}")
    print(f"\nID: {sample['id']}")
    print(f"Question: {sample['question']}")
    print(f"Choices:")
    for choice in sample['choices']:
        print(f"  {choice}")
    if 'answer' in sample:
        print(f"Answer: {sample['answer']}")
    if 'support_frames' in sample:
        print(f"Support Frames: {sample.get('support_frames', [])}")

    # Extract frames
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames_to_show = []

    # Get support frame if available
    if sample.get('support_frames'):
        for timestamp in sample['support_frames'][:1]:  # Show first support frame
            frame_num = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                frames_to_show.append(('Support Frame', frame))

    # Get first and last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if ret:
        frames_to_show.append(('First Frame', first_frame))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, last_frame = cap.read()
    if ret:
        frames_to_show.append(('Last Frame', last_frame))

    cap.release()

    # Display frames
    if frames_to_show:
        fig, axes = plt.subplots(1, len(frames_to_show), figsize=(15, 5))
        if len(frames_to_show) == 1:
            axes = [axes]

        for ax, (title, frame) in zip(axes, frames_to_show):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax.imshow(frame_rgb)
            ax.set_title(title)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig('sample_visualization.png', dpi=150, bbox_inches='tight')
        print(f"\n✅ Visualization saved to 'sample_visualization.png'")
        plt.close()

def main():
    """Main exploration function"""

    print("🚀 Starting Zalo AI Traffic Dataset Exploration\n")

    # Load data
    train_data, test_data = load_data()

    # Analyze training data
    train_samples = analyze_dataset(train_data, "Training Data")

    # Analyze test data
    test_samples = analyze_dataset(test_data, "Public Test Data")

    # Analyze videos
    if os.path.exists(TRAIN_VIDEOS):
        video_stats = analyze_videos(train_samples,
                                     "d:/ZALO_AI/ZaloAI CODE/",
                                     num_samples=10)

    # Visualize sample
    print("\n" + "="*60)
    print("Generating Sample Visualization...")
    print("="*60)
    visualize_sample(train_samples[0], "d:/ZALO_AI/ZaloAI CODE/")

    # Save summary report
    summary = {
        'train_count': len(train_samples),
        'test_count': len(test_samples),
        'unique_videos_train': len(set(s['video_path'] for s in train_samples)),
        'unique_videos_test': len(set(s['video_path'] for s in test_samples)),
    }

    with open('dataset_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n✅ Exploration complete! Summary saved to 'dataset_summary.json'")

if __name__ == "__main__":
    main()
