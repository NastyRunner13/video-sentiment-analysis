import os
import cv2
import numpy as np
import pandas as pd
import subprocess
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.save_pretrained("./bert-tokenizer")

class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir):
        super().__init__()
        
        self.data = pd.read_csv(csv_path)
        self.video_dir = video_dir
        
        self.tokenizer = AutoTokenizer.from_pretrained("./bert-tokenizer")
        
        self.emotion_map = {
            'anger': 0, 'disgust': 1, 'fear': 2, 
            'joy': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6
        }
        self.sentiment_map = {
            'positive': 0, 'negative': 1, 'neutral': 2
        }

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            if not cap.isOpened():
                raise ValueError(f"Video Error: Couldn't open video file or stream: {video_path}")
            
            ret, frame = cap.read()

            if not ret or frame is None:
                raise ValueError(f"Video Error: Couldn't read the first frame of video: {video_path}")
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0
                frames.append(frame)

        except Exception as e:
            raise ValueError(f"Video Error: {str(e)}")
        
        finally:
            cap.release()

        if len(frames) == 0:
            raise ValueError(f"Video Error: Couldn't read frames from video: {video_path}")
        
        # PAD OR TRUNCATE FRAMES
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0]) for _ in range(30 - len(frames))]
        else:
            frames = frames[:30]
        
        # BEFORE PERMUTE: [frames, height, width, channels]
        # AFTER PERMUTE: [frames, channels, height, width]
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
        
        video_path = os.path.join(self.video_dir, video_filename)
        
        if not os.path.exists(video_path):  
            raise FileNotFoundError(f"No video found for filename: {video_path}")
        
        text_inputs = self.tokenizer(
            row['Utterance'],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        #video_frames = self._load_video_frames(video_path)
        #print(video_frames)
    
if __name__ == "__main__":

    meld = MELDDataset('./dataset/dev/dev_sent_emo.csv',
                          './dataset/dev/dev_splits_complete')
    print(meld[0])