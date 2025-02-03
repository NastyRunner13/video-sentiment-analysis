import os
import cv2
import numpy as np
import pandas as pd
import subprocess
import torch
import torch.utils.data.dataloader
import torchaudio
from torch.utils.data import Dataset, DataLoader
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
            'negative': 0, 'positive': 1, 'neutral': 2
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
    
    def _extract_audio_features(self, video_path):
        audio_path = video_path.replace(".mp4", ".wav")

        try:
            subprocess.run([
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=1024,
                hop_length=512,
                n_mels=64
            )

            mel_spec = mel_spectrogram(waveform)

            # NORMALIZE
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :300]

            return mel_spec

        except subprocess.CalledProcessError as e:
            raise ValueError(f"Audio Extraction Error: {str(e)}")
            
        except Exception as e:
            raise ValueError(f"Audio Error: {str(e)}")
        
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        row = self.data.iloc[idx]
        
        try:
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

            video_frames = self._load_video_frames(video_path)
            audio_features = self._extract_audio_features(video_path)

            # MAP SENTIMENT AND EMOTION LABELS
            emotion_label = self.emotion_map[row['Emotion'].lower()]
            sentiment_label = self.sentiment_map[row['Sentiment'].lower()]

            return {
                'text_inputs': {
                    'input_ids': text_inputs['input_ids'].squeeze(),
                    'attention_mask': text_inputs['attention_mask'].squeeze()
                },
                'video_frames': video_frames,
                'audio_features': audio_features,
                'emotion_label': torch.tensor(emotion_label),
                'sentiment_label': torch.tensor(sentiment_label)
            }
        except Exception as e:
            print(f"Error: {str(e)}")
            return None
        
def collate_fn(batch):
    # Filter oout None samples
    batch = list(filter(None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def prepare_dataloaders(train_csv, train_video_dir,
                        dev_csv, dev_video_dir,
                        test_csv, test_video_dir, batch_size=32):
    train_dataset = MELDDataset(train_csv, train_video_dir)
    dev_dataset = MELDDataset(dev_csv, dev_video_dir)
    test_dataset = MELDDataset(test_csv, test_video_dir)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)

    dev_loader = DataLoader(dev_dataset,
                            batch_size=batch_size,
                            collate_fn=collate_fn)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader


if __name__ == "__main__":
    train_loader, dev_loader, test_loader = prepare_dataloaders(
        '../dataset/train/train_sent_emo.csv', '../dataset/train/train_splits',
        '../dataset/dev/dev_sent_emo.csv', '../dataset/dev/dev_splits_complete',
        '../dataset/test/test_sent_emo.csv', '../dataset/test/output_repeated_splits_test'
    )

    for batch in train_loader:
        print(batch['text_inputs'])
        print(batch['video_frames'].shape)
        print(batch['audio_features'].shape)
        print(batch['emotion_label'])
        print(batch['sentiment_label'])
        break