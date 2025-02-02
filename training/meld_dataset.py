import os
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir):
        super().__init__()
        
        self.data = pd.read_csv(csv_path)
        self.video_dir = video_dir
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        self.emotion_map = {
            'anger': 0, 'disgust': 1, 'fear': 2, 
            'joy': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6
        }
        self.setiment_map = {
            'positive': 0, 'negative': 1, 'neutral': 2
        }

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        print("hello")

    
if __name__ == "__main__":

    dataset = MELDDataset('./dataset/dev/dev_sent_emo.csv',
                          './dataset/dev/dev_splits_complete')
    