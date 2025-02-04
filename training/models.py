import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models
from meld_dataset import MELDDataset

import warnings
# Suppress FutureWarnings and UserWarnings that match specific messages
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        for param in self.bert.parameters():
            param.requires_grad = False

        self.projection = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask):
        # EXTRACT BERT EMBEDDINGS
        outputs = self.bert(input_ids, attention_mask)
        # USE CLS TOKEN REPRESSENTATION
        pooler_output = outputs.pooler_output

        return self.projection(pooler_output)
    
class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = vision_models.video.r3d_18(pretrained=True)

        for param in self.backbone.parameters():
            param.requires_grad = False
        
        num_fts = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_fts, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.backbone(x)
    
class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(64, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        for param in self.conv_layers.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = x.squeeze(1)
        features = self.conv_layers(x)
        return self.projection(features.squeeze(-1))
    
class MultimodalTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7)
        )

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )

    def forward(self, text_inputs, video_frames, audio_features):
        text_features = self.text_encoder(
            text_inputs['input_ids'],
            text_inputs['attention_mask']
        )
        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)

        combined_features = torch.cat(
            [text_features,
            video_features,
            audio_features],
            dim=1
        )

        fused_features = self.fusion_layer(combined_features)

        emotion_output = self.emotion_classifier(fused_features)
        sentiment_output = self.sentiment_classifier(fused_features)

        return {
            'emotions': emotion_output,
            'sentiments': sentiment_output 
        }

if __name__ == "__main__":
    dataset = MELDDataset(
        '../dataset/train/train_sent_emo.csv',
        '../dataset/train/train_splits'
    )
    sample = dataset[0]

    model = MultimodalTransformer()
    model.eval()

    text_inputs = {
        'input_ids': sample['text_inputs']['input_ids'].unsqueeze(0),
        'attention_mask': sample['text_inputs']['attention_mask'].unsqueeze(0)
    }

    video_frames = sample['video_frames'].unsqueeze(0)
    audio_features = sample['audio_features'].unsqueeze(0)

    with torch.inference_mode():
        outputs = model(text_inputs, video_frames, audio_features)
        emotion_probs = torch.softmax(outputs['emotions'], dim=-1)[0]
        sentiment_probs = torch.softmax(outputs['sentiments'], dim=-1)[0]

    emotion_map = {
        0: 'anger', 1: 'disgust', 2: 'fear', 
        3: 'joy', 4: 'neutral', 5: 'sadness', 6: 'surprise'
    }
    sentiment_map = {
        0: 'negative', 1: 'positive', 2: 'neutral'
    }

    for i, prob in enumerate(emotion_probs):
        print(f"{emotion_map[i]}: {prob.item():.4f}")

    for i, prob in enumerate(sentiment_probs):
        print(f"{sentiment_map[i]}: {prob.item():.4f}")

    print("Predictions for utterence")