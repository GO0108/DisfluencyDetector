from torch import nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
class Wav2Vec2FeatureExtractor(torch.nn.Module):
    def __init__(
        self,
        feat_layer=24,
        bundle=torchaudio.pipelines.WAV2VEC2_XLSR53,
        device=DEVICE
    ):
        super().__init__()
        self.sample_rate = bundle.sample_rate
        self.model = bundle.get_model().to(DEVICE)
        self.feat_layer = feat_layer
        self.device = device


    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.squeeze(1)
        features, _ = self.model.extract_features(waveform.to(self.device), num_layers=self.feat_layer)
        return features[-1]
'''

class Wav2Vec2FeatureExtractor(nn.Module):
    def __init__(self, freeze=True):
        super(Wav2Vec2FeatureExtractor, self).__init__()
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.freeze=freeze 

    def forward(self, waveform):
        
        waveform = waveform.squeeze(1)

        if self.freeze:
            with torch.no_grad():
                x = self.model(waveform).last_hidden_state 
        else:
            x = self.model(waveform).last_hidden_state
        return x


class CNNExtractor(nn.Module):
    def __init__(self, input_dim=768, num_channels=50, kernel_size=2):
        super(CNNExtractor, self).__init__()
        self.conv = nn.Conv1d(input_dim, num_channels, kernel_size)

    def forward(self, x):
        x = x.permute(0, 2, 1) #batch_size, feature, seq_lenght
        x = self.conv(x)
        return x


class CustomAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(CustomAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Final linear layer for the output
        self.linear = nn.Linear(d_model, d_model)
        
        # Attention dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query, key, value, mask=None, key_padding_mask=None):
        batch_size = query.shape[0]
        
        # Project inputs to query, key, value
        
        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)
        n = self.k_linear.in_features

        # Reshape query, key, value for multi-head attention
        query = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention scores
        energy = torch.matmul(math.log(n)*query, key.permute(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            energy = energy.masked_fill(key_padding_mask == 0, float('-inf'))
        
        attention = F.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        out = torch.matmul(attention, value)
        
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model) # Reshape and concatenate heads
        out = self.linear(out)
        
        return out

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = CustomAttention(d_model, n_heads)
        self.linear1 = nn.Linear(d_model, 2048)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(2048, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

        
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=50, num_heads=10, num_layers=2, custom_att=True):
        super(TransformerEncoder, self).__init__()
        if custom_att:
            self.encoder_layer = CustomTransformerEncoderLayer(d_model=input_dim, n_heads=num_heads)
        else:
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = x.permute(2, 0, 1) # (seq_length-1, batch_size, 50)
        return self.transformer_encoder(x)

class DisfluencyModel(nn.Module):
    def __init__(self):
        super(DisfluencyModel, self).__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor()
        self.cnn_extractor = CNNExtractor()
        self.transformer = TransformerEncoder()
        self.fc = nn.Linear(50, 2)
        
    def forward(self, audio_input):
        features = self.feature_extractor(audio_input)
        local_features = self.cnn_extractor(features)
        high_level_features = self.transformer(local_features)
        output = high_level_features.permute(1, 0, 2) # (batch_size, seq_length-1, 50)
        output = output.mean(dim=1)
        output = self.fc(output)

        return  output

class Wav2Vec_DisfluencyModel(nn.Module):
    def __init__(self):
        super(Wav2Vec_DisfluencyModel, self).__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor(freeze=False)
        self.fc = nn.Linear(768, 2)
        
    def forward(self, audio_input):
        features = self.feature_extractor(audio_input)
        features = features.mean(dim=1) 
        output = self.fc(features)

        return  output
