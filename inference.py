import torch

import torch
import torch.nn as nn
import librosa
import numpy as np 

class ParallelModel(nn.Module):
    def __init__(self,num_emotions):
        super().__init__()
        # conv block
        self.conv2Dblock1 = nn.Sequential(
            # 1. conv block
            nn.Conv2d(in_channels=1,
                       out_channels=16,
                       kernel_size=7,
                       stride=1,
                       padding=1
                      ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3))
            # 2. conv block
        self.conv2Dblock2 = nn.Sequential(nn.Conv2d(in_channels=16,
                       out_channels=32,
                       kernel_size=5,
                       stride=1,
                       padding=1
                      ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3))
            # 3. conv block
        self.conv2Dblock3 = nn.Sequential(nn.Conv2d(in_channels=32,
                       out_channels=64,
                       kernel_size=3,
                       stride=1,
                       padding=1
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3))
            # 4. conv block
        self.conv2Dblock4 = nn.Sequential(nn.Conv2d(in_channels=64,
                       out_channels=64,
                       kernel_size=3,
                       stride=1,
                       padding=1
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3))
        self.conv2Dblock5 = nn.Sequential(nn.Conv2d(in_channels=64,
                       out_channels=128,
                       kernel_size=3,
                       stride=1,
                       padding=1
                      ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3))
        
        # Transformer block
        self.transf_maxpool = nn.MaxPool2d(kernel_size=[2,4], stride=[2,4])
        transf_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, dropout=0.4, activation='relu')
        self.transf_encoder = nn.TransformerEncoder(transf_layer, num_layers=4)
        # Linear softmax layer
        self.out_linear = nn.Linear(640, num_emotions)
        self.dropout_linear = nn.Dropout(p=0.0)
        self.out_softmax = nn.Softmax(dim=1)
    def forward(self,x):
        # conv embedding
        # print("in",x.size())
        conv_embedding = self.conv2Dblock1(x) #(b,channel,freq,time)
        conv_embedding = self.conv2Dblock2(conv_embedding)
        conv_embedding = self.conv2Dblock3(conv_embedding)
        conv_embedding = self.conv2Dblock4(conv_embedding)
        conv_embedding = self.conv2Dblock5(conv_embedding)
        conv_embedding = torch.flatten(conv_embedding, start_dim=1) # do not flatten batch dimension
        # transformer embedding
        x_reduced = self.transf_maxpool(x)
        x_reduced = torch.squeeze(x_reduced,1)
        x_reduced = x_reduced.permute(2,0,1) # requires shape = (time,batch,embedding)
        # print("2",x_reduced.size())
        transf_out = self.transf_encoder(x_reduced)
        transf_embedding = torch.mean(transf_out, dim=0)
        # print("transf", transf_embedding.size())
        # concatenate
        complete_embedding = torch.cat([conv_embedding, transf_embedding], dim=1) 
        # final Linear
        # print("complete",complete_embedding.size())
        complete_embedding = self.dropout_linear(complete_embedding)
        output_logits = self.out_linear(complete_embedding)
        output_softmax = self.out_softmax(output_logits)
        return output_softmax
                                     
def getMELspectrogram(audio):
    mel_spec = librosa.stft(y=audio,
                            n_fft=1024,
                            win_length = 512,
                            window='hann',
                            hop_length = 256,
                            )
    mel_spec = np.abs(mel_spec)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # mel_spec_db = mel_spec
    return mel_spec_db

if __name__ == "__main__":
    import os
    EMOTIONS = {1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 0:'surprise'} # surprise je promenjen sa 8 na 0
    DATA_PATH = "/home/v-haoheliu/audio_speech_actors_01-24"
    SAMPLE_RATE = 22050
    LOAD_PATH = os.path.join(os.getcwd(),'notebooks',"cnn_transformer","cnn_transf_parallel_model100.pt")
    model = ParallelModel(len(EMOTIONS))
    model.load_state_dict(torch.load(os.path.join(LOAD_PATH)))
    print('Model is loaded from {}'.format(os.path.join(LOAD_PATH)))
    
    audio_file = "LJ001-0014.wav"
    x,_ = librosa.load(audio_file,sr=SAMPLE_RATE)
    x = x[-SAMPLE_RATE*3:,...]
    print(x.shape)
    spec = getMELspectrogram(x)
    spec = torch.tensor(spec).float()[None,None,...]
    out = model(spec)
    print(out)
