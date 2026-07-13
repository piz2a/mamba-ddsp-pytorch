import torch
import torch.nn as nn

# Test out training and testing RNN, LSTM, GRU, Transformer, Mamba layers for the same sequence modeling task.
# This is a simple example to show how to use these layers in PyTorch.
# I'm doing this to master writing the PyTorch code in my own.

# What dataset should I use? 
# If I set the input to a sequence of audio data, which form of output should I use?
# A classifier or making output as an another audio sequence? I think the latter is more interesting. So, I will use a dataset of audio sequences and try to predict the next audio sequence given the previous ones.

# Let's find out which datasets are available for audio sequence modeling. I will use the torchaudio library to load the datasets.


class SimpleAudioGRU(nn.Module):
    def __init__(self, n_fft=400, hop_length=160, hidden_dim=128, num_classes=5):
        super(SimpleAudioGRU, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # STFT 결과물인 주파수 성분의 개수가 입력 특징 차원(D_in)이 됩니다.
        # n_fft // 2 + 1 공식에 의해 400의 경우 201 차원이 됩니다.
        input_dim = n_fft // 2 + 1
        
        # [B, T, D_in] -> [B, T, hidden_dim]
        self.gru = nn.GRU(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True
        )
        
        # 최종 분류를 위한 선형 레이어 (시퀀스의 마지막 프레임 정보만 사용)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # 1. 입력 오디오 파형 차원: [B, Raw_Samples] (e.g., [4, 16000])
        batch_size = x.size(0)
        
        # 2. 파이토치 내장 STFT 수행
        # return_complex=True로 설정하여 복소수 형태의 스펙트로그램 추출
        stft = torch.stft(
            x, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            return_complex=True
        ) # 출력 차원: [B, Freq_Bins, Time_Frames]
        
        # 3. 크기(Magnitude) 스펙트로그램 계산하여 실수형 데이터로 변환
        mag_spec = torch.abs(stft) # [B, Freq_Bins, Time_Frames]
        
        # 4. 시퀀스 모델 규격인 [B, T, D]로 맞추기 위해 주파수 축과 시간 축을 뒤집음 (Transpose)
        # [B, Freq_Bins, Time_Frames] -> [B, Time_Frames, Freq_Bins]
        x_seq = mag_spec.transpose(1, 2) 
        
        # 5. GRU 통과
        # output: 모든 시간 축에 대한 은닉 상태 [B, T, hidden_dim]
        # h_n: 맨 마지막 스텝의 은닉 상태 [Num_Layers, B, hidden_dim]
        output, h_n = self.gru(x_seq)
        
        # 6. 분류문제를 풀기 위해 시퀀스의 맨 마지막 프레임(Last Time Step)의 아웃풋만 추출
        # output[:, -1, :] 차원: [B, hidden_dim]
        last_frame_feat = output[:, -1, :]
        
        # 7. 최종 클래스 로직 출력 [B, num_classes]
        out = self.fc(last_frame_feat)
        return out

# --- 코드 작동 및 차원 흐름 검증 테스트 ---
if __name__ == "__main__":
    # 가상의 오디오 데이터 생성
    # 배치 크기 = 4, 오디오 길이 = 1초 (16000 샘플, Sampling Rate 16kHz 가정)
    mock_audio = torch.randn(4, 16000)
    
    # 모델 선언 및 연산
    model = SimpleAudioGRU(n_fft=400, hop_length=160, hidden_dim=128, num_classes=5)
    output = model(mock_audio)
    
    print(f"입력 오디오 차원: {mock_audio.shape}")
    print(f"최종 출력 (Class Logits): {output}") # [4, 5]