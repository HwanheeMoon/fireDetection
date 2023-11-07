import torchvision.models as models
import torch.nn as nn
import torch

class SeqClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers, use_LSTM = False):
        super(SeqClassifier, self).__init__()

        self.cnn = models.efficientnet_b3(pretrained=True)
        #self.cnn = models.resnet101(pretrained=True)

        if use_LSTM:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 이미지 특징 추출

        batch_size, c, h, w = x.size()
        image_features = self.cnn(x)
        image_features = image_features.view(batch_size, -1)
        
        # LSTM 모델 실행
        lstm_input = image_features.unsqueeze(1)  # 시퀀스 차원 추가

        # RNN 입력
        rnn_output, _ = self.rnn(lstm_input)

        # LSTM의 마지막 출력을 사용하여 분류
        output = self.fc(rnn_output[:, -1, :])

        return output
