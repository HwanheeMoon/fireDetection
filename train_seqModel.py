import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torchfusion_utils.metrics import Accuracy
from models.seqModel import SeqClassifier
import warnings
from torch.utils.data import random_split
import torchvision.models as models

warnings.filterwarnings("ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


classes = ['default','fire','smoke']

input_size = 1536  # 이미지 특징의 크기
hidden_size = 128  # RNN hidden 크기
num_classes = len(classes)  # 분류할 클래스 수
num_layers = 2  # RNN 레이어 개수

# 모델 생성
final_model = SeqClassifier(input_size, hidden_size, num_classes, num_layers, use_LSTM=True).to(device) #LSTM + 
#final_model = models.resnet50(pretrained=True).to(device)
#final_model.load_state_dict(torch.load("last-lstm mono.pt"))

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    #transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터셋 폴더 경로
data_dir = "Dataset_AIHub"
#data_dir = "data/data/img_data/train"

# ImageFolder를 사용하여 데이터셋 생성
ds = datasets.ImageFolder(root=data_dir, transform=transform)

train_len = int(len(ds) * 0.8)
val_len = int(len(ds) * 0.2)
dataset, val_dataset = random_split(ds, [train_len, val_len])



# 데이터 로딩을 위한 데이터로더 설정
batch_size = 16
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset,batch_size=batch_size)

learning_rate = 0.0001
num_epochs = 20
loss = 0
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(final_model.parameters(), lr=learning_rate)

print("Training Start...")

train_acc = Accuracy()
val_acc = Accuracy()

max_acc = -9999
min_loss = 1000

for epoch in range(num_epochs):
    now = 0
    train_acc.reset()

    final_model.train()

    for inputs, labels in train_dataloader:
        total = len(train_dataloader)
        optimizer.zero_grad()
        outputs = final_model(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        now += 1
        train_acc.update(outputs, labels)

        print(f"\rTrain : epoch {epoch} : {now}/{total} , Accuracy : {100* train_acc.getValue():.3f} %, Loss: {loss.item():.4f}", end=" ")
    
    print(f"\nTrain : epoch {epoch} Accuracy : {100 * train_acc.getValue():.3f} %, Loss: {loss.item():.4f}")


    final_model.eval()
    now = 0

    print("Validation Start")

    with torch.no_grad():
        val_acc.reset()
        for inputs, labels in val_dataloader:
            total = len(val_dataloader)
            outputs = final_model(inputs.to(device))
            val_loss = criterion(outputs, labels.to(device))
            now += 1
            val_acc.update(outputs, labels)
            if val_acc.getValue() > max_acc and loss.item() < min_loss:
                max_acc = val_acc.getValue()
                min_loss = val_loss.item()
                model_path = 'best_train-lstm.pt'
                torch.save(final_model.state_dict(), model_path)

            print(f"\rval : epoch {epoch} : {now}/{total} , Accuracy : {100* val_acc.getValue():.3f} %, Loss: {val_loss.item():.4f}", end=" ")
    
    print(f"\nval : epoch {epoch} Accuracy : {100 * val_acc.getValue():.3f} %, Loss: {val_loss.item():.4f}")


model_path = 'last-lstm.pt'
torch.save(final_model.state_dict(), model_path)
