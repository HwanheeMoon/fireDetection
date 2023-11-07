import torchvision.transforms as transforms
import torch
import cv2
from models.seqModel import SeqClassifier
import torch
import cv2
import time
import sqlite3
from torchvision import models


def generate_frames(Position):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    classes = ['default', 'fire', 'smoke',]
    input_size = 1536  # 이미지 특징의 크기 reset -- 2048 effi -- 1536
    hidden_size = 128  # GRU hidden 크기
    num_classes = len(classes)  # 증감 클래스 수
    num_layers = 2  # GRU 레이어 개수

    model = SeqClassifier(input_size, hidden_size, num_classes, num_layers, use_LSTM=True).to(device)
    #model = models.resnet50(pretrained=True)
    model.load_state_dict(torch.load('last-lstm-effi.pt')) # classifier model
    model.to(device).eval()

    if Position != 'CCTV':
        cap = cv2.VideoCapture(f'vids/{Position}.mp4')
    else:
        cap = cv2.VideoCapture(0)

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) # 이미지 전처리


    pivot = False
    cnt = 0
    con = sqlite3.connect("db.sqlite3",isolation_level=None)
    cur = con.cursor()
    cont = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #img0 = torch.from_numpy(input_image).to(device).float() / 255.0
        input_image = preprocess(input_image).unsqueeze(0).to(device)

        with torch.no_grad():
            inc_pred = model(input_image).to(device) # classifier 

        _, pred2 = torch.max(inc_pred, 1)
        pred_index = pred2.item()

        if pred_index == 1 or pred_index == 2:
            pivot = True
            cont = True
            if cnt > 1:
                cont = False
            t = time.strftime('%Y.%m.%d - %H:%M:%S')

            if cont:
                cur.execute("INSERT INTO detect_detected_log VALUES (?, ?, ?)", (None, t, cnt))
                #print(f"화재가 검출 되었음 !! ", time.strftime('%Y.%m.%d - %H:%M:%S'))
            

            cnt += 1
            cv2.putText(frame, f"{classes[pred_index]}", (20, 70), cv2.FONT_HERSHEY_COMPLEX, 1.5,
                        (0, 0, 255), 2)
            top_left = (10, 10)  # (x, y) 좌표
            bottom_right = (frame.shape[1] - 10, frame.shape[0] - 10)
            cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 10)
            
        else:
            pivot = False
            cont = False
            if cnt > 0:
                t = time.strftime('%Y.%m.%d - %H:%M:%S')
                cur.execute("INSERT INTO detect_detected_log VALUES (?, ?, ?)", (None, t, cnt))
                #print(f"{cnt} 프레임 동안 화재가 검출 되었음.", time.strftime('%Y.%m.%d - %H:%M:%S'))
            cnt = 0
            cv2.putText(frame, f"{classes[pred_index]}", (20, 70), cv2.FONT_HERSHEY_COMPLEX, 1.5,
                        (0, 0, 255), 2)
                    
        ret, buffer = cv2.imencode('.jpg', frame)
        # JPEG 이미지를 바이트 스트림으로 변환
        frame_bytes = buffer.tobytes()

        # 웹페이지로 프레임 전송
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.01)

        k = cv2.waitKey(1)  # 1 millisecond
        if k == 27:
            exit()


