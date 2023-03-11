import cv2
import numpy as numpy
import torch.nn.functional as f1
import json
import torch
import torchvision.transforms as transforms
from flask import Flask, request, render_template
from model import resnet152


app = Flask(__name__)
weights_path = "./resNet152.pth"
# 加载模型
# select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# create model
model = resnet152(num_classes=400).to(device)
# load model weights
model.load_state_dict(torch.load(weights_path, map_location=device))

model.eval()

# 加载标签
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)


# 转换图像
def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = cv2.imdecode(numpy.fromstring(image_bytes.read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = my_transforms(image).unsqueeze(0)
    return image


# 进行分类
def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model(tensor)
    probs = f1.softmax(outputs, dim=1)
    _, predicted = torch.max(outputs, 1)
    class_name = class_indices[str(predicted.item())]
    confidence = round(probs[0][predicted.item()].item(), 4)
    return class_name, confidence


# 处理上传请求
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction='No file selected')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction='No file selected')
        if file:
            class_name, confidence = get_prediction(file)
            return render_template('index.html', prediction=f'The image is a {class_name} with {confidence} confidence')
    return render_template('index.html')
