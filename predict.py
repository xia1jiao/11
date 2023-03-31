import torch
from model import resnet152
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # 预处理

# load image
img = Image.open("H:/鸟类数据集/train/ABBOTTS BABBLER/001.jpg")  # 导入需要检测的图片
plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# read class_indict
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
model = resnet152(num_classes=400)  # 修改为你训练时一共的种类数
# load model weights
model_weight_path = "./resNet152.pth"  # 导入训练好的模型
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()
# predict_list = []
with torch.no_grad():  # 不对损失梯度进行跟踪
    # predict class
    output = torch.squeeze(model(img))  # 压缩batch维度
    predict = torch.softmax(output, dim=0)  # 得到概率分布
    predict_cla = torch.argmax(predict).numpy()  # argmax寻找最大值对应的索引
    # predict_list.append(predict.tolist()[0])
print(class_indict[str(predict_cla)], predict[predict_cla].numpy())
plt.show()
