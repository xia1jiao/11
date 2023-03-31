import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import matplotlib.pyplot as plt
import os
import torch.optim as optim
from model import resnet18
# 这些是我们导入的依赖库。其中包括PyTorch和相关的深度学习工具，以及一些用于处理数据和评估模型性能的库。

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),  # 长宽比例不变，将最小边缩放到256
                                   transforms.CenterCrop(224),  # 再中心裁减一个224*224大小的图片
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))  # get data root path
    image_path = data_root + "./dataset/"  # data set path
    # 数据集位置

    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    brid_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in brid_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16  #
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)  # nw  单线程编译

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=0)  # nw   单线程编译

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))  #
    # 若不使用迁移学习的方法，注释掉61-69行，并net = resnet18(num_calsses参数)
    net = resnet18()  # 未传入参数，最后一个全连接层有1000个结点  这里也可以使用resnet101、50、34、18等网络，不过得提前下载
    # load pretrain weights

    model_weight_path = "./resnet18_pretrain.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # net.load_state_dict载入模型权重。torch.load(model_weight_path)载入到内存当中还未载入到模型当中
    missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)
    # for param in net.parameters():
    #     param.requires_grad = False
    # 改变fc层的结构
    in_channel = net.fc.in_features  # 输入特征矩阵的深度。net.fc是所定义网络的全连接层
    # 类别个数
    net.fc = nn.Linear(in_channel, 400)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    # optimizer = optim.SGD(net.parameters(), lr=0.0001)
    best_acc = 0.0
    save_path = './resNet18.pth'  # 可以自己命名，后续预测时得对应得上
    for epoch in range(50):  # 训练次数
        # train
        net.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # print train process
            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")
        print()

        # validate
        net.eval()  # 控制训练过程中的Batch normalization
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            for val_data in validate_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))  # 评估模型只有最后一个输出层
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += (predict_y == val_labels.to(device)).sum().item()
            val_accurate = acc / val_num
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)
            print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
                  (epoch + 1, running_loss / step, val_accurate))

        # write_result("resnet18_trainData.txt", epoch + 1, loss[0], running_loss/step, val_accurate)

    print('Finished Training')

# def write_result(fileloc, epoch, trainloss, testloss, testaccuracy):
#     with open(fileloc,"a") as f:
#         data = "Epoch: " + str(epoch) + "\tTrainLoss " + str(trainloss) + "\tTestLoss " + str(testloss) + "\tTestAccuracy " + str(testaccuracy) + "\n"
#         f.write(data)

if __name__ == '__main__':
    main()
