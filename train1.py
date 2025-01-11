import torch
import torch.cuda
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from net1 import MyModel
import os

writer = SummaryWriter(log_dir='logs')

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 训练数据集
train_data_set = datasets.CIFAR10("./dataset", train=True, transform=transform, download=True)

# 测试数据集
test_data_set = datasets.CIFAR10("./dataset", train=False, transform=transform, download=True)

train_data_size = len(train_data_set)
test_data_size = len(test_data_set)

print("训练集：{},验证集:{}".format(train_data_size, test_data_size))

# 加载数据集
train_data_loader = DataLoader(train_data_set, batch_size=64, shuffle=True)
test_data_loader = DataLoader(test_data_set, batch_size=64)

# 网络定义
myModel = MyModel()

# 是否使用gpu
use_gpu = torch.cuda.is_available()
if (use_gpu):
    print("gpu可用")
    myModel = myModel.cuda()

# 训练轮数
epochs = 300
# 损失函数
lossFn = nn.CrossEntropyLoss()
# 优化器
optimizer = SGD(myModel.parameters(), lr=0.01)


def save_onnx_model(model, epoch, input_shape=(1, 3, 32, 32)):
 
    model.eval()
    dummy_input = torch.randn(input_shape)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
        model = model.cuda()
    onnx_file_path = f"autodl-tmp/CIFAR10/model/model1_{epoch}.onnx"
    torch.onnx.export(model, dummy_input, onnx_file_path, verbose=True)


for epoch in range(epochs):
    print("训练轮数{}/{}".format(epoch + 1, epochs))

    # 损失变量
    train_total_loss = 0.0
    test_total_loss = 0.0
    # 精度
    train_total_acc = 0.0
    test_total_acc = 0.0

    # 训练开始
    for data in train_data_loader:
        inputs, labels = data
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        outputs = myModel(inputs)
        loss = lossFn(outputs, labels)
        _, pred = torch.max(outputs, 1)
        acc = torch.sum(pred == labels).item()
        loss.backward()
        optimizer.step()

        train_total_loss += loss.item()
        train_total_acc += acc

    # 测试
    with torch.no_grad():
        for data in test_data_loader:
            inputs, labels = data
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = myModel(inputs)
            loss = lossFn(outputs, labels)
            _, pred = torch.max(outputs, 1)
            acc = torch.sum(pred == labels).item()

            test_total_loss += loss.item()
            test_total_acc += acc

    print("train loss:{},acc:{}. test loss:{},acc:{}".format(train_total_loss, train_total_acc / train_data_size,
                                                             test_total_loss, test_total_acc / test_data_size))
    writer.add_scalar('Loss/train', train_total_loss, epoch)
    writer.add_scalar('Loss/test', test_total_loss, epoch)
    writer.add_scalar('acc/train', train_total_acc / train_data_size, epoch)
    writer.add_scalar('acc/test', test_total_acc / test_data_size, epoch)
    if ((epoch + 1) % 50 == 0):
        torch.save(myModel, "autodl-tmp/CIFAR10/model/model1_{}.pth".format(epoch + 1))
        save_onnx_model(myModel, epoch + 1)