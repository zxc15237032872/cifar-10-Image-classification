import torch
import os
import cv2
import torchvision
from PIL import Image
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

use_gpu = torch.cuda.is_available()
model = torch.load("autodl-tmp/CIFAR10/model/model1_250.pth", map_location=torch.device('cuda' if use_gpu else 'cpu'))
model.eval()
# 指定文件夹路径
folder_path = 'autodl-tmp/CIFAR10/testimages'

# 获取文件夹中所有文件
files = os.listdir(folder_path)

# 获取文件夹中所有图片文件
image_files = [os.path.join(folder_path, f) for f in files if f.endswith('.jpg') or f.endswith('.png')]

# 遍历所有图片文件进行预测
for img in image_files:
    # 获取图片文件名（不含后缀）作为真实类别
    img_name = os.path.splitext(os.path.basename(img))[0]
    image = cv2.imread(img)
    image = cv2.resize(image, (32, 32))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = transform(image)
    image = torch.reshape(image, (1, 3, 32, 32))
    image = image.to('cuda' if use_gpu else 'cpu')
    output = model(image)
    value, index = torch.max(output.data, 1)
    pre_val = classes[index]
    is_correct = pre_val == img_name
    print(f"图片：{img_name:<15}，预测结果：{pre_val:<15}，判断真假：{is_correct}")