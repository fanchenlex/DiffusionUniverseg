from tqdm.auto import tqdm
import numpy as np
import torch
from sde_lib import VESDE
import datasets as datasets
import time
import torch.nn as nn
from collections import defaultdict
from method_unet import UNet
from method_deeplabv3 import PSPNet
# 统一resize到240
from torchvision.transforms import Resize
torch_resize = Resize([240,240])


 # 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dice metric for measuring volume agreement
def dice_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    y_pred = y_pred.long()
    y_true = y_true.long()
    score = 2*(y_pred*y_true).sum() / (y_pred.sum() + y_true.sum())
    return score.item()

def accuracy_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    y_pred = y_pred.long()
    y_true = y_true.long()
    total_pixels = 240 * 240
    correct_pixels = torch.sum(y_pred == y_true)
    accuracy = correct_pixels / total_pixels
    return accuracy.item()

# run inference and compute losses for one test image
@torch.no_grad()
def inference(model, image, label):
    image, label = image.to(device).unsqueeze(dim=0), label.to(device).unsqueeze(dim=0)
    # inference
    soft_pred = model(image)    
    hard_pred = soft_pred.round().clip(0,1).squeeze(dim=0)

    #  score
    # print(hard_pred.size())
    # print(label.size())
    score = dice_score(hard_pred, label.squeeze(dim=0))
    acc = accuracy_score(hard_pred, label.squeeze(dim=0))

    # return a dictionary of all relevant variables
    return {'Image': image,
            'Soft Prediction': soft_pred,
            'Prediction': hard_pred,
            'Ground Truth': label,
            'score': score,
            'acc': acc}

def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        loss_total=[]
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)

            inputs = torch_resize(inputs)
            labels = torch_resize(labels)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_total.append(loss.item())
        print(sum(loss_total))
        
def test(model, d_test, flag):
    if flag:
        model.eval()
    n_predictions = len(d_test)
    results = defaultdict(list)
    # compute inference and save predictions and metrics for n_predictions
    idxs = np.random.permutation(len(d_test))[:n_predictions]

    for i in tqdm(idxs):
        image, label = d_test[i]

        image = torch_resize(image)
        label = torch_resize(label)

        vals = inference(model, image, label)
        for k, v in vals.items():
            results[k].append(v)
    # visualize the results, along with their scores
    scores = results.pop('score')    
    print("Dice score: ", np.array(scores).mean())
    acc = results.pop('acc')
    print("Pixel accuracy: ", np.array(acc).mean())


def main():
    # # 设置设备
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建训练和测试数据加载器
    import UniverSeg.example_data as example_data
    catgory=2
    import UniverSeg.example_data.oasis
    d_support = example_data.oasis.OASISDataset('support', label=catgory)
    d_test = example_data.oasis.OASISDataset('test', label=catgory)

    # print(len(d_support))
    # print(len(d_test))
    #使用dataloader构造train_dataloader
    train_dataloader=torch.utils.data.DataLoader(d_support,
                                            batch_size=2,
                                            shuffle=True, #每个epoch打乱一次
                                            num_workers=1,
                                            drop_last=True
                                            )

    # 初始化模型、损失函数和优化器
    unet_model = UNet(1, 1).to(device)
    DeepLabV3_model = PSPNet(nInputChannels = 1, num_classes = 1, os = 8, backbone="resnet50", pretrained = False, aux_branch=False).to(device)

    model = unet_model

    if model == DeepLabV3_model:
        flag = 1
    else:
        flag = 0

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train(model, train_dataloader, criterion, optimizer, num_epochs=10)
    # 评估模型
    test(model, d_test, flag)

main()