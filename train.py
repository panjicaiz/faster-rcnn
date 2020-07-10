'''
Task: Object Detection
Dataset: wheat
Algorithm: faster-rcnn
Created: 2020/7/9
Refer: 
Description: kaggle wheat detection with some tricks
'''
import torch
import torchvision
from torch import optim
from datasets import ObjectDetectionDataset
from torch.utils.data import DataLoader
import utils
from utils import collate_fn, MetricLogger, SmoothedValue


# hyperparameters
lr = 0.005
momentum = 0.9
weight_decay = 0.0005
step_size = 3
gamma = 0.1
num_epochs = 10
print_freq = 10


# datasets
train_dataset = ObjectDetectionDataset('train', label='train.csv')
train_data_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
valid_dataset = ObjectDetectionDataset('val', label='train.csv')
valid_data_loader = DataLoader(valid_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)


# model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


# train
for epoch in range(num_epochs):
    metric_logger = MetricLogger(delimiter=' ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    model.train()
    for images, targets in metric_logger.log_every(train_data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])



# inference
