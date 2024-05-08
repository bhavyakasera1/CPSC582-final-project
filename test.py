from torch.autograd import Function
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
from dataset import get_dataset, get_dataloader
import numpy as np
from domain_adaptation import DACNN


if __name__ == '__main__':
    image_size = 224
    model = DACNN()
    test_ds = get_dataset('data/Br35H_preprocessed')
    test_dl = get_dataloader(test_ds, batch_size=1)

    model.load_state_dict(torch.load('reverse_grad.pth')["model_state_dict"])
    loss_fn_class = nn.NLLLoss()
    model.eval()

    test_loss = 0
    num_correct = 0
    domains = []
    for batch_idx in range(len(test_dl)):

        X_s, y_s = next(iter(test_dl))
        class_pred, domain_pred = model(X_s)

        domains.append(domain_pred.argmax(dim=1).item())

        loss_s_label = loss_fn_class(class_pred, y_s)        
        test_loss += loss_s_label.item()

        pred_class = torch.argmax(class_pred, dim=1)
        num_correct += torch.sum(pred_class == y_s).item()

        if pred_class != y_s:
            torchvision.utils.save_image(X_s.squeeze(0), f'incorrect/{batch_idx}_pred_{pred_class}_correct_{y_s}.png')

    print(sum(domains))
    print(f'Loss: {test_loss/len(test_dl):.4f} Acc: {num_correct/len(test_dl):.4f}')