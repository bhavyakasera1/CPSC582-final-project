
from torch.autograd import Function
import torch.nn as nn
import torch.optim as optim
import torch
from dataset import get_dataset, get_dataloader
import numpy as np


# Autograd Function objects are what record operation history on tensors,
# and define formulas for the forward and backprop.

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        # Store context for backprop
        ctx.alpha = alpha
        
        # Forward pass is a no-op
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass is just to -alpha the gradient
        output = grad_output.neg() * ctx.alpha

        # Must return same number as inputs to forward()
        return output, None

class DACNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(64, 50, kernel_size=5),
            nn.BatchNorm2d(50), nn.Dropout2d(), nn.MaxPool2d(2),
            nn.ReLU(True),
        )
        self.class_classifier = nn.Sequential(
            nn.Linear(50 * 53 * 53, 100), nn.BatchNorm1d(100), nn.Dropout2d(),
            nn.ReLU(True),
            nn.Linear(100, 100), nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(50 * 53 * 53, 100), nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x, grl_lambda=1.0):
        # Handle single-channel input by expanding (repeating) the singleton dimention
        # x = x.expand(x.data.shape[0], 3, image_size, image_size)
        
        features = self.feature_extractor(x)
        features = features.view(-1, 50 * 53 * 53)
        reverse_features = GradientReversalFn.apply(features, grl_lambda)
        
        class_pred = self.class_classifier(features)
        domain_pred = self.domain_classifier(reverse_features)
        return class_pred, domain_pred
if __name__ == '__main__':
    image_size = 224
    model = DACNN()
    ds_source = get_dataset('data/LGG_preprocessed')
    dl_source = get_dataloader(ds_source, batch_size=128)
    ds_target = get_dataset('data/Br35H_preprocessed')
    dl_target = get_dataloader(ds_target, batch_size=128)

    x0_s, y0_s = next(iter(dl_source))
    x0_t, y0_t = next(iter(dl_target))

    print('source domain: ', x0_s.shape, y0_s.shape)
    print('target domain: ', x0_t.shape, y0_t.shape)

    model(x0_s)
    model(x0_t)

    lr = 1e-3
    n_epochs = 15

    # Setup optimizer as usual
    model = DACNN()
    optimizer = optim.Adam(model.parameters(), lr)

    # Two losses functions this time
    loss_fn_class = nn.NLLLoss()
    loss_fn_domain = nn.NLLLoss()

    batch_size = 128
    # dl_source = torch.utils.data.DataLoader(ds_source, batch_size)
    # dl_target = torch.utils.data.DataLoader(ds_target, batch_size)

    # We'll train the same number of batches from both datasets
    max_batches = min(len(dl_source), len(dl_target)) - 1
    # max_batches = 12

    for epoch_idx in range(n_epochs):
        print(f'Epoch {epoch_idx+1:04d} / {n_epochs:04d}', end='\n=================\n')
        dl_source_iter = iter(dl_source)
        dl_target_iter = iter(dl_target)
        epoch_loss = 0

        for batch_idx in range(max_batches):
            optimizer.zero_grad()
            # Training progress and GRL lambda
            p = float(batch_idx + epoch_idx * max_batches) / (n_epochs * max_batches)
            grl_lambda = 2. / (1. + np.exp(-10 * p)) - 1

            # Train on source domain
            X_s, y_s = next(dl_source_iter)
            y_s_domain = torch.zeros(batch_size, dtype=torch.long) # generate source domain labels

            class_pred, domain_pred = model(X_s, grl_lambda)
            loss_s_label = loss_fn_class(class_pred, y_s)
            loss_s_domain = loss_fn_domain(domain_pred, y_s_domain)

            # Train on target domain
            X_t, _ = next(dl_target_iter) # ignore target domain class labels!
            y_t_domain = torch.ones(batch_size, dtype=torch.long) # generate target domain labels

            _, domain_pred = model(X_t, grl_lambda)
            loss_t_domain = loss_fn_domain(domain_pred, y_t_domain)
            
            loss = loss_t_domain + loss_s_domain + loss_s_label
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            print(f'[{batch_idx+1}/{max_batches}] '
                f'class_loss: {loss_s_label.item():.4f} ' f's_domain_loss: {loss_s_domain.item():.4f} '
                f't_domain_loss: {loss_t_domain.item():.4f} ' f'grl_lambda: {grl_lambda:.3f} '
                )
        torch.save({
                'epoch': epoch_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch_loss': epoch_loss,
                }, f"{'reverse_grad'}.pth")
        print(f'Epoch {epoch_idx+1:04d} / {n_epochs:04d} Loss: {epoch_loss/max_batches:.4f}')