from torch.optim import lr_scheduler
import torch.optim as optim
from model import * 
import matplotlib.pyplot as plt
from torchvision import transforms
from dataset import CPEN455Dataset  # Your custom dataset class


model = random_classifier(NUM_CLASSES=4)
lr = 0.0002
optimizer = optim.Adam(model.parameters(), lr=lr)
num_steps_optimizer1 = 500
epochs = 500
warmup_epochs = 50
scheduler1 = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps_optimizer1, eta_min=0.00015, verbose=True, last_epoch=-1)
scheduler2 = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99995)

lr_schedule = []

for epoch in range(epochs):
    if epoch < warmup_epochs:
        # Linear warmup
        lr_scale = min(1., 0.5 + float(epoch + 1) / (2* float(warmup_epochs)))
        for pg in optimizer.param_groups:
            pg['lr'] = lr * lr_scale
    optimizer.step()
    scheduler2.step()
    lr_schedule.extend([optimizer.param_groups[0]['lr']])

# plotting the simulated LR schedule
plt.figure(figsize=(5, 3))
plt.plot(range(len(lr_schedule)), lr_schedule)
plt.xlabel('Epochs')
plt.ylabel('Learning rate')
plt.tight_layout()
plt.show()

# affine_transfomer = transforms.RandomAffine(translate=(0.1, 0.1))

# dataset = CPEN455Dataset(root_dir='data', mode='train', transform=affine_transfomer)

# img, label = dataset[0]
# plt.figure(figsize=(15, 6))
# plt.imshow(img)


