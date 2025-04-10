from torch.optim import lr_scheduler
import torch.optim as optim
from model import * 
import matplotlib.pyplot as plt

model = random_classifier(NUM_CLASSES=4)
lr = 0.0002
optimizer = optim.Adam(model.parameters(), lr=lr)
num_steps_optimizer1 = 500
epochs = 500
warmup_epochs = 150
scheduler1 = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps_optimizer1, eta_min=0.00015, verbose=True, last_epoch=-1)
scheduler2 = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9999)

lr_schedule = []

for epoch in range(epochs):
    if epoch < warmup_epochs:
        # Linear warmup
        lr_scale = min(1., 0.5 + float(epoch + 1) / (2* float(warmup_epochs)))
        for pg in optimizer.param_groups:
            pg['lr'] = lr * lr_scale
    optimizer.step()
    scheduler1.step()
    lr_schedule.extend([optimizer.param_groups[0]['lr']])

# plotting the simulated LR schedule
plt.figure(figsize=(5, 3))
plt.plot(range(len(lr_schedule)), lr_schedule)
plt.xlabel('Epochs')
plt.ylabel('Learning rate')
plt.tight_layout()
plt.show()