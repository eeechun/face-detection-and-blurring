# %%
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import torchvision.utils
import matplotlib.pyplot as plt

# This is for the progress bar.
from tqdm.auto import tqdm

# %%
# parameters
batch_size = 4
n_epochs = 10
# n_epochs = 0  # to get statistics (control whether to train)
patience = 5  # If no improvement in 'patience' epochs, early stop

## blur
### b15
# train_set_dir = "blur/15/train"
# test_set_dir = "blur/15/test"
# model_path = "./model_b15.ckpt"

### b45
# train_set_dir = "blur/45/train/"
# test_set_dir = "blur/45/test/"
# model_path = "./model_b45.ckpt"

### b99
# train_set_dir = "blur/99/train/"
# test_set_dir = "blur/99/test/"
# model_path = "./model_b99.ckpt"

## pixel
### p4
# train_set_dir = "pixel/4/train/"
# test_set_dir = "pixel/4/test/"
# model_path = "./model_p4.ckpt"

### p8
# train_set_dir = "pixel/8/train/"
# test_set_dir = "pixel/8/test/"
# model_path = "./model_p8.ckpt"

### p16
# train_set_dir = "pixel/16/train/"
# test_set_dir = "pixel/16/test/"
# model_path = "./model_p16.ckpt"

### ori
train_set_dir = "att_img_flat/train/"
test_set_dir = "att_img_flat/test/"
model_path = "./model_ori.ckpt"

# %%
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((128, 128), antialias=True),
        transforms.Normalize((0.5), (0.5)),
    ]
)


# %%
class mydataset(Dataset):
    def __init__(self, path, tfm=transform, files=None):
        super(mydataset).__init__()
        self.path = path
        self.files = sorted(
            [os.path.join(path, x) for x in os.listdir(path) if x.endswith(".png")]
        )
        if files:
            self.files = files
        print(f"One {path} sample", self.files[0])
        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        try:
            label = int(fname.split("s")[-1].split("_")[0])
        except:
            label = -1  # test has no label
        return im, label


# %%
train_set = mydataset(train_set_dir, tfm=transform)
test_set = mydataset(test_set_dir, tfm=transform)

train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
)
test_loader = DataLoader(
    test_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
)

# %%
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 1, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 64, 64]
            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]
            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]
            nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]
            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 41),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


net = Net()

# %%
def fgsm_attack(model, loss, images, labels, eps) :
# adversarial attack
    
    images = images.to(device)
    labels = labels.to(device)
    images.requires_grad = True

    outputs = model(images)
    
    cost = loss(outputs, labels).to(device)
    model.zero_grad()
    cost.backward()

    attack_images = images + eps*images.grad.sign()
    attack_images = torch.clamp(attack_images, -1, 1)
    
    return attack_images

#%%
def imshow(img):
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

# %%
torch.cuda.is_available = lambda: False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize a model, and put it on the device specified.
model = Net().to(device)
model.load_state_dict(torch.load(model_path))


# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

# Initialize trackers, these are not parameters and should not be changed
stale = 0
best_acc = 0

for epoch in range(n_epochs):

    # ---------- Testing ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in testing.
    test_loss = []
    test_accs = []

    # Iterate the testing set by batches.
    for batch in tqdm(test_loader):

        images, labels = batch
        #imshow(torchvision.utils.make_grid(images, normalize=True))

        images = fgsm_attack(model, criterion, images, labels, 0).to(device)
        #imshow(torchvision.utils.make_grid(images, normalize=True))

        with torch.no_grad():
            logits = model(images.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        test_loss.append(loss.item())
        test_accs.append(acc)
        # break

    # The average loss and accuracy for entire testing set is the average of the recorded values.
    test_loss = sum(test_loss) / len(test_loss)
    test_acc = sum(test_accs) / len(test_accs)

    # Print the information.
    print(
        f"[ Test | {epoch + 1:03d}/{n_epochs:03d} ] loss = {test_loss:.5f}, acc = {test_acc:.5f}"
    )


    # if not testing, save the last epoch
    if len(test_loader) == 0:
        torch.save(model.state_dict(), model_path)
        print("saving model at last epoch")

# %%
# get reports
from sklearn.metrics import classification_report, accuracy_score, hamming_loss

model_best = Net().to(device)
model_best.load_state_dict(torch.load(model_path))
model_best.eval()

label_pred = []
label_true = []

with torch.no_grad():
    for data, labels in tqdm(test_loader):
        test_pred = model_best(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        if len(test_label) > 1 and len(labels) > 1:
            label_pred += test_label.squeeze().tolist()
            label_true += labels.squeeze().tolist()
        else:
            label_pred += test_label.tolist()
            label_true += labels.tolist()

    report = classification_report(
        label_true, label_pred, labels=[i for i in range(1, 41)], zero_division=0
    )
    print(report)

    test_acc = accuracy_score(label_true, label_pred)
    test_loss = hamming_loss(label_true, label_pred)

    # Print the information.
    print(f"loss = {test_loss:.5f}, acc = {test_acc:.5f}")

    with open("".join(model_path[2:].split(".")[:-1]) + "_report.txt", "w") as f:
        f.write(report)
        f.write(f"\nacc: {test_acc}\nloss: {test_loss}\n")
