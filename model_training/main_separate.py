import os

from torch.utils.data import Dataset, random_split, DataLoader
import torch.optim as optim
import torch_optimizer as optimx
from s4torch import S4Model
from torch.cuda.amp import GradScaler, autocast

from model_separate import *
from train_single import *


class RawImageDataset(Dataset):
    def __init__(self, root_dir, dtype=np.uint16,  transform=None, mean=0.0, std=1):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = os.listdir(root_dir)
        self.dtype = dtype
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.root_dir, file_name)

        z, x, y = file_name[:-4].split('x')[-3:]
        x = int(x.split('_')[0])
        y = int(y)
        z = int(z.split('-')[-1])

        with open(file_path, 'rb') as f:
            raw_data = f.read()

        if "u16be" in file_name:
            dtype = ">u2"
        elif "u16le" in file_name:
            dtype = "<u2"
        elif "s16be" in file_name:
            dtype = ">i2"
        elif "s16le" in file_name:
            dtype = "<i2"
        else:
            raise ValueError("Error: No data type recognized")

        image_data = np.frombuffer(raw_data, dtype=dtype)
        image = image_data.reshape((z, x * y))
        image = image.transpose().astype("float")

        if self.transform:
            image = self.transform(image_data, x, y, z)

        return torch.from_numpy(image)

def custom_collate_fn(batch):
    concatenated_batch = torch.cat(batch, dim=0)
    return concatenated_batch




class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, data):
        #data = torch.round(torch.cat(data, dim=1))
        data = torch.round(data).int()

        unique_classes, counts = torch.unique(data, return_counts=True)
        total_sum = torch.sum(counts, dtype=torch.float)

        probabilities = counts / total_sum
        aux = probabilities * torch.log2(probabilities)
        entropy = -torch.sum(aux)

        return entropy




def main(num_epochs, level, input_size, criterion, device, train_loader, val_loader, hidden_size, lr,
         optimizer,  mode, block, n, dropout, wd, bn, pretrained):

    entropy_criterion = EntropyLoss()
    l = int(np.ceil(np.log2(input_size)))

    best_entropy = 10000000

    seed = 42
    torch.manual_seed(seed)

    s = input_size
    model_list = []
    optimizer_list = []
    scaler_list = []
    for i in range(level):

        p = int(np.round(s / 2))
        q = int(np.floor(s / 2))

        if p % 2 == 0 and abs(p - (s/2)) > 0.4 and p == q:
            p += 1

        if pretrained:
            model_list = torch.jit.load(mode).to(device)

        else:
            if mode == "MLP":
                model_list = SimpleFcModel(p, hidden_size, q, device=device, dropout=dropout, use_batch_norm=bn).float().to(device)

            elif mode == "S4":
                model_list = S4Model(1, d_model=128, d_output=1, n_blocks=block, n=n, l_max=p, collapse=False).to(device)

        scaler_list.append(GradScaler())

        optimizer_list.append(optimizer(model_list.parameters(), lr=lr))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_list[-1], mode="min", patience=3, factor=0.5, min_lr=0.00000001)
        s = p



    model_list, entropy, epochs = train_single(model_list, criterion, optimizer_list, num_epochs, train_loader, val_loader, entropy_criterion,
                                l, device, scaler_list, level, scheduler, mode)


    print("Entropy: ", entropy, "\t Hyperparameters:", lr, hidden_size, wd, bn, dropout, optimizer,  "\t Num epochs: ", epochs)
    if entropy < best_entropy:
        if mode == "S4":
            traced_model = torch.jit.trace(model_list, torch.rand(512, p, 1).to(device))
        elif mode == "MLP":
            traced_model = torch.jit.trace(model_list, torch.rand(512, p).to(device))
        traced_model.save(f'Hyperion_model_{p}.pth')


if __name__ == "__main__":

    num_epochs = 2
    level = 1
    input_size = 242
    lr = 0.01

    optimizer = optim.Adam
    criterion = nn.L1Loss()

    dataset_directory = "./../../Data/Hyperion"


    # Options for mode:
    #    - "S4"
    #    - "MLP"
    mode = "MLP"

    pretrained = False  # If ture pretrained = path

    # Hyperparameters for S4Model
    block = 4
    n = 64


    # Hyperparameters for MLP
    hidden_size = 1024
    dropout = 0
    wd = 0
    bn = True

    train_size = 2
    val_size = 0

    device = "cuda"

    seed = 42
    torch.manual_seed(seed)

    full_dataset = RawImageDataset(root_dir=dataset_directory, transform=None)


    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
    #val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)
    val_loader = train_loader

    main(num_epochs, level, input_size, criterion, device, train_loader, val_loader, hidden_size, lr,
         optimizer, mode, block, n, dropout, wd, bn, pretrained)