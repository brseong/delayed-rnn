import torch 
from torchvision import datasets, transforms

class SequentialMNIST:
    def __init__(self, permute: bool, input_size: int, train: bool, seed: int):
        self.permute: bool = permute
        self.seed: int = seed
        
        self.mnist = datasets.MNIST(root="datasets/data", train=train, download=True, transform=transforms.ToTensor())
        
        self.input_size: int = input_size
        self.num_classes = 10 
        self.is_classification = True
        
        if self.permute:
            g = torch.Generator()
            g.manual_seed(self.seed + 2)
            self.permutation = torch.randperm(784, generator=g)
            self.dataset_name: str = "psMNIST"
        else:
            self.dataset_name: str = "sMNIST"

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        image = image.view(-1) # (1, 28, 28) -> (784,)
        if self.permute:
            image = image[self.permutation]
        image = image.unsqueeze(-1)
        return image, torch.tensor(label, dtype=torch.long)

    def train(self, model, dataloader, optimizer, device):
        model.train()
        correct: int = 0
        total_loss: float = 0.0
        
        for _, batch in enumerate(dataloader):
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            out = model(x = inputs, train = True)
            loss, num_correct = model.compute_loss(out = out, targets = targets, optimizer = optimizer)
            total_loss += loss.item()
            correct += num_correct
            
        accuracy = correct / len(dataloader.dataset)
        return total_loss, accuracy
    
    @torch.inference_mode()
    def eval(self, model, dataloader, device):
        model.eval()
        correct: int = 0
        total_loss: float = 0.0
        
        for _, batch in enumerate(dataloader):
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            out = model(x = inputs, train = False)
            loss, num_correct = model.compute_loss(out = out, targets = targets)
            total_loss += loss.item()
            correct += num_correct
            
        accuracy = correct / len(dataloader.dataset)
        return total_loss, accuracy