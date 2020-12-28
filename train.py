from datetime import datetime
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import argparse
from model import NeuralNet
import torchvision
import torchvision.transforms as transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', default=2, type=int, metavar='N',\
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--cpus', default=1, type=int,
                        help='number of cpus per node')
    parser.add_argument('--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    args.world_size = args.cpus * args.nodes
    mp.spawn(train, nprocs=args.cpus, args=(args,))


def train(cpu, args):
    rank = args.nr * args.cpus + cpu
    dist.init_process_group(
        backend="gloo",
        init_method="file:///C:/Users/Tung/Desktop/HK201/CN/Extra/setup.txt",
        world_size=args.world_size,
        rank=rank
    )
    torch.manual_seed(0)
    
    # Hyperparameters:
    batch_size = 100 # NOTE: If ran out of memory, try changing this value to 64 or 32
    learning_rate = 0.0001

    # Create model:
    model = NeuralNet()
    # Define loss function and optimizer:
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), learning_rate)
    # Wrap the model for ddp:
    model = nn.parallel.DistributedDataParallel(model, device_ids=None)
    # Data loading:
    train_dataset = torchvision.datasets.MNIST(
        root = './data',
        train=True,
        transform=transforms.ToTensor(),
        download=True)

    # Sampling the dataset to avoid same inputs order:
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank = rank
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler
    )

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass:
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward pass:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # For logging:
            if (i + 1) % batch_size and cpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format\
                    (epoch + 1, args.epochs, i + 1, total_step,loss.item()))
    
    if cpu == 0:
        print("Training completed in: " + str(datetime.now() - start))

if __name__ == "__main__":
    main()
