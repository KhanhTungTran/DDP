# Distributed data parallel for Neural network training

This is a basic repository for demonstration on how to take advantage of multi-machines for distributed data parallel (DDP) training on a Neural network.

The implementation is based on *Pytorch*, specificially the module *torch.nn.parallel.DistributedDataParallel*: 
>"This container parallelizes the application of the given module by splitting the input across the specified devices by chunking in the batch dimension. The module is replicated on each machine and each device, and each such replica handles a portion of the input. During the backwards pass, gradients from each node are averaged."

## Dependencies
```
pip install -r requirements.txt
```

## Usage

Example: using 4 different machines (or CPU cores):

On node 0:
```
python train.py --nodes 2 --nr 0
```
Then, on the other nodes:
```
python train.py --nodes 2 --nr i
```
for i∈1,2,3. In other words, we run this script on each node, telling it to launch args.cpus processes that sync with each other before training begins.

## Project status and Roadmap
With the current setup, please clear the content of the file setup.txt before re-running

**TODO list:**
- Create a GUI to display loss value
- Make it work on an network

## Author
Current author: [Tran Khanh Tung](https://github.com/KhanhTungTran)

## References
- [Distributed data parallel training in Pytorch](https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html)
- [DISTRIBUTED DATA PARALLEL](https://pytorch.org/docs/master/notes/ddp.html)
