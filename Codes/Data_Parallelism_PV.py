import os
import time
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import torch.multiprocessing as mp
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.nn.pytorch import ChebConv, GATConv


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        torch.manual_seed(42)

        # self.conv1 = GATConv(7200, 3600, 1)
        # self.conv2 = GATConv(3600, 7200, 1)
        # self.conv3 = GATConv(7200, 3600, 1)
        # self.conv4 = GATConv(3600, 7200, 1)
        self.conv1 = GATConv(120, 60, 1)
        self.conv2 = GATConv(60, 120, 1)
        self.conv3 = GATConv(120, 60, 1)
        self.conv4 = GATConv(60, 120, 1)

    def forward(self, blocks, x):
        h_1 = self.conv1(blocks[0], x)
        h_a = self.conv2(blocks[1], h_1)
        h_2 = self.conv3(blocks[0], x)
        h_f = self.conv4(blocks[1], h_2)

        return h_a, h_f




def run(proc_id, devices, dgl_graph):
    # Initialize distributed training context.
    dev_id = devices[proc_id]
    dist_init_method = "tcp://{master_ip}:{master_port}".format(
        master_ip="127.0.0.1", master_port="12345"
    )
    if torch.cuda.device_count() < 1:
        device = torch.device("cpu")
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=dist_init_method,
            world_size=len(devices),
            rank=proc_id,
        ) 
    else:
        torch.cuda.set_device(dev_id)
        device = torch.device("cuda:" + str(dev_id))
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=dist_init_method,
            world_size=len(devices),
            rank=proc_id,
        )

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

    dataloader = dgl.dataloading.DataLoader(
        dgl_graph, dgl_graph.nodes().long(), sampler,
        device=device,
        use_ddp=True,
        batch_size=25,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    #     train_dataloader = dgl.dataloading.DataLoader(
    #     # The following arguments are specific to DataLoader.
    #     graph,  # The graph
    #     train_nids,  # The node IDs to iterate over in minibatches
    #     sampler,  # The neighbor sampler
    #     device=device,  # Put the sampled MFGs on CPU or GPU
    #     use_ddp=True,  # Make it work with distributed data parallel
    #     # The following arguments are inherited from PyTorch DataLoader.
    #     batch_size=1024,  # Per-device batch size.
    #     # The effective batch size is this number times the number of GPUs.
    #     shuffle=True,  # Whether to shuffle the nodes for every epoch
    #     drop_last=False,  # Whether to drop the last incomplete batch
    #     num_workers=0,  # Number of sampler processes
    # )


    model = Model().to(device)
    # Wrap the model with distributed data parallel module.
    if device == torch.device("cpu"):
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=None, output_device=None
        )
    else:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], output_device=device
        )


    opt = torch.optim.Adam(model.parameters())
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    m1 = [[1] for i in range(1,121)]
    m2 = [[i] for i in range(1,121)]
    m = torch.Tensor([m1, m2])
    m = torch.squeeze(m).T
    
    
    K = 12
    d = [ [ 0 for y in range(120) ] for x in range(10) ]
    for i in range(10):
      for j in range(120):
        if int(j/K) == i:
          d[i][j] = 1/K
    E = torch.as_tensor(d)
    
    W = [ [ 1 for y in range(10) ] for x in range(10) ]
    for i in range(10):
      for j in range(10):
        W[i][j] = abs(j-i)
    W = torch.as_tensor(W)
    W = W.fill_diagonal_(1)
    
    
    I = [ 1 for x in range(10) ]
    I = torch.as_tensor(I)
    L = torch.diag(torch.matmul(W, I), 0) - W


    # m1 = [[1] for i in range(1,7201)]
    # m2 = [[i] for i in range(1,7201)]
    # m = torch.Tensor([m1, m2])
    # m = torch.squeeze(m).T
    
    
    # K = 720
    # d = [ [ 0 for y in range(7200) ] for x in range(10) ]
    # for i in range(10):
    #   for j in range(7200):
    #     if int(j/K) == i:
    #       d[i][j] = 1/K
    # E = torch.as_tensor(d)
    
    # W = [ [ 1 for y in range(10) ] for x in range(10) ]
    # for i in range(10):
    #   for j in range(10):
    #     W[i][j] = abs(j-i)
    # W = torch.as_tensor(W)
    # W = W.fill_diagonal_(1)
    
    
    # I = [ 1 for x in range(10) ]
    # I = torch.as_tensor(I)
    # L = torch.diag(torch.matmul(W, I), 0) - W


    # m = m.to(h_f.device)
    # E = E.to(h_f.device)
    # L = L.to(h_f.device)

    best_model_path = "st_DynGNN_best.pt"
    best_loss = 0
    start_time = time.time()

    # Copied from previous tutorial with changes highlighted.
    for epoch in range(100):
        model.train()
        losses = []
        with tqdm.tqdm(dataloader) as tq:
            for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                # print(device)
                mfgs = [b for b in mfgs]
                input_features = mfgs[0].srcdata['x']
                ground_truth = mfgs[1].dstdata['x']
                h_a, h_f = model(mfgs, input_features)
                # print(h_a)
                m = m.to(h_f.device)
                E = E.to(h_f.device)
                L = L.to(h_f.device)
                loss = criterion(torch.squeeze(h_a)+torch.squeeze(h_f), ground_truth)  + \
                10*torch.std(torch.diff(torch.squeeze(h_a))) +\
                0 * torch.var(torch.squeeze(h_a)) +\
                100 * torch.sum(torch.abs(torch.matmul(torch.matmul(torch.linalg.inv(torch.matmul(torch.transpose(m, 0, 1), m)), torch.transpose(m, 0, 1)), torch.transpose(torch.squeeze(h_f), 0, 1))[1])) +\
                5 * torch.sum(torch.diagonal(torch.matmul(torch.matmul(torch.matmul(torch.squeeze(h_f), E.T).float(), L.float()), torch.matmul(E, torch.squeeze(h_f).T).float()), 0))
                # loss = criterion(torch.squeeze(h_a)+torch.squeeze(h_f), ground_truth)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        epoch_loss = np.mean(losses)

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")


    
   
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Training time: {elapsed_time} seconds")
    model.eval()
    torch.save(model.state_dict(), best_model_path)

