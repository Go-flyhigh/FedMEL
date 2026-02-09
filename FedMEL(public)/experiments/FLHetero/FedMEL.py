import argparse
import copy
import csv
import logging
import random
from collections import OrderedDict, defaultdict
from pathlib import Path
import numpy as np
import torch
from tqdm import trange

from experiments.FLHetero.Blocks import build_fused_state_dict, load_fused_weights_into_heteros
from experiments.FLHetero.Models.CNNs import (CNN_1_large, CNN_2_large, CNN_3_large, CNN_4_large, CNN_5_large)
from experiments.FLHetero.node import BaseNodes
from experiments.utils import get_device, set_logger, set_seed, str2bool

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"

random.seed(2022)


def get_dataset_config(data_name):
    """
    Get dataset configuration based on data name
    
    Returns:
        num_classes: number of classes
        input_size: input image size
    """
    if data_name == 'cifar10':
        return 10, 32
    elif data_name == 'cifar100':
        return 100, 32
    elif data_name == 'tiny-imagenet':
        return 200, 64
    elif data_name in ['mnist', 'fashion-mnist']:
        return 10, 28
    else:
        raise ValueError(f"Unsupported dataset: {data_name}")

torch.autograd.set_detect_anomaly(True)

def test_acc_large_single(net_large, testloader,criteria):
    net_large.eval()
    with torch.no_grad():
        test_acc = 0
        num_batch = 0

        for batch in testloader:
            num_batch += 1
            # batch = next(iter(testloader))

            img, label = tuple(t.to(device) for t in batch)

            large_pred, _ = net_large(img)
            pred = large_pred

            test_loss = criteria(pred, label)
            test_acc += pred.argmax(1).eq(label).sum().item() / len(label)
        mean_test_loss = test_loss / num_batch
        mean_test_acc = test_acc / num_batch
    return mean_test_loss, mean_test_acc

class Server_Blocks(torch.nn.Module):
    def __init__(self,convblock, linearblock): 
        super().__init__()  
        self.conv_block = convblock
        self.linear_block = linearblock


def train(data_name: str, data_path: str, classes_per_node: int, num_nodes: int, fraction: float,
          steps: int, epochs: int, optim: str, lr: float, inner_lr: float,
          embed_lr: float, wd: float, inner_wd: float, embed_dim: int, hyper_hid: int,
          n_hidden: int, n_kernels: int, bs: int, device, eval_every: int, save_path: Path,
          seed: int,LowProb:float) -> None:

    num_classes, input_size = get_dataset_config(data_name)
    logging.info(f"Dataset config: {data_name}, num_classes={num_classes}, input_size={input_size}x{input_size}")

    ###############################
    # init nodes, hnet, local net #
    ###############################
    nodes = BaseNodes(data_name, data_path, num_nodes, classes_per_node=classes_per_node,
                      LowProb=LowProb, batch_size=bs)

    # -------compute aggregation weights-------------#
    train_sample_count = nodes.train_sample_count
    eval_sample_count = nodes.eval_sample_count
    test_sample_count = nodes.test_sample_count

    client_sample_count = [train_sample_count[i] + eval_sample_count[i] + test_sample_count[i] for i in
                           range(len(train_sample_count))]   
    # -----------------------------------------------#

    print(data_name)
    in_channels = 1 if data_name in ["mnist", "fashion-mnist"] else 3
    if data_name in ["cifar10", "cifar100", "tiny-imagenet", "mnist", "fashion-mnist"]:
        net_1 = CNN_1_large(in_channels=in_channels, n_kernels=n_kernels, out_dim=num_classes, input_size=input_size)
        net_2 = CNN_2_large(in_channels=in_channels, n_kernels=n_kernels, out_dim=num_classes, input_size=input_size)
        net_3 = CNN_3_large(in_channels=in_channels, n_kernels=n_kernels, out_dim=num_classes, input_size=input_size)
        net_4 = CNN_4_large(in_channels=in_channels, n_kernels=n_kernels, out_dim=num_classes, input_size=input_size)
        net_5 = CNN_5_large(in_channels=in_channels, n_kernels=n_kernels, out_dim=num_classes, input_size=input_size)
    else:
        raise ValueError("choose data_name from ['cifar10', 'cifar100', 'tiny-imagenet', 'mnist', 'fashion-mnist']")
    #assign the nets to device "GPU"
    net_1 = net_1.to(device)
    net_2 = net_2.to(device)
    net_3 = net_3.to(device)
    net_4 = net_4.to(device)
    net_5 = net_5.to(device)
    net_set = [net_1, net_2, net_3, net_4, net_5]

    ##################
    # init optimizer #  
    ##################

    criteria = torch.nn.CrossEntropyLoss()

    ################
    # init metrics #
    ################
    step_iter = trange(steps)

    PM_large_acc = defaultdict()
    PM_large = defaultdict()

    for i in range(num_nodes):
        PM_large_acc[i] = 0
        PM_large[i] = copy.deepcopy(net_set[i%5].state_dict())

    server_fused = build_fused_state_dict(
        Server_Blocks(
            net_set[0].Block1,
            net_set[0].Block2,
        )
    )
    def _fused_to_device(fused_sd: dict, device):
        return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in fused_sd.items()}

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    orininal_file = str(save_path / f"Hetero_FedMEL_N{num_nodes}_C{fraction}_R{steps}_E{epochs}_B{bs}_NonIID_{data_name}"
                                    f"_class_{classes_per_node}_low{LowProb}.csv")
    with open(orininal_file, 'w', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')

        for step in step_iter:  # step is round
            frac = fraction
            select_nodes = random.sample(range(num_nodes), int(frac * num_nodes))

            large_local_trained_acc = []
            results = []
            single_large_acc = []
            LNs = defaultdict() # collect adapter-only fused weights for aggregation
            fused_on_dev = _fused_to_device(server_fused, device)
            logging.info(f'#----Round:{step}----#') 
            for c in select_nodes:
                node_id = c

                net_large = net_set[node_id % 5]
                net_large.load_state_dict(PM_large[node_id])
                
                if fused_on_dev is not None:
                    load_fused_weights_into_heteros(
                        Server_Blocks(net_large.Block1, net_large.Block2),
                        fused_on_dev,
                    )

                optimizer_large = torch.optim.SGD(params=net_large.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

                for i in range(epochs):
                    net_large.train()
                    for j, batch in enumerate(nodes.train_loaders[node_id]):
                        img, label = tuple(t.to(device) for t in batch)

                        large_pred, _ = net_large(img)
                        loss = criteria(large_pred, label)

                        optimizer_large.zero_grad()

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net_large.parameters(), 50)

                        optimizer_large.step()


                # collect local NN parameters
                PM_large[node_id] = copy.deepcopy(net_large.state_dict())
                net_large.eval()
                # collect local NN parameters
                # Upload fused adapter weights only to keep hetero backbones personalized.
                LNs[node_id] = build_fused_state_dict(
                    Server_Blocks(
                        net_large.Block1,
                        net_large.Block2,
                    )
                )
                # evaluate trained local model
                _, trained_acc = test_acc_large_single(net_large, nodes.test_loaders[node_id], criteria)
                large_local_trained_acc.append(trained_acc)
                PM_large_acc[node_id] = trained_acc

            mean_large_trained_acc = round(np.mean(large_local_trained_acc), 4)

            results = [mean_large_trained_acc]

            print(f'Round {step} | Mean Large Acc: {mean_large_trained_acc}')


            mywriter.writerow(results)
            file.flush()

            client_agg_weights = OrderedDict()
            select_nodes_sample_count = OrderedDict()
            for i in range(len(select_nodes)):
                select_nodes_sample_count[select_nodes[i]] = client_sample_count[select_nodes[i]]
            for i in range(len(select_nodes)):
                client_agg_weights[select_nodes[i]] = select_nodes_sample_count[select_nodes[i]] / sum(select_nodes_sample_count.values())

            weight_keys = list(next(iter(LNs.values())).keys())

            server_fused_next = OrderedDict()
            for key in weight_keys:
                key_sum = 0
                for id, model in LNs.items():
                    key_sum += client_agg_weights[id] * model[key]
                server_fused_next[key] = key_sum   
            # Server keeps the fused weights; no broadcast to all clients
            server_fused = server_fused_next
            logging.info("Blocks updated; clients will pull it when selected next round.")

        logging.info('Federated Learning has been successfully!')

    new_file = str(save_path / f"Done_Hetero_FedMEL_N{num_nodes}_C{fraction}_R{steps}_E{epochs}_B{bs}_NonIID_{data_name}"
                               f"_class_{classes_per_node}_low{LowProb}.csv")
    os.rename(orininal_file, new_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Federated Learning with Lookahead experiment"
    )

    #############################
    #       Dataset Args        #
    #############################

    parser.add_argument(
        "--data-name", type=str, default="cifar100", choices=['cifar10', 'tiny-imagenet','cifar100', 'mnist'], help="dir path for MNIST dataset"
    )
    parser.add_argument("--data-path", type=str, default="data", help="dir path for MNIST dataset")
    parser.add_argument("--num-nodes", type=int, default=100, help="number of simulated nodes")
    parser.add_argument("--fraction", type=float, default=0.1, help="number of sampled nodes in each round")

    ##################################
    #       Optimization args        #
    ##################################

    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--optim", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")
    parser.add_argument("--batch-size", type=int, default=512) # 512
    parser.add_argument("--epochs", type=int, default=10, help="number of inner steps")

    ################################
    #       Model Prop args        #
    ################################
    parser.add_argument("--n-hidden", type=int, default=3, help="num. hidden layers")
    parser.add_argument("--inner-lr", type=float, default=5e-3, help="learning rate for inner optimizer")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--inner-wd", type=float, default=5e-3, help="inner weight decay")
    parser.add_argument("--embed-dim", type=int, default=-1, help="embedding dim")
    parser.add_argument("--embed-lr", type=float, default=None, help="embedding learning rate")
    parser.add_argument("--hyper-hid", type=int, default=100, help="hypernet hidden dim")
    parser.add_argument("--spec-norm", type=str2bool, default=False, help="hypernet hidden dim")
    parser.add_argument("--nkernels", type=int, default=16, help="number of kernels for cnn model")
    #############################
    #       General args        #
    #############################
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval-every", type=int, default=30, help="eval every X selected epochs")
    parser.add_argument("--save-path", type=str, default="Results/FedMEL", help="dir path for output file")
    parser.add_argument("--seed", type=int, default=42, help="seed value")

    args = parser.parse_args()
    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    set_logger()
    set_seed(args.seed)

    device = get_device(gpus=args.gpu)

    if args.data_name == 'cifar10':
        args.classes_per_node = 2
        args.LowProb = 0.4
    elif args.data_name == 'cifar100':
        args.classes_per_node = 10
        args.LowProb = 0.4
    elif args.data_name == 'tiny-imagenet':
        args.classes_per_node = 20
        args.LowProb = 0.4
    else:
        args.classes_per_node = 2
    train(
        data_name=args.data_name,
        data_path=args.data_path,
        classes_per_node=args.classes_per_node,
        num_nodes=args.num_nodes,
        fraction=args.fraction,
        steps=args.num_steps,
        epochs=args.epochs,
        optim=args.optim,
        lr=args.lr,
        inner_lr=args.inner_lr,
        embed_lr=args.embed_lr,
        wd=args.wd,
        inner_wd=args.inner_wd,
        embed_dim=args.embed_dim,
        hyper_hid=args.hyper_hid,
        n_hidden=args.n_hidden,
        n_kernels=args.nkernels,
        bs=args.batch_size,
        device=device,
        eval_every=args.eval_every,
        save_path=args.save_path,
        seed=args.seed,
        LowProb=args.LowProb,
    )
