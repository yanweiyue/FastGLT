import torch
import torch.nn.functional as F
from args import parser_loader


import util
import copy
import warnings
from pruning import FastScheduler


from ogb.nodeproppred import PygNodePropPredDataset,Evaluator
from torch_geometric.utils import to_undirected, add_self_loops
from model import DeeperGCN

import time


warnings.filterwarnings('ignore')

@torch.no_grad()
def test(model, x, edge_index, y_true, split_idx, evaluator):
    model.eval()
    out = model(x, edge_index)

    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def train(net_gcn, x, edge_index, y_true, train_idx, optimizer,pruner,epoch, args):

    net_gcn.train()
    optimizer.zero_grad()
    pred = net_gcn(x, edge_index)[train_idx]
    if args["spar_adj"]:
        net_gcn.adj_mask1_train.retain_grad()
    loss = F.nll_loss(pred, y_true.squeeze(1)[train_idx])
    loss.backward()
    if epoch >= args['pretrain_epoch'] and epoch <args['pretrain_epoch'] + args['total_epoch']:
        if args["spar_adj"]:
            net_gcn.adj_mask1_train.grad.data.add_(1e-4* torch.sign(net_gcn.adj_mask1_train.data))  #r1 regularization
        if pruner():  # grow and cut step
            optimizer.step()
    else:
        optimizer.step()
    return loss.item()



def run_get_mask(args):
    rewind_weight=None
    device = args['device']
    
    dataset = PygNodePropPredDataset(name=args["dataset"])
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(args["dataset"])

    x = data.x.to(device)
    y_true = data.y.to(device)
    train_idx = split_idx['train'].to(device)

    edge_index = data.edge_index.to(device)
    edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)


    edge_index = add_self_loops(edge_index, num_nodes=data.num_nodes)[0]

    args["in_channels"] = data.x.size(-1)
    args["num_tasks"] = dataset.num_classes
    
    
    net_gcn = DeeperGCN(args,edge_index,data.num_nodes).to(device)
    
    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    
    acc_test = 0.0
    best_val_acc = {'val_acc': 0, 'epoch': 0, 'test_acc': 0, "adj_spar": 0, "wei_spar": 0}
    best_mask = None
    mask_ls = []
    
    rewind_weight = copy.deepcopy(net_gcn.state_dict())
    pruner = lambda:False
    last_acc_val = None
    
    start_time = time.time()
    for epoch in range(args['total_epoch']+args['pretrain_epoch']+args['retrain_epoch']):
        if epoch == args['pretrain_epoch']:
            print("========================================================")
            print("pretrain finish")
            print("========================================================")
            pruner = lambda: True
            if args["remain"] is not None:
                if args['spar_adj']:
                    util.add_trainable_mask_noise(net_gcn,c=1e-4)
                T_end = args["total_epoch"]
                pruner = FastScheduler(net_gcn, optimizer, remain=args["remain"], alpha=args["alpha"], delta=args["delta"], static_topo=args["static_topo"], T_end=T_end,  accumulation_n=args["accumulation_n"],ignore_linear_layers=not args["spar_wei"],
                                   pretrain=args['pretrain_epoch'],ignore_parameters=not args['spar_adj'],beta=args['beta'],warmup_steps=args["warmup_steps"])
                
        
        if epoch == args["total_epoch"]+args['pretrain_epoch']:
            print("========================================================")
            print("retrain start")
            print("========================================================")
            with torch.no_grad():
                if args['spar_adj']:
                    for gcn in net_gcn.gcns:
                        gcn.adj_mask1_train =None
                net_gcn.load_state_dict(rewind_weight)
                pruner.backward_masks = best_mask
        
        
        loss=train(net_gcn, x, edge_index, y_true, train_idx, optimizer,pruner,epoch, args)
        result = test(net_gcn, x, edge_index, y_true, split_idx, evaluator)
        acc_train, acc_val, acc_test = result
        
        
        with torch.no_grad():
            if epoch<args['pretrain_epoch']:
                wspar_here,aspar_here = (0,0)
            else:
                wspar_here,aspar_here = pruner.getS()
            
            meet = ((args['spar_adj'] and aspar_here >= ((1-args["remain"])*100-1)) or not args['spar_adj']) and \
                    ((args['spar_wei'] and wspar_here >= ((1-args["remain"])*100-1)) or not args['spar_wei']) and \
                        epoch >= args['pretrain_epoch']
            
            if acc_val > best_val_acc['val_acc'] and meet:
                best_val_acc['test_acc'] = acc_test
                best_val_acc['val_acc'] = acc_val
                best_val_acc['epoch'] = epoch
                best_mask = copy.deepcopy(pruner.backward_masks)
                best_val_acc['wei_spar'] = wspar_here 
                best_val_acc['adj_spar'] = aspar_here
                if epoch<args["pretrain_epoch"]+args["total_epoch"]:
                    best_val_acc['time'] = time.time() - start_time

            print("Epoch:[{}] L:[{:.3f}] Train:[{:.2f}] Val:[{:.2f}] Test:[{:.2f}] WS:[{:.2f}%] AS:[{:.2f}%] time:[{:.2f}]|"
                  .format(epoch, loss, acc_train * 100, acc_val * 100, acc_test * 100, wspar_here, aspar_here,time.time()-start_time), end=" ")
            if meet:
                print("Best Val:[{:.2f}] Test:[{:.2f}] AS:[{:.2f}%] WS:[{:.2f}%] at Epoch:[{}] time:[{:.2f}]"
                      .format(
                          best_val_acc['val_acc'] * 100,
                          best_val_acc['test_acc'] * 100,
                          best_val_acc['adj_spar'],
                          best_val_acc['wei_spar'],
                          best_val_acc['epoch'],
                          best_val_acc['time']))
            else:
                print("")
            
    
    return best_mask



if __name__ == "__main__":
    args = parser_loader()
    print(args)
    util.fix_seed(args['seed'])
    best_mask = run_get_mask(args)
    
