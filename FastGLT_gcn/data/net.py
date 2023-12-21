import torch
import torch.nn as nn
import pdb
import copy
import math
import utils

# from torch_geometric.nn import GCNConv
from layers import GCNConv, BinaryStep, MaskedLinear
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_adj, to_undirected

import matplotlib.pyplot as plt


# class net_gcn(nn.Module):

#     def __init__(self, embedding_dim, device, spar_wei, spar_adj, mode="prune"):
#         super().__init__()

#         self.mode = mode # prune or retain

#         self.layer_num = len(embedding_dim) - 1
#         # self.net_layer = nn.ModuleList([nn.Linear(embedding_dim[ln], embedding_dim[ln+1], bias=False) for ln in range(self.layer_num)])
#         self.net_layer = nn.ModuleList([  GCNConv(embedding_dim[ln], embedding_dim[ln+1]) for ln in range(self.layer_num) ])
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(p=0.5)
#         self.normalize = utils.torch_normalize_adj

#         self.spar_wei = spar_wei
#         self.spar_adj = spar_adj
#         self.edge_mask_archive = []

#         self.device = device

#         if self.spar_adj:
#             self.adj_thresholds = nn.ParameterList(
#             [nn.Parameter(torch.Tensor([-1])) for i in range(self.layer_num)])  # 0.03
    

#         ### PGExplainer
#         self.edge_learner = nn.Sequential(
#             nn.Linear(embedding_dim[0] * 2, 2048),
#             nn.ReLU(),
#             # nn.Linear(1024, 256),
#             # nn.ReLU(),
#             nn.Linear(2048,1)
#         )
#         # self.edge_learner2 = nn.Sequential(
#         #     nn.Linear(embedding_dim[1] * 2, 1024),
#         #     nn.ReLU(),
#         #     nn.Linear(1024,1)
#         # )

#         # NodeFormer
#         self.Wk = nn.Linear(embedding_dim[0], 128)
#         self.Wq = nn.Linear(embedding_dim[0], 128)
#         self.Wv = nn.Linear(embedding_dim[0], 128)
#         self.Wo = nn.Linear(4, 1)

#         ### UGS soft mask
#         # self.edge_mask = nn.Parameter(torch.ones(size=(edge_index.shape[1],), requires_grad=True))
#         # self.edge_mask.requires_grad = False
#         # rand = (2 * torch.rand(self.edge_mask.shape) - 1) * 1e-5
#         # rand = rand.to(self.edge_mask.device) 
#         # rand = rand * self.edge_mask
#         # self.edge_mask.add_(rand)
#         # self.edge_mask.requires_grad = True

#         ### Scalabel GNNGuard
#         self.sim_learner_src = nn.Linear(embedding_dim[0], 512)
#         self.sim_learner_tgt = nn.Linear(embedding_dim[0], 512)
#         self.mid_learner = nn.Parameter( nn.init.xavier_uniform(torch.empty(512, 512)), requires_grad=True)
        

    
#     def __create_learner_input(self, edge_index, embeds):
#         row, col = edge_index[0], edge_index[1]
#         row_embs, col_embs = embeds[row], embeds[col]
#         # node_emb = embeds[node_id].repeat(row.size(0), 1)
#         edge_learner_input = torch.cat([row_embs, col_embs], 1)
#         return edge_learner_input

    
#     def forward_retain(self, x, edge_index, val_test, edge_masks):
#         # print(edge_index[0].shape)
#         # print(edge_index[1].shape)
#         # print(edge_masks[0].requires_grad)
#         # print(edge_masks[0])
#         assert len(edge_index) == self.layer_num
#         for ln in range(self.layer_num):
#             x = self.net_layer[ln](x, edge_index, edge_mask=edge_masks[ln])
#             if ln == self.layer_num - 1:
#                 break
#             x = F.relu(x)
#             if val_test:
#                 continue
#             x = self.dropout(x)
#         return x


#     def forward(self, x, edge_index, val_test=False, **kwargs):

#         if self.mode == "retain":
#             return self.forward_retain(x, edge_index, val_test, kwargs['edge_masks'])
#         if self.spar_adj:
#             self.edge_weight = edge_weight = self.learn_soft_edge2(x, edge_index)
#             # edge_weight = self.att_coef3(x, edge_index)
#             edge_mask = None
#             self.edge_mask_archive = []
#         else:
#             # edge_weight = self.att_coef3(x, edge_index)
#             self.edge_weight = edge_weight = self.learn_soft_edge2(x, edge_index)
        
#         for ln in range(self.layer_num):
#             if self.spar_adj and kwargs['pretrain']:
#                 x = self.net_layer[ln](x, edge_index, edge_mask=edge_weight)
#             elif self.spar_adj:
#                 edge_mask = self.adj_pruning(edge_index, edge_weight, self.adj_thresholds[ln], edge_mask)
#                 # edge_mask = self.adj_pruning(edge_index, edge_weight, self.adj_thresholds[ln])
#                 print(f"l{ln}: [{(1 - edge_mask.sum() / edge_mask.shape[0])*100 :.3f}%]", end=" | ")
#                 self.edge_mask_archive.append(copy.deepcopy(edge_mask.detach()))
#                 x = self.net_layer[ln](x, edge_index, edge_mask=(edge_mask))
#             # print(f"edge mean:[{self.edge_mask.mean():.4f}] max:[{self.edge_mask.max():.4f}] min:[{self.edge_mask.min():.4f}]")
#             else:
#                 x = self.net_layer[ln](x, edge_index, edge_mask=edge_weight)
#                 # x = self.net_layer[ln](x, edge_index)

#             if ln == self.layer_num - 1:
#                 break
#             x = F.relu(x)
#             if val_test:
#                 continue
#             x = self.dropout(x)
#         # print("")
#         # print(f"thres 1: {self.adj_thresholds[0].item():.3f} thres 2: {self.adj_thresholds[1].item():.3f}")
#         return x
    
#     # PGExplainer
#     def learn_soft_edge(self, x, edge_index, ln=0):
#         input = self.__create_learner_input(edge_index, x)
#         if ln == 0:
#             edge_weight =  self.edge_learner(input).squeeze(-1)
#         elif ln == 1:
#             edge_weight =  self.edge_learner2(input).squeeze(-1)
#         else:
#             raise NotImplementedError

#         # edge_weight = torch.sigmoid(edge_weight)
#         # out = F.relu(out)
#         # print(f"out mean:[{out.mean():.4f}] max:[{out.max():.4f}] min:[{out.min():.4f}]")

#         """row norm
#         """
#         # deg = torch.zeros(edge_index.max().item() + 1,
#         #         dtype=torch.float, device=self.device)
#         # # TODO originally '1'
#         # deg.scatter_add_(0, edge_index[0], out)
#         # out = out / (deg[edge_index[0]] + 1e-8)  # 计算每个边的权重

#         """symmetric norm
#         """
#         # row, col = edge_index
#         # num_nodes = edge_index.max().item() + 1
#         # deg = scatter_add(out, row, dim=0, dim_size=num_nodes)
#         # deg_inv_sqrt = deg.pow(-0.5)
#         # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
#         # out = deg_inv_sqrt[row] * out * deg_inv_sqrt[col]

#         """min max norm
#         """
#         # min_w = torch.min(out)
#         # max_w = torch.max(out)
#         # out = (out - min_w) / (max_w - min_w)

#         # print(f"out mean:[{out.mean():.4f}] max:[{out.max():.4f}] min:[{out.min():.4f}]")

#         print(
#             f"[BEFORE] edge mean:[{edge_weight.mean().item():.8f}] max:[{edge_weight.max().item():.8f}] min:[{edge_weight.min().item():.8f}] | ", end="")

#         # temp softmax
#         deg = torch.zeros(edge_index.max().item() + 1,
#                           dtype=torch.float, device=self.device)
#         exp_wei = torch.exp(edge_weight / 0.8) # cora 0.8 citeseer
#         deg.scatter_add_(0, edge_index[0], exp_wei)
#         # print(deg)
#         edge_weight = exp_wei / (deg[edge_index[0]] + 1e-8)  # 计算每个边的权重

#         print(
#             f"[AFTER] edge mean:[{edge_weight.mean().item():.4f}] max:[{edge_weight.max().item():.4f}] min:[{edge_weight.min().item():.4f}] std:[{edge_weight.std().item():.4f}]")
        
#         return edge_weight

#     # GNNGuard
#     def learn_soft_edge2(self, x, edge_index):
#         row, col = edge_index
#         # print(x.device)
#         row_embs, col_embs = self.sim_learner_src(x[row]), self.sim_learner_tgt(x[col])
#         edge_weight =  torch.einsum("ik,ik->i",row_embs,col_embs)
        
#         print(
#             f"[BEFORE] edge mean:[{edge_weight.mean().item():.8f}] max:[{edge_weight.max().item():.8f}] min:[{edge_weight.min().item():.8f}] | ", end="")

#         # temp softmax
#         deg = torch.zeros(edge_index.max().item() + 1,
#                           dtype=torch.float, device=self.device)
#         exp_wei = torch.exp(edge_weight / 10)
#         deg.scatter_add_(0, edge_index[0], exp_wei)
#         # print(deg)
#         edge_weight = exp_wei / (deg[edge_index[0]] + 1e-8)  # 计算每个边的权重

#         print(
#             f"[AFTER] edge mean:[{edge_weight.mean().item():.4f}] max:[{edge_weight.max().item():.4f}] min:[{edge_weight.min().item():.4f}] std:[{edge_weight.std().item():.4f}]")

#         return edge_weight

        


#     def att_coef3(self, x, edge_index, ln=0):
#         x = x.unsqueeze(0)
#         B, N = x.size(0), x.size(1)

#         if not ln:
#             query = self.Wq(x).reshape(-1, N, 1, 128)  # TODO self.in_dim
#             key = self.Wk(x).reshape(-1, N, 1, 128)
#             value = self.Wv(x).reshape(-1, N, 1, 128)
#         else:
#             query = self.Wq2(x).reshape(-1, N, 1, 128)  # TODO self.in_dim
#             key = self.Wk2(x).reshape(-1, N, 1, 128)
#             value = self.Wv2(x).reshape(-1, N, 1, 128)

#         seed = torch.ceil(torch.abs(torch.sum(query) * 1e5)).to(torch.int32)
#         projection_matrix = utils.create_projection_matrix(
#             50, query.shape[-1], seed=seed).to(query.device)  # (50, 64)
#         # TODO self.nb_random_features
#         query = query / math.sqrt(0.25)
#         key = key / math.sqrt(0.25)
#         query_prime = utils.softmax_kernel_transformation(
#             query, True, projection_matrix)
#         key_prime = utils.softmax_kernel_transformation(
#             key, False, projection_matrix)
#         query_prime = query_prime.permute(
#             1, 0, 2, 3)  # [N, B, H, M] 19717, 1, 1, 50
#         key_prime = key_prime.permute(1, 0, 2, 3)  # [N, B, H, M]
#         value = value.permute(1, 0, 2, 3)  # [N, B, H, D] 19717, 1, 1, 128

#         start, end = edge_index
#         query_end, key_start = query_prime[end], key_prime[start]
#         edge_attn_num = torch.einsum(
#             "ebhm,ebhm->ebh", query_end, key_start)  # [E, B, H]
#         edge_attn_num = edge_attn_num.permute(1, 0, 2)  # [B, E, H]
#         attn_normalizer = utils.denominator(query_prime, key_prime)  # [N, B, H]
#         edge_attn_dem = attn_normalizer[end]  # [E, B, H]
#         edge_attn_dem = edge_attn_dem.permute(1, 0, 2)  # [B, E, H]

#         # [mean agg]
#         # edge_weight = (edge_attn_num /
#         #                edge_attn_dem).squeeze(-1).squeeze(0)  # [B, E, H]
#         # edge_weight = edge_weight.mean(dim=1)
#         # [MLP agg]
#         edge_weight = (edge_attn_num / edge_attn_dem)
#         # print(f"edge all ZERO:[{((edge_weight - 0) < 1e-5).sum().item() / edge_weight.numel():.3f}]")

#         # if not ln:
#         #     edge_weight = self.Wo(edge_weight).squeeze(-1).squeeze(0)
#         # else:
#         #     edge_weight = self.Wo2(edge_weight).squeeze(-1).squeeze(0)
#         edge_weight = edge_weight.squeeze(-1).squeeze(0)

#         # edge_weight = torch.sigmoid(edge_weight)

#         print(
#             f"[BEFORE] edge mean:[{edge_weight.mean().item():.8f}] max:[{edge_weight.max().item():.8f}] min:[{edge_weight.min().item():.8f}] | ", end="")


#         ### temp softmax
#         deg = torch.zeros(edge_index.max().item() + 1,
#                           dtype=torch.float, device=self.device)
#         exp_wei = torch.exp(edge_weight / 0.0001)
#         deg.scatter_add_(0, edge_index[0], exp_wei)
#         edge_weight = exp_wei / (deg[edge_index[0]])  # 计算每个边的权重


#        # mean_w = torch.mean(edge_weight)
#         # std_w = torch.std(edge_weight)
#         # edge_weight = (edge_weight - mean_w) / std_w

#         # min_w = torch.min(edge_weight)
#         # max_w = torch.max(edge_weight)
#         # edge_weight = (edge_weight - min_w) / (max_w - min_w)

#         # edge_weight = torch.sigmoid(edge_weight)
#         # deg = torch.zeros(edge_index.max().item() + 1,
#         #                   dtype=torch.float, device=self.device)
#         # TODO originally '1'
#         # deg.scatter_add_(0, edge_index[0], edge_weight)
#         # edge_weight = edge_weight / (deg[edge_index[0]] + 1e-8)  # 计算每个边的权重

#         # print(
#         #     f"[AFTER] edge mean:[{edge_weight.mean().item():.4f}] max:[{edge_weight.max().item():.4f}] min:[{edge_weight.min().item():.4f}]")

#         # normalize edge weight
#         # deg = torch.zeros(edge_index.max().item() + 1,
#         #                   dtype=torch.float, device=self.device)
#         # print(edge_weight.shape)

#         # print(edge_index.shape)
#         # print(edge_weight.shape)

#         # row, col = edge_index
#         # d_in = degree(col, query.shape[1]).float()
#         # d_norm = 1. / d_in[col]
#         # d_norm_ = d_norm
#         # # d_norm_ = d_norm.reshape(1, -1, 1).repeat(1, 1, 1) # TODO 最后一项是head
#         # link_loss = -torch.mean(edge_weight.log() * d_norm_)

#         ####
#         # deg.scatter_add_(0, edge_index[0], edge_weight)

#         # print(f"[BEFORE] edge mean:[{edge_weight.mean().item():.4f}] max:[{edge_weight.max().item():.4f}] min:[{edge_weight.min().item():.4f}] | ", end="")

#         # 计算每个边的权重
#         # edge_weight = edge_weight / (deg[edge_index[0]] + 1e-8)

#         print(f"[AFTER] edge mean:[{edge_weight.mean().item():.4f}] max:[{edge_weight.max().item():.4f}] min:[{edge_weight.min().item():.4f}] std:[{edge_weight.std().item():.4f}]")

#         # edge_weight_norm
#         return edge_weight


    
#     def adj_pruning(
#         self,
#         edge_index: torch.Tensor,
#         edge_weight: torch.Tensor,
#         threshold: torch.Tensor,
#         prev_mask=None
#     ):
#         # mean_w = torch.mean(edge_weight)
#         # std_w = torch.std(edge_weight)
#         # edge_weight = (edge_weight - mean_w) / std_w + 0.5

#         # min_w = torch.min(edge_weight)
#         # max_w = torch.max(edge_weight)
#         # edge_weight = (edge_weight - min_w) / (max_w - min_w)
    
#         # deg = torch.zeros(edge_index.max().item() + 1,
#         #         dtype=torch.float, device=self.device)
#         # # TODO originally '1'
#         # deg.scatter_add_(0, edge_index[0], edge_weight)
#         # edge_weight = edge_weight / (deg[edge_index[0]] + 1e-8)  # 计算每个边的权重

#         # print(f"pruning mean:[{edge_weight.mean():.4f}] max:[{edge_weight.max():.4f}] min:[{edge_weight.min():.4f}]")

#         # node level
#         # mask = BinaryStep.apply(edge_weight - threshold[edge_index[0]])
#         # layer level
#         # mask = BinaryStep.apply(edge_weight - torch.sigmoid(threshold))
#         mask = BinaryStep.apply(edge_weight - (threshold))

#         # edge_weight = edge_weight * mask
#         # edge_weight = torch.masked_select(edge_weight, edge_weight != 0)
#         return mask * prev_mask if prev_mask is not None else mask

class ThresAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        device = input.device
        output = torch.where(input < -62.5, torch.tensor(0.0).to(device), input).to(device)  # 小于-100时
        output = torch.where((input >= -62.5) & (input <= 0),
                             0.008 * input + 0.5, output).to(device)  # [-100, 0] 之间
        output = torch.where(input > 0, torch.tensor(0.5).to(device), output).to(device)  # 大于0时
        ctx.save_for_backward(input)  # 保存 input 以备反向传播使用
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(input < -62.5) | (input > 0)] = 0  # 导数为0的区间
        grad_input[(input >= -62.5) & (input <= 0)] = 0.008  # [-100, 0] 之间的导数
        return grad_input


class AdaptiveSparsityModule(nn.Module):
    def __init__(self, threshold_init):
        super().__init__()
        self.keep_threshold = nn.Parameter(torch.tensor(threshold_init))

    def forward(self, score, tau=1):
        B = score.size(0)
        if not self.training:
            idx = score > self.keep_threshold
            if B==1:
                return score, idx
            k = idx.sum().item() // B
            return score, k
        
        y_soft = torch.sigmoid((score - self.keep_threshold) / tau)
        y_hard = (y_soft > 0.5).float()
        ret = y_hard - y_soft.detach() + y_soft
        return ret

class net_gcn_dense(nn.Module):
    def __init__(self, embedding_dim, device, spar_wei, spar_adj, num_nodes, mode="prune"):
        super().__init__()

        self.mode = mode

        self.layer_num = len(embedding_dim) - 1
        
        self.spar_wei = spar_wei
        self.spar_adj = spar_adj
        self.edge_mask_archive = []

        if self.spar_wei:
            self.net_layer = nn.ModuleList([ MaskedLinear(embedding_dim[ln], embedding_dim[ln+1], bias=False) for ln in range(self.layer_num)])
        else:
            self.net_layer = nn.ModuleList([nn.Linear(embedding_dim[ln], embedding_dim[ln+1], bias=False) for ln in range(self.layer_num)])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        # self.adj_nonzero = torch.nonzero(adj, as_tuple=False).shape[0]
        # self.adj_mask1_train = nn.Parameter(self.generate_adj_mask(adj))
        # self.adj_mask2_fixed = nn.Parameter(self.generate_adj_mask(adj), requires_grad=False)
        self.normalize = utils.torch_normalize_adj



        self.device = device
        
        if self.spar_adj:
            # self.adj_thresholds = nn.ParameterList(
            # [nn.Parameter(torch.Tensor([-8])) for i in range(self.layer_num)])  # 0.03

            # long-tail
            self.adj_thresholds = nn.ParameterList(
                [nn.Parameter(torch.ones(size=(num_nodes,)) * -54) for i in range(self.layer_num)])

            # # STE
            # self.adj_thresholds = nn.ParameterList(
            #     [nn.Parameter(torch.ones(size=(num_nodes,)) * -20) for i in range(self.layer_num)])


        ### PGExplainer
        self.edge_learner = nn.Sequential(
            nn.Linear(embedding_dim[0] * 2, 2048),
            nn.ReLU(),
            # nn.Linear(1024, 256),
            # nn.ReLU(),
            nn.Linear(2048,1)
        )

        ### Scalabel GNNGuard
        self.sim_learner_src = nn.Linear(embedding_dim[0], 512)
        self.sim_learner_tgt = nn.Linear(embedding_dim[0], 512)
        # self.mid_learner = nn.Parameter( nn.init.xavier_uniform(torch.empty(512, 512)), requires_grad=True)
    
    # def load_wei_mask(self, wei_masks):
    #     for i in range(self.layer_num):
    #         self.net_layer[i].threshold
    
    def __create_learner_input(self, edge_index, embeds):
        row, col = edge_index[0], edge_index[1]
        row_embs, col_embs = embeds[row], embeds[col]
        # node_emb = embeds[node_id].repeat(row.size(0), 1)
        edge_learner_input = torch.cat([row_embs, col_embs], 1)
        return edge_learner_input


    def forward_retain(self, x, edge_index, val_test, edge_masks, wei_masks):
        adj_ori = to_dense_adj(edge_index)[0]
        # print(wei_masks)
        for ln in range(self.layer_num):
            adj = adj_ori * edge_masks[ln] if len(edge_masks) != 0 else adj_ori
            adj = self.normalize(adj, device=self.device)
            x = torch.mm(adj, x)
            x = self.net_layer[ln](x, wei_masks[ln])
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if val_test:
                continue
            x = self.dropout(x)
        return x

    def forward(self, x, edge_index, val_test=False, **kwargs):

        if self.mode == "retain":
            return self.forward_retain(x, edge_index, val_test, kwargs['edge_masks'], kwargs['wei_masks'])

        if self.spar_adj:
            edge_weight = self.learn_soft_edge(x, edge_index)
            # _, edge_weight = to_undirected(edge_index, edge)
            # self.edge_weight = edge_weight
            adj_mask = None
            self.edge_mask_archive = []

        else:
            # edge_weight = self.learn_soft_edge2(x, edge_index)
            edge_weight = None
        
        adj_ori = to_dense_adj(edge_index, edge_attr=edge_weight)[0]
        
        # adj = torch.mul(adj, self.adj_mask1_train)
        # adj = torch.mul(adj, self.adj_mask2_fixed)
        # print(adj)
        # print(adj.shape)
        # adj = self.normalize(adj, self.device)
        #adj = torch.mul(adj, self.adj_mask2_fixed)
        for ln in range(self.layer_num):
            
            if self.spar_adj and not kwargs['pretrain']:
                # long-tail
                # adj_mask = self.adj_pruning(adj_ori + torch.eye(adj_ori.shape[0]).to(self.device), self.adj_thresholds[ln], adj_mask)
                # STE
                adj_mask = self.adj_pruning2(adj_ori + torch.eye(adj_ori.shape[0]).to(self.device), self.adj_thresholds[ln], adj_mask)
                if val_test: print(f"l{ln}: [{(1 - (adj_mask.nonzero().shape[0] / edge_index.shape[1]))*100 :.3f}%]", end=" | ")
                self.edge_mask_archive.append(copy.deepcopy(adj_mask.detach()))
                adj = adj_mask * adj_ori
                # self.edge_mask_archive.append(copy.deepcopy(adj.detach()))
            else:
                adj = adj_ori
            self.edge_weight = adj_ori[edge_index[0], edge_index[1]]


            adj = self.normalize(adj, self.device) if not kwargs['pretrain'] else adj
            # adj = utils.row_normalize_adjacency_matrix(adj) if not kwargs['pretrain'] else adj
            # if (adj.sum(1) == 0).sum() > 0:
            #     print("aaaaaaaa")
                # a

            x = torch.mm(adj, x)
            x = self.net_layer[ln](x)
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if val_test:
                continue
            x = self.dropout(x)
        return x
    
    # PGExplainer
    def learn_soft_edge(self, x, edge_index, ln=0):
        input = self.__create_learner_input(edge_index, x)
        if ln == 0:
            edge_weight =  self.edge_learner(input).squeeze(-1)
        elif ln == 1:
            edge_weight =  self.edge_learner2(input).squeeze(-1)
        else:
            raise NotImplementedError
        # print("i am used!!")
        # edge_weight = torch.sigmoid(edge_weight)

        # temp softmax
        # deg = torch.zeros(edge_index.max().item() + 1,
        #                   dtype=torch.float, device=self.device)

        # exp_wei = torch.exp(edge_weight / 3) # Citeseer 1.5 Cora 3
        # deg.scatter_add_(0, edge_index[0], exp_wei)
        # # print(deg)
        # edge_weight = exp_wei / (deg[edge_index[0]] )  # 计算每个边的权重
                # edge_weight = 1 * torch.sigmoid(edge_weight)
        mean = edge_weight.mean()
        variance = edge_weight.var()
        
        # 对原始数组进行线性变换
        edge_weight = ((edge_weight - mean) * torch.sqrt(0.0001 / variance)) + 1

        # print(
        #     f"[BEFORE] edge mean:[{edge_weight.mean().item():.8f}] max:[{edge_weight.max().item():.8f}] min:[{edge_weight.min().item():.8f}] | ", end="")

        # # temp softmax
        # deg = torch.zeros(edge_index.max().item() + 1,
        #                   dtype=torch.float, device=self.device)
        # exp_wei = torch.exp(edge_weight / 0.8)
        # deg.scatter_add_(0, edge_index[0], exp_wei)
        # # print(deg)
        # edge_weight = exp_wei / (deg[edge_index[0]] + 1e-8)  # 计算每个边的权重

        # print(
        #     f"[AFTER] edge mean:[{edge_weight.mean().item():.4f}] max:[{edge_weight.max().item():.4f}] min:[{edge_weight.min().item():.4f}] std:[{edge_weight.std().item():.4f}]")
        
        return edge_weight
    
    # GNNGuard
    def learn_soft_edge2(self, x, edge_index):
        row, col = edge_index
        # print(x.device)
        row_embs, col_embs = self.sim_learner_src(x[row]), self.sim_learner_tgt(x[col])
        # left = torch.einsum("ik,kk->ik",row_embs,self.mid_learner)
        edge_weight =  torch.einsum("ik,ik->i",row_embs, col_embs)
        # edge_weight = torch.sigmoid(edge_weight)

        # print(
        #     f"[BEFORE] edge mean:[{edge_weight.mean().item():.8f}] max:[{edge_weight.max().item():.8f}] min:[{edge_weight.min().item():.8f}] | ", end="")

        # temp softmax
        deg = torch.zeros(edge_index.max().item() + 1,
                          dtype=torch.float, device=self.device)

        exp_wei = torch.exp(edge_weight / 3)
        deg.scatter_add_(0, edge_index[0], exp_wei)
        # print(deg)
        edge_weight = exp_wei / (deg[edge_index[0]] )  # 计算每个边的权重

        # print(
        #     f"[AFTER] edge mean:[{edge_weight.mean().item():.4f}] max:[{edge_weight.max().item():.4f}] min:[{edge_weight.min().item():.4f}] std:[{edge_weight.std().item():.4f}]")

        return edge_weight


    # def adj_pruning(self, edge_index, edge_weight, thres, prev_mask=None):
    #     mask = BinaryStep.apply(edge_weight - torch.sigmoid(thres))
    #     return mask * prev_mask if prev_mask is not None else mask

    def adj_pruning(self, adj, thres, prev_mask):
        mask = BinaryStep.apply(adj - utils.log_custom(thres).view(-1,1))
        # mask = BinaryStep.apply(adj - ThresAct.apply(thres).view(-1,1))
        # mask = BinaryStep.apply(adj - (thres).view(-1,1))

        return mask * prev_mask if prev_mask is not None else mask
    
    def adj_pruning2(self, adj, thres, prev_mask, tau=0.1, val_test=False):
        B = adj.size(0)
        y_soft = torch.sigmoid((adj - utils.log_custom(thres.unsqueeze(-1))) / tau)
        y_hrad = (y_soft > 0.5).float()
        ret = y_hrad - y_soft.detach() + y_soft
        return ret * prev_mask if prev_mask is not None else ret


    def generate_adj_mask(self, input_adj):
        
        sparse_adj = input_adj
        zeros = torch.zeros_like(sparse_adj)
        ones = torch.ones_like(sparse_adj)
        mask = torch.where(sparse_adj != 0, ones, zeros)
        return mask

    def generate_wei_mask(self):
        if not self.spar_wei:
            return []
        with torch.no_grad():
            wei_masks = []
            for layer in self.net_layer:
                wei_masks.append(layer.mask.detach())
        return wei_masks


class net_gcn_admm(nn.Module):

    def __init__(self, embedding_dim, adj):
        super().__init__()

        self.layer_num = len(embedding_dim) - 1
        self.net_layer = nn.ModuleList([nn.Linear(embedding_dim[ln], embedding_dim[ln+1], bias=False) for ln in range(self.layer_num)])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.adj_nonzero = torch.nonzero(adj, as_tuple=False).shape[0]
        self.adj_layer1 = nn.Parameter(copy.deepcopy(adj), requires_grad=True)
        self.adj_layer2 = nn.Parameter(copy.deepcopy(adj), requires_grad=True)
        
    def forward(self, x, adj, val_test=False):

        for ln in range(self.layer_num):
            if ln == 0:
                x = torch.mm(self.adj_layer1, x)
            elif ln == 1:
                x = torch.mm(self.adj_layer2, x)
            else:
                assert False
            x = self.net_layer[ln](x)
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if val_test:
                continue
            x = self.dropout(x)
        return x

    # def forward(self, x, adj, val_test=False):

    #     for ln in range(self.layer_num):
    #         x = torch.mm(self.adj_list[ln], x)
    #         x = self.net_layer[ln](x)
    #         if ln == self.layer_num - 1:
    #             break
    #         x = self.relu(x)
    #         if val_test:
    #             continue
    #         x = self.dropout(x)
    #     return x

    def generate_adj_mask(self, input_adj):
        
        sparse_adj = input_adj
        zeros = torch.zeros_like(sparse_adj)
        ones = torch.ones_like(sparse_adj)
        mask = torch.where(sparse_adj != 0, ones, zeros)
        return mask

class net_gcn_baseline(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.layer_num = len(embedding_dim) - 1
        self.net_layer = nn.ModuleList([nn.Linear(embedding_dim[ln], embedding_dim[ln+1], bias=False) for ln in range(self.layer_num)])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, adj, val_test=False):

        for ln in range(self.layer_num):
            x = torch.mm(adj, x)
            # x = torch.spmm(adj, x)
            x = self.net_layer[ln](x)
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if val_test:
                continue
            x = self.dropout(x)
        return x


class net_gcn_multitask(nn.Module):

    def __init__(self, embedding_dim, ss_dim):
        super().__init__()

        self.layer_num = len(embedding_dim) - 1
        self.net_layer = nn.ModuleList([nn.Linear(embedding_dim[ln], embedding_dim[ln+1], bias=False) for ln in range(self.layer_num)])
        self.ss_classifier = nn.Linear(embedding_dim[-2], ss_dim, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        

    def forward(self, x, adj, val_test=False):

        x_ss = x

        for ln in range(self.layer_num):
            x = torch.spmm(adj, x)
            x = self.net_layer[ln](x)
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if val_test:
                continue
            x = self.dropout(x)

        if not val_test:
            for ln in range(self.layer_num):
                x_ss = torch.spmm(adj, x_ss)
                if ln == self.layer_num - 1:
                    break
                x_ss = self.net_layer[ln](x_ss)
                x_ss = self.relu(x_ss)
                x_ss = self.dropout(x_ss)
            x_ss = self.ss_classifier(x_ss)

        return x, x_ss

