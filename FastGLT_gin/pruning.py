import math
import numpy as np
import torch
from utils import get_W,compute_edge_score_from_edge_index





class IndexMaskHook:
    def __init__(self, layer, scheduler):
        self.layer = layer
        self.scheduler = scheduler
        self.dense_grad = None



    def __name__(self):
        return 'IndexMaskHook'



    @torch.no_grad()
    def __call__(self, grad):
        
        mask = self.scheduler.backward_masks[self.layer]
        # only calculate dense_grads when necessary
        if self.scheduler.check_if_backward_hook_should_accumulate_grad():
            if self.dense_grad is None:
                # initialize as all 0s so we can do a rolling average
                self.dense_grad = torch.zeros_like(grad)
            self.dense_grad += grad / self.scheduler.accumulation_n
        else:
            self.dense_grad = None
        return grad * mask



def _create_step_wrapper(scheduler, optimizer):
    
    _unwrapped_step = optimizer.step
    def _wrapped_step():
        _unwrapped_step()
        scheduler.reset_momentum()
        scheduler.apply_mask_to_weights()
    optimizer.step = _wrapped_step




"""
code partially from https://github.com/nollied/rigl-torch
"""
class FastScheduler:
    def __init__(self, model, optimizer, remain=1, T_end=400, sparsity_distribution='uniform',
                 ignore_linear_layers=True,ignore_parameters=True ,delta=100, alpha=0.3, static_topo=False,
                 accumulation_n=1, state_dict=None,pretrain=False,beta=1,warmup_steps=0):
        if remain <= 0 or remain > 1:
            raise Exception('remain must be on the interval (0, 1]. Got: %f' % remain)
        self.model = model
        self.optimizer = optimizer

        self.W, self._linear_layers_mask,self._parameters_mask = get_W(model, return_linear_layers_mask=True)
        # modify optimizer.step() function to call "reset_momentum" after
        _create_step_wrapper(self, optimizer)
        
        self.remain = remain
        self.N = [torch.numel(w) for w in self.W]  # include how many weights
        
        
        if state_dict is not None:
            self.load_state_dict(state_dict)
            self.apply_mask_to_weights()

        else:
            self.sparsity_distribution = sparsity_distribution
            self.static_topo = static_topo
            self.accumulation_n = accumulation_n
            self.ignore_linear_layers = ignore_linear_layers
            self.backward_masks = None
            
            self.ignore_parameters = ignore_parameters
            
            # define sparsity allocation
            self.S = []
            for i, (W, is_linear,is_para) in enumerate(zip(self.W, self._linear_layers_mask,self._parameters_mask)):
                # when using uniform sparsity, the first layer is always 100% dense
                # UNLESS there is only 1 layer
                #is_first_layer = i == 0
                #if is_first_layer and self.sparsity_distribution == 'uniform' and len(self.W) > 1:
                    #self.S.append(0)
                if is_linear and self.ignore_linear_layers:
                    # if choosing to ignore linear layers, keep them 100% dense
                    self.S.append(0)
                elif is_para and self.ignore_parameters:
                    self.S.append(0)
                else:
                    self.S.append((1-remain)*beta)
            
            # warmup
            assert warmup_steps>0 or beta>=1,"can't reach special sparisity"
            self.warmup_steps = warmup_steps
            self.warmup_k = math.pow(beta,-1/warmup_steps) if warmup_steps>0 else 1
            self.warmup_ratio = (1-beta)*(1-remain)/warmup_steps if warmup_steps>0 else 1
            # randomly sparsify model according to S
            if not pretrain:
                self.random_sparsify()
            else:
                self.fine_tuning_sparsity()
            # scheduler keeps a log of how many times it's called. this is how it does its scheduling
            self.step = 0
            self.FastGLT_steps = 0

            # define the actual schedule
            self.delta_T = delta
            self.alpha = alpha
            self.T_end = T_end

        # also, register backward hook so sparse elements cannot be recovered during normal training
        self.backward_hook_objects = []
        for i, w in enumerate(self.W):
            # if sparsity is 0%, skip
            if self.S[i] <= 0:
                self.backward_hook_objects.append(None)
                continue

            if getattr(w, '_has_FastGLT_backward_hook', False):
                raise Exception('This model already has been registered to a FastScheduler.')
            self.backward_hook_objects.append(IndexMaskHook(i, self))  # use the last hook
            w.register_hook(self.backward_hook_objects[-1])  # when backward call IndexMaskHook.In an iteration the mask is fixed,so the masked grads and weights keep zero
            setattr(w, '_has_FastGLT_backward_hook', True)
        
        if pretrain:
            self.reset_momentum()
            self.apply_mask_to_weights()
            self.apply_mask_to_gradients()

        assert self.accumulation_n > 0 and self.accumulation_n < delta
        assert self.sparsity_distribution in ('uniform', )


    def __str__(self):
        s = 'FastScheduler(\n'
        s += 'layers=%i,\n' % len(self.N)

        # calculate the number of non-zero elements out of the total number of elements
        N_str = '['
        S_str = '['
        sparsity_percentages = []
        total_params = 0
        total_conv_params = 0
        total_nonzero = 0
        total_conv_nonzero = 0

        for N, S, mask, W, is_linear in zip(self.N, self.S, self.backward_masks, self.W, self._linear_layers_mask):
            actual_S = torch.sum(W[mask == 0] == 0).item()
            N_str += ('%i/%i, ' % (N-actual_S, N))
            sp_p = float(N-actual_S) / float(N) * 100
            S_str += '%.2f%%, ' % sp_p
            sparsity_percentages.append(sp_p)
            total_params += N
            total_nonzero += N-actual_S
            if not is_linear:
                total_conv_nonzero += N-actual_S
                total_conv_params += N

        N_str = N_str[:-2] + ']'
        S_str = S_str[:-2] + ']'
        
        s += 'nonzero_params=' + N_str + ',\n'
        s += 'nonzero_percentages=' + S_str + ',\n'
        s += 'total_nonzero_params=' + ('%i/%i (%.2f%%)' % (total_nonzero, total_params, float(total_nonzero)/float(total_params)*100)) + ',\n'
        s += 'total_CONV_nonzero_params=' + ('%i/%i (%.2f%%)' % (total_conv_nonzero, total_conv_params, float(total_conv_nonzero)/float(total_conv_params)*100)) + ',\n'
        s += 'step=' + str(self.step) + ',\n'
        s += 'num_FastGLT_steps=' + str(self.FastGLT_steps) + ',\n'
        s += 'ignoring_linear_layers=' + str(self.ignore_linear_layers) + ',\n'
        s += 'sparsity_distribution=' + str(self.sparsity_distribution) + ',\n'

        return s + ')'


    def __call__(self):
        self.step += 1
        if self.static_topo:
            return True
        if (self.step % self.delta_T) == 0 and self.step <= self.T_end: # check schedule
            self._FastGLT_step()
            self.FastGLT_steps += 1
            return False
        if self.step > self.T_end:
            if self.step == self.T_end + 1:
                print("reset")
                self.reset_momentum()
                self.apply_mask_to_weights()
                self.apply_mask_to_gradients() 
            return False
        return True


    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if type(v) == dict:
                self.load_state_dict(v)
            setattr(self, k, v)
    
    
    def state_dict(self):
        obj = {
            'remain': self.remain,
            'S': self.S,
            'N': self.N,
            'hyperparams': {
                'delta_T': self.delta_T,
                'alpha': self.alpha,
                'T_end': self.T_end,
                'ignore_linear_layers': self.ignore_linear_layers,
                'static_topo': self.static_topo,
                'sparsity_distribution': self.sparsity_distribution,
                'accumulation_n': self.accumulation_n,
            },
            'step': self.step,
            'FastGLT_steps': self.FastGLT_steps,
            'backward_masks': self.backward_masks,
            '_linear_layers_mask': self._linear_layers_mask,
        }

        return obj

           
    @torch.no_grad()
    def reset_momentum(self):
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            # if sparsity is 0%, skip
            if s <= 0:
                continue

            param_state = self.optimizer.state[w]
            if 'momentum_buffer' in param_state:
                # mask the momentum matrix
                buf = param_state['momentum_buffer']
                buf *= mask


    @torch.no_grad()
    def apply_mask_to_weights(self):
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            # if sparsity is 0%, skip
            if s <= 0:
                continue
            w *= mask


    @torch.no_grad()
    def apply_mask_to_gradients(self):
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            # if sparsity is 0%, skip
            if s <= 0:
                continue
            w.grad *= mask

    
    @torch.no_grad()
    def random_sparsify(self):
        self.backward_masks = []
        for l, w in enumerate(self.W):
            # if sparsity is 0%, skip
            if self.S[l] <= 0:
                self.backward_masks.append(None)
                continue

            n = self.N[l]
            s = int(self.S[l] * n)
            perm = torch.randperm(n,device=w.device) # generate random permutation
            perm = perm[:s]  # select s weights
            flat_mask = torch.ones(n, device=w.device)
            flat_mask[perm] = 0
            mask = torch.reshape(flat_mask, w.shape)
            mask = mask.bool()
            w *= mask
            self.backward_masks.append(mask)

            
    @torch.no_grad()
    def fine_tuning_sparsity(self):
        self.backward_masks = []
        # init the weight mask
        for l, w in enumerate(self.W):
            # if sparsity is 0%, skip
            if self.S[l] <= 0:
                self.backward_masks.append(None)
                continue
            
            n_total = self.N[l]  # the total num of the weights
            s = int(self.S[l] * n_total)  # the sparisity num of the weights
            n_keep = n_total - s
            
            score_drop = torch.abs(w)
            
            # create drop mask
            _, sorted_indices = torch.topk(score_drop.view(-1), k=n_total)
            
            new_values = torch.where(
                            torch.arange(n_total, device=w.device) < n_keep,
                            torch.ones_like(sorted_indices),
                            torch.zeros_like(sorted_indices))
            flat_mask = new_values.scatter(0, sorted_indices, new_values)  #new_values[sorted_indices[i]]=new_values[i]
            
            mask = torch.reshape(flat_mask,w.shape)
            mask = mask.bool()
            w *= mask
            self.backward_masks.append(mask)
        
    
    def check_if_backward_hook_should_accumulate_grad(self):
        """
        Used by the backward hooks. Basically just checks how far away the next FastGLT step is, 
        if it's within `self.accumulation_n` steps, return True.
        """
        if self.step >= self.T_end:
            return False
        steps_til_next_FastGLT_step = self.delta_T - (self.step % self.delta_T)
        return steps_til_next_FastGLT_step <= self.accumulation_n


    def cosine_annealing(self):
        return self.alpha / 2 * (1 + np.cos((self.step * np.pi) / self.T_end))


    @torch.no_grad()
    def _FastGLT_step(self):
        drop_fraction = self.cosine_annealing()

        # if distributed these values will be populated
        #is_dist = dist.is_initialized()
        #world_size = dist.get_world_size() if is_dist else None
        # prun the network
        for l, (w, is_linear,is_para) in enumerate(zip(self.W, self._linear_layers_mask,self._parameters_mask)):
            # if sparsity is 0%, skip
            if self.S[l] <= 0:
                continue

            current_mask = self.backward_masks[l]
            # calculate raw scores
            score_w = torch.abs(w)
            score_grad = torch.abs(self.backward_hook_objects[l].dense_grad)
            
            if is_para:
                score_grow = torch.abs(compute_edge_score_from_edge_index(self.model.graph.edges(),self.model.num_nodes,current_mask.squeeze(),device=self.model.device)).reshape(-1,1) # TODO
                # print(score_grow.mean().item())
                score_grow += score_grad  # cora needed use this
                # print(score_grow.mean().item())
            elif is_linear:
                score_grow = score_grad
            score_drop = score_w 
            
            # calculate drop/grow quantities
            n_total = self.N[l]
            n_ones = torch.sum(current_mask).item()
            n_grow = int(n_ones * drop_fraction)
            n_prune = n_grow
            if self.FastGLT_steps<self.warmup_steps:
                n_prune += int((self.warmup_k-1)*self.S[l]*n_total)
                self.S[l]*=self.warmup_k
                #n_prune += n_total*self.warmup_ratio
                #self.S[l] += self.warmup_ratio
                assert n_prune<n_total,"when doing warmup,n_prune is larger than n_total,please make beta or warmup_steps larger or make alpha smaller"
            n_keep = n_ones - n_prune

            # create drop mask
            _, sorted_indices = torch.topk(score_drop.view(-1), k=n_total)
            new_values = torch.where(
                            torch.arange(n_total, device=w.device) < n_keep,
                            torch.ones_like(sorted_indices),
                            torch.zeros_like(sorted_indices))  #new_values is a tensor which the front is 1 and the back is 0
            
            
            mask1 = new_values.scatter(0, sorted_indices, new_values)  # new_values[sorted_index[i]] = new_values[i]

            # flatten grow scores
            score_grow = score_grow.view(-1)

            # set scores of the enabled connections(ones) to min(s) - 1, so that they have the lowest scores
            score_grow_lifted = torch.where(
                                mask1 == 1, 
                                torch.ones_like(mask1) * (torch.min(score_grow) - 1),
                                score_grow)

            # create grow mask
            _, sorted_indices = torch.topk(score_grow_lifted, k=n_total)
            new_values = torch.where(
                            torch.arange(n_total, device=w.device) < n_grow,
                            torch.ones_like(sorted_indices),
                            torch.zeros_like(sorted_indices))
            mask2 = new_values.scatter(0, sorted_indices, new_values)

            mask2_reshaped = torch.reshape(mask2, current_mask.shape)
            
            if is_linear:
                grow_tensor = torch.zeros_like(w)  #!gnn
            elif is_para:
                grow_tensor = torch.ones_like(w)*torch.mean(w[current_mask])   #!edge 
            REINIT_WHEN_SAME = False
            if REINIT_WHEN_SAME:
                raise NotImplementedError()
            else:
                new_connections = ((mask2_reshaped == 1) & (current_mask == 0))

            # update new weights to be initialized as zeros and update the weight tensors
            new_weights = torch.where(new_connections.to(w.device), grow_tensor, w)
            w.data = new_weights

            mask_combined = torch.reshape(mask1 + mask2, current_mask.shape).bool()

            # update the mask
            current_mask.data = mask_combined
            self.reset_momentum()
            self.apply_mask_to_weights()
            self.apply_mask_to_gradients() 

    @torch.no_grad()
    def getS(self):
        total_weight = 0
        total_weight_zero = 0 
        total_paras = 0
        total_paras_zero = 0
        for N, S, mask, W, is_linear,is_para in zip(self.N, self.S, self.backward_masks, self.W, self._linear_layers_mask,self._parameters_mask ):
            if S > 0:
                actual_mask = torch.sum(W[mask == 0] == 0).item()  
                if is_para:
                    total_paras += N
                    total_paras_zero += actual_mask
                if  is_linear:
                    total_weight += N
                    total_weight_zero += actual_mask
        spar_weight = total_weight_zero/total_weight if total_weight>0 else 0
        spar_adj = total_paras_zero/total_paras if total_paras>0 else 0
        return spar_weight*100,spar_adj*100
                    
