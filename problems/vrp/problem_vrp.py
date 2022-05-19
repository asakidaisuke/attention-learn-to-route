from torch.utils.data import Dataset
import torch
import os
import pickle
import random
import numpy as np

from problems.vrp.state_cvrp import StateCVRP
from utils.beam_search import beam_search


class CVRP(object):

    NAME = 'cvrp'  # Capacitated Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi, state):
        batch_size, graph_size = dataset['demand'].size()
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -CVRP.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        d = demand_with_depot.gather(1, pi)

        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        for i in range(pi.size(1)):
            used_cap += d[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (used_cap <= CVRP.VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"

        # Gather dataset in order of tour
        # loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        # d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        # return (
        #     (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
        #     + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
        #     + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        # ), None
        # CHANGE change cost
        return state.cost, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateCVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = CVRP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


# class SDVRP(object):

#     NAME = 'sdvrp'  # Split Delivery Vehicle Routing Problem

#     VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

#     @staticmethod
#     def get_costs(dataset, pi):
#         batch_size, graph_size = dataset['demand'].size()

#         # Each node can be visited multiple times, but we always deliver as much demand as possible
#         # We check that at the end all demand has been satisfied
#         demands = torch.cat(
#             (
#                 torch.full_like(dataset['demand'][:, :1], -SDVRP.VEHICLE_CAPACITY),
#                 dataset['demand']
#             ),
#             1
#         )
#         rng = torch.arange(batch_size, out=demands.data.new().long())
#         used_cap = torch.zeros_like(dataset['demand'][:, 0])
#         a_prev = None
#         for a in pi.transpose(0, 1):
#             assert a_prev is None or (demands[((a_prev == 0) & (a == 0)), :] == 0).all(), \
#                 "Cannot visit depot twice if any nonzero demand"
#             d = torch.min(demands[rng, a], SDVRP.VEHICLE_CAPACITY - used_cap)
#             demands[rng, a] -= d
#             used_cap += d
#             used_cap[a == 0] = 0
#             a_prev = a
#         assert (demands == 0).all(), "All demand must be satisfied"

#         # Gather dataset in order of tour
#         loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
#         d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

#         # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
#         return (
#             (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
#             + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
#             + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
#         ), None

#     @staticmethod
#     def make_dataset(*args, **kwargs):
#         return VRPDataset(*args, **kwargs)

#     @staticmethod
#     def make_state(*args, **kwargs):
#         return StateSDVRP.initialize(*args, **kwargs)

#     @staticmethod
#     def beam_search(input, beam_size, expand_size=None,
#                     compress_mask=False, model=None, max_calc_batch_size=4096):
#         assert model is not None, "Provide model"
#         assert not compress_mask, "SDVRP does not support compression of the mask"

#         fixed = model.precompute_fixed(input)

#         def propose_expansions(beam):
#             return model.propose_expansions(
#                 beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
#             )

#         state = SDVRP.make_state(input)

#         return beam_search(state, beam_size, propose_expansions)


def make_instance(args):
    depot, loc, demand, capacity, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size
    }

# def create_time_window(size, window_scale = 15):
#     tensor = torch.randint(15, (size, 2))
# #     for t in tensor:
# #         if t[0] > t[1]:
# #             temp = t[1].clone().detach()
# #             t[1] = t[0]
# #             t[0] = temp
# #         elif t[0] == t[1]:
# #             t[1] = t[1] + int(window_scale * 0.2)
# #         if t[1] - t[0] > 3:
# #             t[1] -= 3
# #         t[1] += 1  # prevent unsined jobs
#     for t in tensor:
#         r = random.uniform(0,1.0)
#         if r < 0.5:
#             t[1] = t[0] + 2
#         elif r >= 0.5 and r < 0.7:
#             t[0] = 0
#             if t[1] == 0 or t[1] == 1:
#                 t[1] = 3
#         else:
#             t[1] = t[0] + 2
#     tensor = torch.cat((torch.zeros((1, 2)), tensor))
#     return tensor

def create_time_window(size, require):
    tensor = (torch.FloatTensor(size,2).uniform_(100, 849).int()).float() / 100
    tensor = torch.maximum(tensor, require.view(size,-1)+0.01)
    for t in tensor:
        r = random.gauss(115.96, 35.78) / 100
        t[1] = min(t[0] + max(r, 0.01), 10.0)
    tensor = torch.cat((torch.zeros((1, 2)), tensor))
    return tensor

def create_demand(size):
    base = (torch.FloatTensor(size).normal_(mean=15, std=10).int()).float() / 1000
    low_clip = torch.ones((1, size)).float() / 1000
    hight_clip = torch.ones((1,size)).float() * 42 / 1000
    low_clipped = torch.maximum(base, low_clip)
    hight_clipped = torch.minimum(hight_clip, low_clipped)
    return hight_clipped[0]

class VRPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(VRPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[offset:offset+num_samples]]
            # give the time window information
            for data in self.data:
                data['service_time'] = torch.Tensor([0])

            for i in range(len(self.data)):
                cated_array = torch.cat((self.data[i]['depot'][None, 0:], self.data[i]['loc']))
                distance = torch.cdist(cated_array, cated_array, p=2)
                self.data[i]['time_window'] = create_time_window(len(data['loc']), distance[0][1:])
                self.data[i]['matrix'] = distance
                self.data[i]['time_window'][:, 1] = 10
                
                if False: # for weighted
#                     self.data[i]['matrix'] *= 0.0
                    self.data[i]['matrix'] += torch.Tensor(size+1,size+1).uniform_(0.0, 2.0)
                    ind = np.diag_indices(self.data[i]['matrix'].shape[0])
                    self.data[i]['matrix'][ind[0], ind[1]] = torch.zeros(self.data[i]['matrix'].shape[0])
                    diff = self.data[i]['time_window'].T[1] - self.data[i]['time_window'].T[0]
                    self.data[i]['time_window'].T[0] = torch.maximum(self.data[i]['time_window'].T[0], self.data[i]['matrix'][0]+0.01)
                    self.data[i]['time_window'].T[0][0] = 0
                    self.data[i]['time_window'].T[1] = diff + self.data[i]['time_window'].T[0]
#                     self.data[i]['time_window'].T[0] *= 0.8 
#                     self.data[i]['time_window'].T[1] *= 1.2



        else:

            self.data = []
            for i in range(num_samples):
                loc = torch.FloatTensor(size, 2).uniform_(0, 1)
                demand = create_demand(size)
                depot = torch.FloatTensor(2).uniform_(0, 1)
                service_time = torch.Tensor([0.1])
#                 service_time = 0
                
                cated_array = torch.cat((depot[None, 0:], loc))
                distance = torch.cdist(cated_array, cated_array, p=2)
                distance[1:] += service_time
                
                time_window = create_time_window(size, distance[0][1:])
                time_window[:, 1] = 10
                
                if False: # for weighted
#                     distance *= 0.0
                    distance += torch.Tensor(size+1,size+1).uniform_(0.0, 2.0)
                    ind = np.diag_indices(distance.shape[0])
                    distance[ind[0], ind[1]] = torch.zeros(distance.shape[0])
                    diff = time_window.T[1] - time_window.T[0]
                    time_window.T[0] = torch.maximum(time_window.T[0], distance[0]+0.01)
                    time_window.T[0][0] = 0
                    time_window.T[1] = diff + time_window.T[0]
#                     time_window.T[0] *= 0.8 
#                     time_window.T[1] *= 1.2
                
                self.data.append({
                    'loc': loc,
                    'demand': demand,
                    'depot': depot,
                    'time_window': time_window,
                    'service_time': service_time,
                    'matrix': distance
                })
                


        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
