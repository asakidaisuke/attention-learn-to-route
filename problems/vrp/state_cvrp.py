import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateCVRP(NamedTuple):
    # Fixed input
    # coords: torch.Tensor  # Depot + loc
    demand: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    time_window: torch.Tensor # [batch_size, node_num, 2]
    current_time: torch.Tensor
    VEHICLE_CAPACITY = 1.0  # Hardcoded
    current_time_list: list
    cost: torch.Tensor
    service_time: torch.Tensor
    matrix: torch.Tensor

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    # @property
    # def dist(self):
    #     return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):

        # depot = input['depot']
        # loc = input['loc']
        demand = input['demand']
        time_window = input['time_window']
        service_time = input['service_time']
        matrix = input['matrix']

        batch_size, n_loc, _ = matrix.size()
        n_loc -= 1 # subtract depot
        return StateCVRP(
            # coords=torch.cat((depot[:, None, :], loc), -2),
            demand=demand,
            ids=torch.arange(batch_size, dtype=torch.int64, device=matrix.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=matrix.device),
            used_capacity=demand.new_zeros(batch_size, 1),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=matrix.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=matrix.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=matrix.device), # Add step dimension
            i=torch.zeros(1, dtype=torch.int64, device=matrix.device),  # Vector with length num_steps
            time_window=time_window,
            current_time=torch.zeros(batch_size, 1, device=matrix.device),
            current_time_list = [], cost = torch.zeros(batch_size, device=matrix.device),
            service_time=service_time,
            matrix=matrix
        )

    # def get_final_cost(self):
    #
    #     assert self.all_finished()
    #
    #     return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected, reset_time_mask):

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        selected = selected[:, None]  # Add dimension for step

        n_loc = self.demand.size(-1)  # Excludes depot

        # Add the length
        # cur_coord = self.coords[self.ids, selected]
        # get timewidow start time
        start_time = self.time_window[:, :, 0].gather(1, selected[:, 0].unsqueeze(1))

        is_depot = ((selected == 0).sum(axis=1).reshape(-1,)!=0).type(torch.bool)
        # lengths = (cur_coord - self.cur_coord).norm(p=2, dim=-1)# (batch_dim, 1)
        batch_size = self.matrix.shape[0]
        num_node = self.matrix.shape[1]
        select_start_from_matrix = torch.gather(self.matrix, 1, self.prev_a[:, :, None].expand(-1, -1, num_node)).view(batch_size, -1)
        lengths = torch.gather(select_start_from_matrix, 1, selected)

        # use later time for current time
        current_time = torch.concat((self.current_time + lengths, start_time), 1).max(1).values[:, None]

        # add cost
        cost = self.cost
        cost[reset_time_mask | is_depot] += current_time.squeeze()[reset_time_mask | is_depot]

#         current_time += self.service_time
        current_time[reset_time_mask | is_depot] = 0
        current_time_list = self.current_time_list
        current_time_list.append(current_time)
        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        #selected_demand = self.demand.gather(-1, torch.clamp(prev_a - 1, 0, n_loc - 1))
        selected_demand = self.demand[self.ids, torch.clamp(selected - 1, 0, n_loc - 1)]

        # Increase capacity if depot is not visited, otherwise set to 0
        #used_capacity = torch.where(selected == 0, 0, self.used_capacity + selected_demand)
        used_capacity = (self.used_capacity + selected_demand) * (selected != 0).float()

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, selected[:, :, None], 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, selected - 1)

        prev_a = selected
        return self._replace(
            prev_a=prev_a, used_capacity=used_capacity, visited_=visited_,
            lengths=lengths, i=self.i + 1, current_time=current_time,
            current_time_list=current_time_list, cost=cost
        )

    def all_finished(self):
        return self.i.item() >= self.demand.size(-1) and self.visited.all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """

        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, 1:]
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.demand.size(-1))

        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
        exceeds_cap = (self.demand[self.ids, :] + self.used_capacity[:, :, None] > self.VEHICLE_CAPACITY)

        # add time window mask
        # start_point = torch.gather(self.coords, 1, self.prev_a[:, None].expand(-1, -1, 2))
        # time_window_mask = self.time_window[:,:,1] - (
        #         torch.sqrt(torch.pow(self.coords - start_point, 2).sum(dim=2)) + self.current_time) < 0
        batch_size = self.matrix.shape[0]
        num_node = self.matrix.shape[1]
        distances = torch.gather(
            self.matrix, 1, self.prev_a[:, None].expand(-1, -1, num_node)).view(batch_size,-1)
        time_window_mask = self.time_window[:,:,1] - (distances + self.current_time) < 0
        time_window_mask = time_window_mask[:, 1:]  # depot is excluded
        time_window_mask = time_window_mask[:, None, :]
        mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap | time_window_mask
        reset_time_mask = ((mask_loc == False).sum(axis=2).reshape(-1,)==0).type(torch.bool)

        # Nodes that cannot be visited are already visited or too much demand to be served now
        mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap | time_window_mask

        # Cannot visit the depot if just visited and still unserved nodes
        # change or from and TODO
        mask_depot = (self.prev_a == 0) | ((mask_loc == 0).int().sum(-1) > 0)
        mask_depot[(((self.prev_a == 0) & ((mask_loc == 0).int().sum(-1) == 0)).view(-1))] = False

        return torch.cat((mask_depot[:, :, None], mask_loc), -1), reset_time_mask

    def construct_solutions(self, actions):
        return actions
