import os
import numpy as np
import torch
import pickle
import pandas as pd

from torch.utils.data import DataLoader
from generate_data import generate_vrp_data
from utils import load_model
from problems import CVRP

result_list = []
for i in range(0, 100):
    model, _ = load_model('outputs/cvrp_20/vrp20_rollout_20220415T150415/epoch-' + str(i) + '.pt')
    torch.manual_seed(1234)
    SIZE = 100
    num_samples = 56
    # dataset = CVRP.make_dataset(size=100, num_samples=1000)
    dataset = CVRP.make_dataset(size=SIZE, num_samples=num_samples)

    index = 0
    paths = os.listdir("solomon/csv/100")
    paths.sort()
    for data_path in paths:
        df = pd.read_csv("solomon/csv/100/" + data_path).drop(columns=["Unnamed: 0"])
        df = df[:26]

        df['XCOORD.'] = df['XCOORD.'] / 100
        df['YCOORD.'] = df['YCOORD.'] / 100
        df["DEMAND"] = df["DEMAND"] / 200
        df["READY TIME "] = df["READY TIME "] / 100
        df["DUE DATE "] = df["DUE DATE "] / 100

        dataset[index]['depot'] = torch.Tensor(df[['XCOORD.', 'YCOORD.']].values.tolist()[0])
        dataset[index]['loc'] = torch.Tensor(df[['XCOORD.', 'YCOORD.']].values.tolist()[1:])
        dataset[index]['demand'] = torch.Tensor(df["DEMAND"].values.tolist()[1:])
        dataset[index]['time_window'] = torch.Tensor(df[['READY TIME ', 'DUE DATE ']].values.tolist())
        #         dataset[index]["service_time"] = torch.Tensor([df['SERVICE TIME'].values[-1] / 100])
        cated_array = torch.cat((dataset[index]['depot'][None, 0:], dataset[index]['loc']))
        distance = torch.cdist(cated_array, cated_array, p=2)
        # self.data[i]['loc'] = distance[1:]
        # self.data[i]['depot'] = distance[0, :]
        dataset[index]['matrix'] = distance
        index += 1

    # Need a dataloader to batch instances
    dataloader = DataLoader(dataset, batch_size=1000)

    # Make var works for dicts
    batch = next(iter(dataloader))

    # Run the model
    model.eval()
    model.set_decode_type('greedy')
    with torch.no_grad():
        length, log_p, pi = model(batch, return_pi=True)
    tours = pi


    def to_depot_distance(coords, location, i):
        return np.linalg.norm(np.array(coords[location[i - 1]]) * 100 - np.array(coords[location[i]]) * 100)


    total_list = []
    for instance_index in range(56):
        total = 0
        for i in range(len(model.state.current_time_list)):
            current = float(model.state.current_time_list[i][instance_index] * 100)
            if i != 0 and current == 0.0:
                total += previous + to_depot_distance(model.state.coords[instance_index], tours[instance_index], i)
            previous = current
        total_list.append(total)
    result_list.append(sum(total_list))
    print(i, " ", sum(total_list))

# model, _ = load_model('outputs/cvrp_20/vrp20_rollout_20220410T102822/epoch-5.pt')
#
# torch.manual_seed(1234)
# SIZE=20
# num_samples=56
# # dataset = CVRP.make_dataset(size=100, num_samples=1000)
# dataset = CVRP.make_dataset(size=SIZE, num_samples=num_samples)
#
# import pandas as pd
# import os
#
# index = 0
# paths = os.listdir("solomon/csv/100")
# paths.sort()
# for data_path in paths:
#     df = pd.read_csv("solomon/csv/100/" + data_path).drop(columns = ["Unnamed: 0"])
#     df = df[:26]
#
#     df['XCOORD.'] = df['XCOORD.'] / 100
#     df['YCOORD.'] = df['YCOORD.'] / 100
#     df["DEMAND"] = df["DEMAND"] / 200
#     df["READY TIME "] = df["READY TIME "] / 100
#     df["DUE DATE "] = df["DUE DATE "] / 100
#
#     dataset[index]['depot'] = torch.Tensor(df[['XCOORD.', 'YCOORD.']].values.tolist()[0])
#     dataset[index]['loc'] = torch.Tensor(df[['XCOORD.', 'YCOORD.']].values.tolist()[1:])
#     dataset[index]['demand'] = torch.Tensor(df["DEMAND"].values.tolist()[1:])
#     dataset[index]['time_window'] = torch.Tensor(df[['READY TIME ', 'DUE DATE ']].values.tolist())
# #     dataset[index]["service_time"] = torch.Tensor([df['SERVICE TIME'].values[-1] / 100])
#     index += 1
#
# # Need a dataloader to batch instances
# dataloader = DataLoader(dataset, batch_size=1000)
#
# # Make var works for dicts
# batch = next(iter(dataloader))
#
# # Run the model
# model.eval()
# model.set_decode_type('greedy')
# with torch.no_grad():
#     length, log_p, pi = model(batch, return_pi=True)
# tours = pi
#
# def to_depot_distance(coords, location, i):
#     return np.linalg.norm(np.array(coords[location[i-1]]) * 100 - np.array(coords[location[i]]) * 100)
#
# total_list = []
# for instance_index in range(56):
#     total = 0
#     for i in range(len(model.state.current_time_list)):
#         if instance_index == 112:
#             print(int(tours[instance_index][i]), model.state.current_time_list[i][instance_index] * 100, dataset.data[instance_index]['time_window'][pi[instance_index][i]] * 100)
#         current = float(model.state.current_time_list[i][instance_index] * 100)
#         if i != 0 and current == 0.0:
#             total += previous + to_depot_distance(model.state.coords[instance_index], tours[instance_index], i)
#         previous = current
#     print(instance_index," : ", total)
#     total_list.append(total)
#
# print(sum(total_list))