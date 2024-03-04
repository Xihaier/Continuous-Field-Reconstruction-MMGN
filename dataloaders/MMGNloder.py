import torch
import numpy as np

from torch.utils.data import Dataset


class ZscoreStandardizer(object):
    """  
    Normalization transformation
    if reduce_dim = [0]: The mean is computed over different time volumes.
    if reduce_dim = []:  The mean is computed over all data points.
    """  
    def __init__(self, x, reduce_dim=[0]):
        self.mean = torch.mean(x, reduce_dim, keepdim=True).squeeze()
        self.std  = torch.std(x, reduce_dim, keepdim=True).squeeze()
        self.epsilon = 1e-10
        assert self.mean.shape == self.std.shape

    def do(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.epsilon)

    def undo(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + self.epsilon) + self.mean


class MinMaxStandardizer(object):
    """  
    Min-Max transformation
    if reduce_dim = [0]: The min/max is computed over different time volumes.
    if reduce_dim = []:  The min/max is computed over all data points.
    """  
    def __init__(self, x, reduce_dim=[0]):
        if reduce_dim:
            self.minVal = torch.min(x, reduce_dim[0], keepdim=True)[0].squeeze()
            self.maxVal = torch.max(x, reduce_dim[0], keepdim=True)[0].squeeze()
        else:
            self.minVal = torch.min(x)
            self.maxVal = torch.max(x)

        self.epsilon = 1e-10

        assert self.minVal.shape == self.maxVal.shape

    def do(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.minVal) / (self.maxVal - self.minVal) + self.epsilon

    def undo(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.epsilon) * (self.maxVal - self.minVal) + self.minVal
    

def get_MMGN(data_dir, n_data, preprocessing, reduce_dim, task, sampling_rate):
    """
    Args:
        data_dir (string): dataset file path
        n_data (int): number of data samples 
        preprocessing (list): data preprocessing method
        reduce_dim (list): data preprocessing reduced dimension
        task (string): 
            task 1: fixed n_points,  fixed positions
            task 2: random n_points, fixed positions
            task 3: fixed n_points,  random positions
            task 4: random n_points, random positions
        sampling_rate (list): sampling rate lower and upper bound
    Returns:
        grid_lat (torch.tensor): (lat, lon) = (192, 288)
        grid_lon (torch.tensor): (lat, lon) = (192, 288)
        gst      (torch.tensor): (n_data, lat, lon) = (n_data, 192, 288)
        indices  (torch.tensor): (n_data, n_points)
    """
    # load data
    dat = np.load(data_dir)
    lat = torch.from_numpy(dat["lats"])
    lon = torch.from_numpy(dat["lons"])
    gst = torch.from_numpy(dat["temperature"][:n_data,:,:])
    del dat

    # pre-process data
    lat_lon_pre, gst_pre = preprocessing
    lat_lon_reduce_dim, gst_reduce_dim = reduce_dim
    normalizer = {}

    if lat_lon_pre == "zscore":
        normalizer["lat"] = ZscoreStandardizer(lat, lat_lon_reduce_dim)
        normalizer["lon"] = ZscoreStandardizer(lon, lat_lon_reduce_dim)        
    elif lat_lon_pre == "minmax":
        normalizer["lat"] = MinMaxStandardizer(lat, lat_lon_reduce_dim)
        normalizer["lon"] = MinMaxStandardizer(lon, lat_lon_reduce_dim)

    if gst_pre == "zscore":
        normalizer["temperature"] = ZscoreStandardizer(gst, gst_reduce_dim)      
    elif gst_pre == "minmax":
        normalizer["temperature"] = MinMaxStandardizer(gst, gst_reduce_dim)

    lat = normalizer["lat"].do(lat)
    lon = normalizer["lon"].do(lon)
    gst = normalizer["temperature"].do(gst)
    grid_lat, grid_lon = torch.meshgrid(lat, lon, indexing="ij")

    if task == "task1":
        n_points = int(np.round(sampling_rate[1] * grid_lat.numel()))
        indices  = torch.randperm(grid_lat.numel())
        indices  = indices[:n_points]
        indices  = indices.repeat(n_data, 1)
    elif task == "task2":
        rates = np.random.uniform(sampling_rate[0], sampling_rate[1], n_data)
        n_points = [int(np.round(rates[idx] * grid_lat.numel())) for idx in range(n_data)]
        max_n_points = max(n_points)
        indices  = torch.randperm(grid_lat.numel())
        ind_lst = []
        for idx in range(n_data):
            # sampling_rate[0] = 0.01 then five times to ensure sufficiency
            if sampling_rate[0] == 0.01:
                ind_lst.append(torch.cat((indices[:n_points[idx]], indices[:n_points[idx]], indices[:n_points[idx]], indices[:n_points[idx]], indices[:n_points[idx]]))[:max_n_points][None,:])
            else:
                ind_lst.append(torch.cat((indices[:n_points[idx]], indices[:n_points[idx]]))[:max_n_points][None,:])
        indices = torch.cat(ind_lst, dim=0)
    elif task == "task3":
        n_points = int(np.round(sampling_rate[1] * grid_lat.numel()))
        ind_lst = []
        for idx in range(n_data):
            indices  = torch.randperm(grid_lat.numel())
            ind_lst.append(indices[:n_points][None,:])
        indices = torch.cat(ind_lst, dim=0)        
    elif task == "task4":
        rates = np.random.uniform(sampling_rate[0], sampling_rate[1], n_data)
        n_points = [int(np.round(rates[idx] * grid_lat.numel())) for idx in range(n_data)]
        max_n_points = max(n_points)
        ind_lst = []
        for idx in range(n_data):
            indices  = torch.randperm(grid_lat.numel())
            # sampling_rate[0] = 0.01 then five times to ensure sufficiency
            if sampling_rate[0] == 0.01:
                ind_lst.append(torch.cat((indices[:n_points[idx]], indices[:n_points[idx]], indices[:n_points[idx]], indices[:n_points[idx]], indices[:n_points[idx]]))[:max_n_points][None,:])
            else:
                ind_lst.append(torch.cat((indices[:n_points[idx]], indices[:n_points[idx]]))[:max_n_points][None,:])
        indices = torch.cat(ind_lst, dim=0)
    return grid_lat, grid_lon, gst, indices, normalizer


class MMGN(Dataset):
    def __init__(self, 
        temperature: torch.Tensor,
        grid_lat: torch.Tensor,
        grid_lon: torch.Tensor,
        indices: torch.Tensor,
    ):
        self.temperature = temperature
        self.grid_lat = grid_lat.reshape(-1)
        self.grid_lon = grid_lon.reshape(-1)
        self.indices = indices

    def __len__(self):
        return len(self.temperature)

    def __getitem__(self, idx):
        ind = self.indices[idx]
        lat = self.grid_lat[ind]
        lon = self.grid_lon[ind]
        temp = self.temperature[idx].reshape(-1)
        temp = temp[ind]
        
        # input lat, lon | output temperature | idx for latent variables
        # size (n_points, 2) | (n_points, 1)
        input_x = torch.cat((torch.unsqueeze(lat, dim=1), torch.unsqueeze(lon, dim=1)), dim=1)
        output_y = torch.unsqueeze(temp, dim=1)
        
        # free memory
        del lat, lon, temp, ind
        return input_x, output_y, idx
    

class vizMMGN(Dataset):
    def __init__(self, 
        temperature: torch.Tensor,
        grid_lat: torch.Tensor,
        grid_lon: torch.Tensor,
    ):
        self.temperature = temperature
        self.grid_lat = grid_lat
        self.grid_lon = grid_lon

    def __len__(self):
        return len(self.temperature)

    def __getitem__(self, idx):
        lat = self.grid_lat
        lon = self.grid_lon
        temp = self.temperature[idx]

        # input lat, lon | output temperature | idx for latent variables
        # size (n_points, 2) | (n_points, 1)
        input_x = torch.cat((lat.reshape(-1, 1), lon.reshape(-1, 1)), dim=1)
        output_y = temp.reshape(-1, 1)

        # free memory
        del lat, lon, temp
        return input_x, output_y, idx