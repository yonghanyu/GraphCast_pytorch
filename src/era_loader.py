
import xarray as xr
import numpy as np
import einops
import yaml
from pathlib import Path
from collections import OrderedDict
import torch
from torch.utils.data import Dataset
from data_util import *
from typing import Dict
import zarr
import dask

# CONSTANTS:
MIN_LATITUDE = -90
MAX_LATITUDE = 90
MIN_LONGITUDE = 0
MAX_LONGITUDE = 360


def _check_uniform_spacing_and_get_delta(vector):
    diff = np.diff(vector)
    if not np.all(np.isclose(diff[0], diff)):
        raise ValueError(f'Vector {diff} is not uniformly spaced.')
    return diff[0]


def _weight_for_latitude_vector_with_poles(latitude):
    """Weights for uniform latitudes of the form [+- 90, ..., -+90]."""
    delta_latitude = np.abs(_check_uniform_spacing_and_get_delta(latitude))
    if (not np.isclose(np.max(latitude), 90.) or
            not np.isclose(np.min(latitude), -90.)):
        raise ValueError(
            f'Latitude vector {latitude} does not start/end at +- 90 degrees.')
    weights = np.cos(np.deg2rad(latitude)) * \
        np.sin(np.deg2rad(delta_latitude/2))
    # The two checks above enough to guarantee that latitudes are sorted, so
    # the extremes are the poles
    weights[[0, -1]] = np.sin(np.deg2rad(delta_latitude/4)) ** 2
    return weights


class ERA5DataConfig:
    def __init__(self, cfg_path):
        self.cfg_path = Path(cfg_path)
        if not self.cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        self.cfg = self.__load()
        self.__surface_variables = OrderedDict()
        self.__pressure_variables = OrderedDict()
        for var in self.cfg['surface_variables']:
            self.__surface_variables[var['name']] = var
        for var in self.cfg['pressure_variables']:
            self.__pressure_variables[var['name']] = var

    def __load(self):
        with open(self.cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg

    @property
    def resolution(self):
        return self.cfg['resolution']

    @property
    def pressure_levels(self):
        return self.cfg['pressure_levels']

    @property
    def interval(self):
        return self.cfg['interval']

    @property
    def start_date(self):
        return self.cfg['start_date']

    @property
    def end_date(self):
        return self.cfg['end_date']

    @property
    def pressure_data_dir(self):
        return self.cfg['pressure_data_dir']

    @property
    def surface_data_dir(self):
        return self.cfg['surface_data_dir']

    @property
    def pressure_variables(self):
        return self.__pressure_variables

    @property
    def surface_variables(self):
        return self.__surface_variables


class ERA5Stats:
    def __init__(self,
                 cfg,
                 mean_path,
                 std_path,
                 std_diff_path) -> None:
        self.pressure_name_map = dict()
        self.surface_name_map = dict()
        self.pressure_levels = cfg.pressure_levels

        for name, name_tup in cfg.pressure_variables.items():
            self.pressure_name_map[name_tup['long_name']] = name
        for name, name_tup in cfg.surface_variables.items():
            self.surface_name_map[name_tup['long_name']] = name

        if 'geopotential' in self.surface_name_map:
            self.surface_name_map['geopotential_at_surface'] = self.surface_name_map['geopotential']
            del self.surface_name_map['geopotential']

        mean_path = Path(mean_path)
        if not mean_path.exists():
            raise FileNotFoundError(f"Stats file not found: {mean_path}")
        self.std_diff_stats = xr.open_dataset(std_diff_path)
        self.mean_stats = xr.open_dataset(mean_path)
        self.std_stats = xr.open_dataset(std_path)
        self.std_diff_pressure_stats = self.__select_var(self.std_diff_stats)
        self.mean_pressure_stats = self.__select_var(self.mean_stats).copy()
        self.std_pressure_stats = self.__select_var(self.std_stats)

        self.mean_surface_stats = self.__select_var(
            self.mean_stats, pressure_vars=False, time_feature=False)
        self.std_surface_stats = self.__select_var(
            self.std_stats, pressure_vars=False, time_feature=False)
        self.std_diff_surface_stats = self.__select_var(
            self.std_diff_stats, False, False)
        # print(self.mean_pressure_stats)
        self.mean_stats.close()
        self.std_diff_stats.close()
        self.std_stats.close()

    def __select_var(self,
                     ds: xr.Dataset,
                     pressure_vars=True,
                     time_feature=False):
        name_maps = self.pressure_name_map if pressure_vars else self.surface_name_map
        var_to_select = list(name_maps.keys())
        if time_feature:
            var_to_select += ['day_progress',
                              'day_progress_sin',
                              'day_progress_cos',
                              'year_progress',
                              'year_progress_sin',
                              'year_progress_cos']

        var_to_select_ = []
        for k in var_to_select:
            if k in ds.data_vars:
                var_to_select_.append(k)
        var_to_select = var_to_select_
        data = ds[var_to_select]

        name_maps = {**name_maps}
        ks = list(name_maps.keys())
        for k in ks:
            if k not in var_to_select:
                del name_maps[k]
        if pressure_vars:
            name_maps = {**name_maps, 'level': 'pressure'}
        else:
            name_maps = {**name_maps}
        data = data.rename(name_maps)
        if pressure_vars:
            data = data.sel(pressure=self.pressure_levels)
        data = data.load()

        return data


# the dataset should be in netcdf4 format

# TODO: now the one day of data contains 24 data points, change is required if the interval is not 1 hour
# assuming zarr
# Notice that data is arranged in the following way:
# longitude: 0.0, 0.25, 0.5, 0.75, 1.0, ..., 359.75
# latitude: 90.0, 89.75, 89.5, 89.25, 89.0, ..., -90.0
class ERA5Dataset(Dataset):
    # we assume the data always start from january 1st, 00:00 and end at december 31st, 23:00 for simplicity
    def __init__(self,
                 cfg,
                 # data_dir,
                 pressure_data_dir,
                 surface_data_dir,
                 mode='train',
                 epoch_len=1,
                 normalize=False,
                 fillna=False,
                 add_time_feature=False,
                 data_mean_path=None,
                 sampling_freq=3,
                 data_std_path=None,
                 data_diff_std_path=None):

        self.cfg = cfg
        assert self.cfg.resolution % 0.25 == 0, "Resolution should be multiple of 0.25"

        self.mode = mode
        # self.data_dir = Path(data_dir)
        self.pressure_data_dir = Path(pressure_data_dir)
        self.surface_data_dir = Path(surface_data_dir)
        self.epoch_len = epoch_len
        self.normalize = normalize
        self.fillna = fillna
        self.sampling_freq = sampling_freq
        self.add_time_feature = add_time_feature

        self.era5_stats = ERA5Stats(
            cfg, data_mean_path, data_std_path, data_diff_std_path)

        # data is stored in zarr format, in yearly manner
        pressure_fnames, surface_fnames = self.__parse_fname()
        self.pressure_var_names = list(self.cfg.pressure_variables.keys())
        self.surface_var_names = list(self.cfg.surface_variables.keys())

        self.latitude = np.arange(
            MIN_LATITUDE, MAX_LATITUDE + self.cfg.resolution, self.cfg.resolution)[:: -1]

        # fix this, change 721/1440 to variable
        self.latitude_idx = np.arange(
            0, 721, int(self.cfg.resolution / 0.25))

        self.longitude = np.arange(
            MIN_LONGITUDE, MAX_LONGITUDE, self.cfg.resolution)
        self.longitude_idx = np.arange(
            0, 1440, int(self.cfg.resolution / 0.25))

        self.pressure_datasets, time_steps, self.total_time_step = self.__open_datasets(
            pressure_fnames)
        self.surface_datasets, _, total_time_step = self.__open_datasets(
            surface_fnames)

        # find the pressure level index
        all_pressure_levels = list(self.pressure_datasets[list(
            self.pressure_datasets.keys())[0]]['pressure'])

        self.pressure_idx = [all_pressure_levels.index(
            p) for p in self.cfg.pressure_levels]
        if len(self.pressure_idx) == len(all_pressure_levels):
            self.pressure_idx = None

        # assert self.total_time_step == total_time_step, "Total time steps of pressure and surface data should be the same"

        # do this in case of missalignment of year
        all_years = sorted(list(time_steps.keys()))
        self.start_year = all_years[0]
        self.end_year = all_years[-1]

        assert len(self.pressure_datasets) == len(
            self.surface_datasets), "Should have the same number of datasets for pressure and surface data"

        self.idx_map, self.year_idx_offset = self.__build_idx_map(time_steps)

        self.n_channel = len(self.cfg.pressure_levels) * \
            len(self.pressure_var_names) + len(self.cfg.surface_variables)

        self.n_forcing_channel = 6
        self.subtract_num = 0
        if 'lsm' in self.surface_var_names:
            self.n_channel -= 1
            self.n_forcing_channel += 1
            self.subtract_num += 1
        if 'z' in self.surface_var_names:
            self.n_channel -= 1
            self.n_forcing_channel += 1
            self.subtract_num += 1
        if 'tisr' in self.surface_var_names:
            self.n_channel -= 1
            self.n_forcing_channel += 1
            self.subtract_num += 1

        self.loss_weight = self.__build_loss_weight(
            self.cfg, self.pressure_var_names, self.surface_var_names)

        self.mean, self.std, self.forcing_mean, self.forcing_std, self.diff_std = self.__build_normalization_factor()
        self.idx2var_name = self.__build_idx_to_varname()
        self.land_sea_mask = None
        self.geopotential_at_surface = None

    def __build_forcing(self, year, year_idx, tisr=None):
        date = np.datetime64(str(year)) + np.timedelta64(year_idx, 'h')
        seconds_since_start = date.astype('datetime64[s]').astype(np.int64)
        day_progress = get_day_progress(
            seconds_since_start, self.longitude) * 2 * np.pi
        year_progress = get_year_progress(seconds_since_start) * 2 * np.pi

        # progress + land sea mask
        # shape [9, lat, lon]
        forcing = np.empty((self.n_forcing_channel, len(
            self.latitude_idx), len(self.longitude_idx)), dtype=np.float32)
        forcing[0, :, :] = self.land_sea_mask
        forcing[1, :, :] = self.geopotential_at_surface
        if tisr is not None:
            forcing[2, :, :] = tisr
            forcing[3, :, :] = day_progress
            forcing[4, :, :] = np.sin(day_progress)
            forcing[5, :, :] = np.cos(day_progress)
            forcing[6, :, :] = year_progress
            forcing[7, :, :] = np.sin(year_progress)
            forcing[8, :, :] = np.cos(year_progress)

        else:
            forcing[2, :, :] = day_progress
            forcing[3, :, :] = np.sin(day_progress)
            forcing[4, :, :] = np.cos(day_progress)
            forcing[5, :, :] = year_progress
            forcing[6, :, :] = np.sin(year_progress)
            forcing[7, :, :] = np.cos(year_progress)
        return forcing

    def __build_idx_to_varname(self):
        self.idx_to_varname = dict()
        for i, var_name in enumerate(self.pressure_var_names):
            for j, level in enumerate(self.cfg.pressure_levels):
                self.idx_to_varname[i * len(self.cfg.pressure_levels) +
                                    j] = f"{var_name}_{level}"
        offset = (i + 1) * len(self.cfg.pressure_levels)

        i = 0
        for var_name in self.surface_var_names:
            if var_name in ['lsm', 'z', 'tisr']:
                continue
            self.idx_to_varname[offset + i] = var_name
            i += 1

        # print('self.n_channel shape:', self.n_channel)
        assert offset + i == self.n_channel, "Number of channels should be the same"
        return self.idx_to_varname

    def __build_normalization_factor(self):
        mean = np.zeros((self.n_channel), dtype=np.float32)
        std = np.zeros((self.n_channel), dtype=np.float32)
        std_diff = np.zeros((self.n_channel), dtype=np.float32)
        mean_forcing = np.zeros((self.n_forcing_channel), dtype=np.float32)
        std_forcing = np.ones((self.n_forcing_channel), dtype=np.float32)

        n_pressure_level = len(self.cfg.pressure_levels)
        for i, var_name in enumerate(self.pressure_var_names):
            s, e = i * n_pressure_level, (i + 1) * n_pressure_level
            mean[s:e] = self.era5_stats.mean_pressure_stats[var_name].values
            std[s:e] = self.era5_stats.std_pressure_stats[var_name].values
            std_diff[s:e] = self.era5_stats.std_diff_pressure_stats[var_name].values
        offset = e
        i = 0
        for var_name in self.surface_var_names:
            if var_name in ['lsm', 'z', 'tisr']:
                continue
            if var_name in ['tp']:
                mean[offset + i] = 0.
                std[offset + i] = 1.
            else:
                mean[offset + i] = self.era5_stats.mean_surface_stats[var_name].values
                std[offset + i] = self.era5_stats.std_surface_stats[var_name].values
                std_diff[offset +
                         i] = self.era5_stats.std_diff_surface_stats[var_name].values
            i += 1

        i = 0
        for var_name in ['lsm', 'z', 'tisr']:
            if var_name not in self.surface_var_names:
                continue
            mean_forcing[i] = self.era5_stats.mean_surface_stats[var_name].values
            std_forcing[i] = self.era5_stats.std_surface_stats[var_name].values
            i += 1

        return mean[None, :, None, None], std[None, :, None, None], mean_forcing[None, :, None, None], std_forcing[None, :, None, None], std_diff[None, :, None, None]

    def __build_idx_map(self, time_steps: OrderedDict):
        """__build_idx_map build a map from index to year

        Args:
            time_steps (Dict): a dictionary of year to each year's time steps
        """

        idx_map = OrderedDict()
        year_idx_offset = OrderedDict()
        accum_time_step = 0
        for year, yearly_time_step in time_steps.items():
            for i in range(yearly_time_step):
                idx_map[accum_time_step + i] = year
            year_idx_offset[year] = accum_time_step
            accum_time_step += yearly_time_step

        assert accum_time_step == self.total_time_step, "Total time steps should be the same"
        return idx_map, year_idx_offset

    def __translate_idx(self, idx):
        """__translate_idx translated the index of dataset to year and index of the zarr file

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        year = self.idx_map[idx]
        offset = self.year_idx_offset[year]
        time_step = idx - offset
        return time_step * self.cfg.interval

    def __build_area_weight(self):
        weight = _weight_for_latitude_vector_with_poles(self.latitude)
        weight = torch.from_numpy(weight)
        return einops.rearrange(weight, 'l -> 1 1 l 1')

    def __build_loss_weight(self, cfg, pressure_var_names, surface_var_names):
        slope = 1. / sum(cfg.pressure_levels)
        pressure_weight = np.ones(len(cfg.pressure_levels), dtype=np.float32)
        pressure_weight = pressure_weight * slope * cfg.pressure_levels

        single_var_names = []
        for i in surface_var_names:
            if i in ['z', 'lsm', 'tisr']:
                continue
            single_var_names.append(i)
        n_surface = len(single_var_names)
        n_pressure = len(pressure_var_names)

        surface_weight = np.ones(n_surface, dtype=np.float32)

        for i, surface_var in enumerate(single_var_names):
            surface_weight[i] = 0.1
            if surface_var == 't2m' or surface_var == 'tcc':
                surface_weight[i] = 1
        surface_weight = torch.from_numpy(surface_weight)
        pressure_weight = torch.from_numpy(pressure_weight)
        pressure_weight = einops.repeat(
            pressure_weight, 'l -> n l', n=n_pressure).contiguous()
        pressure_weight = pressure_weight.view(-1)
        weight = torch.cat([pressure_weight, surface_weight], dim=-1)
        return einops.rearrange(weight, 'n -> 1 n 1 1')

    def __parse_fname(self):
        """__parse_fname return a list of pressure and surface data file names, ordered by year

        Returns:
            _type_: _description_
        """
        start_year = int(self.cfg.start_date.year)
        end_year = int(self.cfg.end_date.year)
        # prefix = str(self.data_dir) + '/'
        pressure_prefix = str(self.pressure_data_dir) + '/'
        surface_prefix = str(self.surface_data_dir) + '/'

        pressure_fnames = OrderedDict()
        surface_fnames = OrderedDict()

        for year in range(start_year, end_year + 1):
            pressure_fname = f"{year}.zarr"
            if Path(pressure_prefix + pressure_fname).exists():
                pressure_fnames[year] = pressure_prefix + pressure_fname
            else:
                continue
            # else:
            #    print(f"File {pressure_prefix + pressure_fname} does not exist")
            #    exit(1)
            surface_fname = f"{year}.zarr"
            if Path(surface_prefix + surface_fname).exists():
                surface_fnames[year] = surface_prefix + surface_fname
            # else:
            #    print(f"File {surface_prefix + surface_fname} does not exist")
            #    exit(1)
        return pressure_fnames, surface_fnames

    def __open_datasets(self, fnames: OrderedDict):
        """__open_datasets open the datasets from the file names and select the variables
                           determine the time steps of each dataset
        Args:
            fnames (Dict): _description_

        Returns:
            _type_: _description_
        """
        datasets = OrderedDict()
        timesteps = OrderedDict()

        total_time_steps = 0

        for year in fnames.keys():
            fname = fnames[year]
            ds = zarr.open_consolidated(fname)
            # ds = xr.open_dataset(fname, engine='zarr', consolidated=True, chunks={'time': 1}, decode_cf=True, mask_and_scale=True)
            datasets[year] = ds
            time_step_year = len(ds['time'])
            assert time_step_year % self.cfg.interval == 0, f'Time step of year {year} is not divisible by interval {self.cfg.interval}'
            timesteps[year] = time_step_year // self.cfg.interval
            total_time_steps += timesteps[year]
        return datasets, timesteps, total_time_steps

    def test_nan(self, datapoint):
        # check if there is any nan value in the datapoint
        # ERA5 data is already preprocessed, so we assume there is no nan value
        if False in datapoint.notnull():
            return True

    def __sample_single_step(self, data_idx, store_idx, store: np.ndarray, pressure_dataset, surface_dataset):
        """__sample_single_step read single step of data from the dataset into store
            dataset has shape [all_pressure_level, time, lat, lon] for atmospheric data and [time, lat, lon] for surface data 
            store has shape [time, channel, downsampled_lat, downsampled_lon] 
            num_of_channel = selected_pressure_level * num_of_var + num_of_surface_var

        Args:
            data_idx (_type_): index of time step in dataset
            store_idx (_type_): index of time step in store
            store (_type_): np.ndarray of shape [time, channel, lat, lon]
            pressure_dataset (_type_): zarr dataset of atmospheric data
            surface_dataset (_type_): zarr dataset of surface data
        """

        def read_sel_data(sel_pressure=False):
            # fuse read and select with np take
            if self.cfg.resolution == 0.25:
                # no need to downsample resolution
                if not sel_pressure:
                    # no need to select pressure level
                    return var[:, data_idx, :, :] if len(var.shape) == 4 else var[data_idx, :, :]
                else:
                    return np.take(var[:, data_idx, :, :], self.pressure_idx, axis=0)
            else:
                if not sel_pressure:
                    if len(var.shape) == 4:
                        d = np.take(np.take(var[:, data_idx, :, :], self.latitude_idx, axis=-2), self.longitude_idx,
                                    axis=-1)
                        return d
                    else:
                        d = np.take(np.take(var[data_idx, :, :], self.latitude_idx, axis=-2), self.longitude_idx,
                                    axis=-1)
                        return d
                else:
                    return np.take(
                        np.take(
                            np.take(var[:, data_idx, :, :],
                                    self.latitude_idx, axis=-2),
                            self.longitude_idx, axis=-1),
                        self.pressure_idx, axis=0)

        n_pressure_level = len(self.cfg.pressure_levels)
        sel_pressure = False
        if self.pressure_idx is not None:
            sel_pressure = True
        # read & sel the pressure data
        for i, var_name in enumerate(self.pressure_var_names):
            var = pressure_dataset[var_name]
            s, e = i * n_pressure_level, (i + 1) * n_pressure_level
            store[store_idx, s:e, :, :] = read_sel_data(sel_pressure)

        # read & sel the surface data
        offset = e
        i = 0
        for var_name in self.surface_var_names:
            if var_name in ['lsm', 'z', 'tisr']:
                continue
                # print(surface_dataset)
            var = surface_dataset[var_name]
            # store[store_idx, offset + i, :, :] = read_sel_data(var[data_idx, :, :], False)
            store[store_idx, offset + i, :, :] = read_sel_data(False)
            i += 1

        if self.land_sea_mask is None:
            assert 'lsm' in self.surface_var_names, "Land sea mask is not provided"
            var = surface_dataset['lsm']
            self.land_sea_mask = np.empty(
                (len(self.latitude_idx), len(self.longitude_idx)), dtype=np.float32)
            self.land_sea_mask[:] = read_sel_data(False)

        if self.geopotential_at_surface is None:
            assert 'z' in self.surface_var_names, "Geopotential at surface is not provided"
            var = surface_dataset['z']
            self.geopotential_at_surface = np.empty(
                (len(self.latitude_idx), len(self.longitude_idx)), dtype=np.float32)
            self.geopotential_at_surface[:] = read_sel_data(False)

        tisr = None
        if 'tisr' in self.surface_var_names:
            var = surface_dataset['tisr']
            tisr = read_sel_data(False)
        return tisr

    def __sample_data(self, idx, epoch_len: int = 1):
        """__sample_data select the data from the datasets based on the index and epoch length
                        for atomspheric data, the shape is [pressure_level, time, lat, lon]
                        for surface data, the shape is [time, lat, lon]

        Args:
            idx (_type_): _description_
            datasets (_type_): _description_
            epoch_len (int, optional): _description_. Defaults to 1.

        Returns:
            np.ndarray: a numpy array of shape [epoch_len, channel, lat, lon]
        """

        # since the dataset is not continuous, we need to sample the data from different years
        # this may happen when the idx is at the end of the year, and the epoch_len is greater than 1
        idxes_to_sample = OrderedDict()
        for i in range(epoch_len):
            idx_to_sample = idx + i     # * self.cfg.interval
            year = self.idx_map[idx_to_sample]
            if year not in idxes_to_sample:
                idxes_to_sample[year] = []
            idxes_to_sample[year].append(self.__translate_idx(idx_to_sample))

        # sort the years
        ordered_years = sorted(list(idxes_to_sample.keys()))
        data = np.empty((epoch_len, self.n_channel, len(
            self.latitude_idx), len(self.longitude_idx)), dtype=np.float32)
        forcing = np.empty((epoch_len, self.n_forcing_channel, len(
            self.latitude_idx), len(self.longitude_idx)), dtype=np.float32)

        store_idx = 0
        for year in ordered_years:
            pressure_dataset = self.pressure_datasets[year]
            surface_dataset = self.surface_datasets[year]
            for _, idx in enumerate(idxes_to_sample[year]):
                tisr = self.__sample_single_step(
                    idx, i, data, pressure_dataset, surface_dataset)
                forcing[store_idx, :, :, :] = self.__build_forcing(year, idx, tisr)
                store_idx += 1
        return data, forcing

    def __getitem__(self, idx):
        # idx is the starting index of prediction epoch
        # return data point sampled by interval, resolution and pressure level.
        # shape [epoch_len, channel, lat, lon]
        idx = idx * self.sampling_freq
        if idx + self.epoch_len > self.total_time_step:
            # warnings.warn("Index exceeds dataset length; returning last available data.")
            idx = self.total_time_step - self.epoch_len

        data, forcing = self.__sample_data(idx, self.epoch_len)

        # print('>>>>>> data shape in dataloader:', data.shape)
        input_data = data[:-1, :, :, :]
        forcing = forcing[:-1, :, :, :]
        output_data = data[1:, :, :, :]
        # output_data = output_data - input_data
        if self.normalize:
            input_data = (input_data - self.mean) / self.std
            forcing = (forcing - self.forcing_std) / self.forcing_std

        return {
            'data': input_data,  # [epoch_len, channel, lat, lon]
            'target': output_data,
            'forcing': forcing,  # [channel, lat, lon]
        }

    def __len__(self):
        return self.total_time_step // self.sampling_freq



