from pathlib import Path
import xarray as xr
from typing import List, Tuple, Dict
import numpy as np


SECONDS_PER_HOUR = 3600
HOUR_PER_DAY = 24
SECOUDS_PER_DAY = SECONDS_PER_HOUR * HOUR_PER_DAY
AVG_DAY_PER_YEAR = 365.25
AVG_SECONDS_PER_YEAR = SECOUDS_PER_DAY * AVG_DAY_PER_YEAR


def preprocess_pressure_data(ds):
    fname = Path(ds.encoding['source']).stem
    pressure_level = int(fname.split('_')[1])
    ds['pressure'] = pressure_level
    ds = ds.set_coords('pressure')
    ds = ds.expand_dims('pressure')
    return ds


# UNIX time start from 1970-01-01 00:00:00 UTC
# FIXME: might be inaccurate since we are using average days per year
def get_year_progress(seconds_since_start: np.ndarray) -> np.ndarray:
    years_since_start = (seconds_since_start /
                         np.float64(AVG_SECONDS_PER_YEAR))
    return np.mod(years_since_start, 1.0).astype(np.float32)


def get_day_progress(seconds_since_start: np.ndarray,
                     longitude: np.ndarray) -> np.ndarray:
    day_progress_greenwich = (
        np.mod(seconds_since_start, SECOUDS_PER_DAY) / SECOUDS_PER_DAY
    )
    longitude_offsets = np.deg2rad(longitude) / (2 * np.pi)
    day_progress = np.mod(
        day_progress_greenwich[..., np.newaxis] + longitude_offsets, 1.0
    )
    return day_progress.astype(np.float32)


# FIXME: this is how graphcast featurize the progress
# TODO: update this
def featurize_progress(name: str,
                       progress,
                       dim=('time',)) -> Dict[str, xr.DataArray]:
    progress_phase = progress * 2 * np.pi
    return {
        name: xr.Variable(dims=dim, data=progress_phase),
        f'{name}_sin': xr.Variable(dims=dim, data=np.sin(progress_phase)),
        f'{name}_cos': xr.Variable(dims=dim, data=np.cos(progress_phase))
    }


def variable_to_stacked(
    variable: xr.Variable,
    sizes,
    preserved_dims,
) -> xr.Variable:
    stack_to_channels_dims = [
        d for d in variable.dims if d not in preserved_dims]
    if stack_to_channels_dims:
        variable = variable.stack(channels=stack_to_channels_dims)
    dims = {dim: variable.sizes.get(
        dim) or sizes[dim] for dim in preserved_dims}
    dims["channels"] = variable.sizes.get("channels", 1)
    return variable.set_dims(dims)


def dataset_to_stacked(
    dataset: xr.Dataset,
    preserved_dims=("latitude", "longitude", "time"),
) -> xr.DataArray:
    data_vars = [
        variable_to_stacked(dataset.variables[name],  dataset.sizes,
                            preserved_dims)
        for name in sorted(dataset.data_vars.keys())
    ]
    coords = {
        dim: coord
        for dim, coord in dataset.coords.items()
        if dim in preserved_dims
    }
    return xr.DataArray(
        data=xr.Variable.concat(data_vars, dim="channels"), coords=coords)


def stacked_to_dataset(
        stacked_array: xr.Variable,
        template: xr.Dataset,
        preserved_dims=("latitude", "longitude", "time"),
):
    channels_sizes = {}
    var_names = sorted(template.data_vars.keys())
    for name in var_names:
        template_var = template[name]
        if not all(dim in template_var.dims for dim in preserved_dims):
            raise ValueError(
                f"Variable {name} in template does not contain all preserved dims"
            )
        channels_sizes[name] = {
            dim: size for dim, size in template_var.sizes.items() if dim not in preserved_dims
        }

    channels = {name: np.prod(list(unstack_sizes.values()), dtype=np.int64)
                for name, unstack_sizes in channels_sizes.items()}
    total_channels = sum(channels.values())
    found_channels = stacked_array.sizes['channels']

    if found_channels != total_channels:
        raise ValueError(
            f"Expected {total_channels} channels, but found {found_channels}"
        )

    data_vars = {}
    index = 0
    for name in var_names:
        template_var = template[name]
        var = stacked_array.isel(
            {"channels": slice(index, index + channels[name])})
        index += channels[name]
        var = var.unstack({"channels": channels_sizes[name]})
        var = var.transpose(*template_var.dims)
        data_vars[name] = xr.DataArray(
            data=var,
            coords=template_var.coords,
            # This might not always be the same as the name it's keyed under; it
            # will refer to the original variable name, whereas the key might be
            # some alias e.g. temperature_850 under which it should be logged:
            name=template_var.name,
        )
    # pytype:disable=not-callable,wrong-arg-count
    return type(template)(data_vars)
