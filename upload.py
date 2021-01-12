#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import datetime

import s3fs
import xarray as xr
import yaml
import numpy as np
import requests
import ipdb
from tqdm import tqdm
from intake import open_catalog

requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS = "AES256-SHA"

SECRETS_FILENAME = "secrets.yaml"
SOURCE_CATALOG_URL = (
    "https://raw.githubusercontent.com/eurec4a/eurec4a-intake/master/catalog.yml"
)

LOCAL_CATALOG_FILENAME = "catalog.yml"
DATA_TYPE = "lowres"


def _read_auth_info():
    with open(SECRETS_FILENAME, "r") as fh:
        return yaml.load(fh)


def create_local_dataset(cat, filename):
    datasets = []
    kws = {}
    if DATA_TYPE == "highres":
        dt = datetime.timedelta(hours=1)
        endpoint = cat.barbados.bco.CORAL_LIDAR_highres
        kws['version'] = "2020.09.07/"
    elif DATA_TYPE == "lowres":
        dt = datetime.timedelta(days=1)
        endpoint = cat.barbados.bco.CORAL_LIDAR
    else:
        raise NotImplementedError(DATA_TYPE)

    t_start = datetime.datetime(year=2020, month=1, day=9)
    t_end = datetime.datetime(year=2020, month=2, day=29)
    n_files = int((t_end - t_start)/dt)
    for n in tqdm(range(n_files), desc="downloading source data"):
        date = t_start + n*dt
        cat_entry = endpoint(date=date, content_type="t", **kws)
        ds = cat_entry.to_dask()

        ds["time"] = (
            ("Time"),
            np.datetime64(date) + ds.Time.values.astype("timedelta64[s]"),
        )
        ds = ds.swap_dims(dict(Time="time"))
        datasets.append(ds)

    ds_all = xr.concat(datasets, dim="time")
    ds_all["Altitude"] = ds_all.Altitude.isel(time=0)

    ds_all = ds_all.swap_dims(dict(Length="Altitude"))
    ds_all = ds_all.rename(dict(Altitude="alt"))
    del(ds_all["Time"])

    if "_NCProperties" in ds_all:
        del ds_all.attrs["_NCProperties"]

    ds_all.to_netcdf(filename)
    print("done", flush=True)


def _add_cat_entry(ds, remote_path, description, auth_info):
    print(f"storing in minio.denby.eu eurec4a-environment/{remote_path}...", end="", flush=True)
    s3_access_key = auth_info["s3_access_key"]
    s3_secret_key = auth_info["s3_secret_key"]

    fs = s3fs.S3FileSystem(
        anon=False,
        key=s3_access_key,
        secret=s3_secret_key,
        client_kwargs={"endpoint_url": "https://minio.denby.eu"},
    )

    cat_entry = {
        "description": description,
        "driver": "zarr",
        "args": {
            "urlpath": f"simplecache::s3://eurec4a-environment/{remote_path}",
            "consolidated": True,
            "storage_options": {
                "s3": {
                    "anon": True,
                    "client_kwargs": {"endpoint_url": "https://minio.denby.eu"}, # noqa
                }
            },
        },
    }
    mapper = fs.get_mapper(f"eurec4a-environment/{remote_path}")

    ds.to_zarr(mapper, consolidated=True)

    print("done!", flush=True)

    return cat_entry


def _add_to_local_catalog(name, cat_entry):
    with open(LOCAL_CATALOG_FILENAME, "r") as fh:
        cat = yaml.load(fh)

    if cat["sources"] is None:
        cat["sources"] = {}
    cat.setdefault("sources", {})
    cat['sources'][name] = cat_entry

    with open(LOCAL_CATALOG_FILENAME, "w") as fh:
        fh.write(yaml.dump(cat, default_flow_style=None))

    print(f"Added `{name}` to local catalog `{LOCAL_CATALOG_FILENAME}`")


def _test_local_catalog_entry(name):
    cat = open_catalog(LOCAL_CATALOG_FILENAME)
    print(cat[name].to_dask())


def load_and_cleanup(filename_local):
    print("cleaning...", end="", flush=True)
    ds = xr.open_dataset(filename_local)

    ds["WaterVaporMixingRatio"] = ds.WaterVaporMixingRatio.where(
        ds.WaterVaporMixingRatio < 1.0e32, other=np.nan
    )

    da_q_coral = ds.WaterVaporMixingRatio
    da_q_coral = da_q_coral.where((da_q_coral >= 0.0) * (da_q_coral < 20.0), np.nan)
    ds["WaterVaporMixingRatio"] = da_q_coral

    # only pick out water vapour and temperature variables
    ds = ds[["WaterVaporMixingRatio", "Temperature355"]].compute()
    print("done!", flush=True)

    return ds


def main():
    name = f"coral_{DATA_TYPE}"
    cat = open_catalog(SOURCE_CATALOG_URL)
    local_filename = Path(f"coral_{DATA_TYPE}_local.nc")
    if not local_filename.exists():
        create_local_dataset(cat=cat, filename=local_filename)

    auth_info = _read_auth_info()
    ds = load_and_cleanup(local_filename)

    cat_entry = _add_cat_entry(
        ds=ds,
        remote_path=f"bco/{name}",
        description="CORAL LIDAR at BCO",
        auth_info=auth_info,
    )

    _add_to_local_catalog(name=name, cat_entry=cat_entry)
    _test_local_catalog_entry(name=name)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()