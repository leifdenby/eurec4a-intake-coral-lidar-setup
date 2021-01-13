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
    "https://raw.githubusercontent.com/leifdenby/eurec4a-intake/coral-highres/catalog.yml"
)

LOCAL_CATALOG_FILENAME = "catalog.yml"


def _read_auth_info():
    with open(SECRETS_FILENAME, "r") as fh:
        return yaml.load(fh)


def create_local_dataset(cat, filename, data_resolution):
    datasets = []
    kws = {}
    if data_resolution == "high":
        dt = datetime.timedelta(hours=1)
        kws['version'] = "2020.08.12"
        endpoint = lambda t: cat.barbados.bco.CORAL_LIDAR_highres(t_start=t, **kws)  # noqa
    elif data_resolution == "low":
        dt = datetime.timedelta(days=1)
        kws['version'] = "2020.09.07"
        endpoint = lambda t: cat.barbados.bco.CORAL_LIDAR(date=t, **kws)  # noqa
    else:
        raise NotImplementedError(data_resolution)

    t_start = datetime.datetime(year=2020, month=1, day=9, tzinfo=datetime.timezone.utc)
    t_end = datetime.datetime(year=2020, month=2, day=29, tzinfo=datetime.timezone.utc)
    n_files = int((t_end - t_start)/dt)
    for n in tqdm(range(n_files), desc="downloading source data"):
        t = t_start + n*dt
        cat_entry = endpoint(t=t)
        ds = cat_entry.to_dask()

        ds["time"] = (
            ("Time"),
            [datetime.datetime.utcfromtimestamp(ts) for ts in ds.UnixTime.values]
        )
        ds = ds.swap_dims(dict(Time="time"))
        datasets.append(ds)

    ds_all = xr.concat(datasets, dim="time")
    ds_all["Altitude"] = ds_all.Altitude.isel(time=0)

    ds_all = ds_all.swap_dims(dict(Length="Altitude"))
    ds_all = ds_all.rename(dict(Altitude="alt"))
    del(ds_all["Time"])

    ds_all.attrs['version'] = kws['version']

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


def load_and_cleanup(filename_local, data_resolution):
    print("cleaning...", end="", flush=True)
    ds = xr.open_dataset(filename_local)

    ds["WaterVaporMixingRatio"] = ds.WaterVaporMixingRatio.where(
        ds.WaterVaporMixingRatio < 1.0e32, other=np.nan
    )

    da_q_coral = ds.WaterVaporMixingRatio
    da_q_coral = da_q_coral.where((da_q_coral >= 0.0) * (da_q_coral < 20.0), np.nan)
    ds["WaterVaporMixingRatio"] = da_q_coral

    if data_resolution == "low":
        # only pick out water vapour and temperature variables
        ds = ds[["WaterVaporMixingRatio", "Temperature355"]].compute()
    elif data_resolution == "high":
        ds = ds[["WaterVaporMixingRatio"]].compute()
    else:
        raise NotImplementedError(data_resolution)
    print("done!", flush=True)

    return ds


def main(data_resolution):
    name = f"coral_{data_resolution}res"
    cat = open_catalog(SOURCE_CATALOG_URL)
    local_filename = Path("local-data") / f"{name}_local.nc"
    if not local_filename.exists():
        local_filename.parent.mkdir(exist_ok=True, parents=True)
        create_local_dataset(cat=cat, filename=local_filename, data_resolution=data_resolution)

    auth_info = _read_auth_info()
    ds = load_and_cleanup(local_filename, data_resolution=data_resolution)

    cat_entry = _add_cat_entry(
        ds=ds,
        remote_path=f"bco/{name}",
        description="CORAL LIDAR at BCO",
        auth_info=auth_info,
    )

    _add_to_local_catalog(name=name, cat_entry=cat_entry)
    _test_local_catalog_entry(name=name)


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("resolution", type=str, default="low")
    args = argparser.parse_args()

    with ipdb.launch_ipdb_on_exception():
        main(data_resolution=args.resolution)
