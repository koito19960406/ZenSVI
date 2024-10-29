# encoding: utf-8
# author: yujunhou
# contact: hou.yujun@u.nus.edu

import geopandas as gp
import pandas as pd
import requests


def get_data_from_url(url):
    try:
        r = requests.get(url, timeout=None)
        while r.status_code != 200:
            r = requests.get(url, timeout=None)  # try again
        if r.json()["status"]["apiCode"] == 600:
            data = r.json()["result"]["data"]  # get a JSON format of the response
            return data
        else:
            return None
    except Exception as e:
        print(f"Error: {e}")


def data_to_dataframe(data):
    try:
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        print(f"Error: {e}")


def get_points_in_sequence(sequenceId):
    try:
        url = f"https://api.openstreetcam.org/2.0/sequence/{sequenceId}/photos?itemsPerPage=1000000&join=user,photo,photos,attachment,attachments"
        data = get_data_from_url(url)
        if data:
            df = data_to_dataframe(data)
            points = gp.GeoDataFrame(df, geometry=gp.points_from_xy(df.lng, df.lat))
            return points
        else:
            empty_df = pd.DataFrame()
            return empty_df
    except Exception as e:
        print(f"Error: {e}")


def clip_points_with_shape(points, shape):
    try:
        if not points.empty:
            points = gp.clip(
                points, shape.geometry.unary_union
            )  # clip the points with the union of all polygons in the shape gdf
            return points
        else:
            return points
    except Exception as e:
        print(f"Error: {e}")


def get_sequences_in_shape(shape):
    try:
        ls = []  # empty list to collect sequences
        shape = shape.explode(ignore_index=True)  # explode the shape gdf in case there's any multipolygon in any row
        for _, row in shape.iterrows():
            minx, miny, maxx, maxy = (
                row.geometry.bounds[0],
                row.geometry.bounds[1],
                row.geometry.bounds[2],
                row.geometry.bounds[3],
            )  # find the extent of each polygon geometry
            url = f"https://api.openstreetcam.org/2.0/sequence/?bRight={miny},{maxx}&tLeft={maxy},{minx}&itemsPerPage=1000000"  # use the extent to query for sequences existing in the extent
            data = get_data_from_url(url)
            if data:
                df = data_to_dataframe(data)
                ls.append(df)  # append the collected df of sequences to the list
            else:
                df = pd.DataFrame()  # if 0 sequences collected, create an empty dataframe
                ls.append(df)
        seqs = pd.concat(ls, ignore_index=True)  # concat all collected sequences into a dataframe
        return seqs
    except Exception as e:
        print(f"Error: {e}")


def get_points_in_shape(shape):
    try:
        df_seqs = get_sequences_in_shape(shape)
        if df_seqs.empty:
            print("No data from KartaView.")
            return
        else:
            ls_gdf = []
            for _, seq in df_seqs.iterrows():
                sequenceId = seq["id"]
                points = get_points_in_sequence(sequenceId)
                points = clip_points_with_shape(points, shape)
                ls_gdf.append(points)
            points_all = pd.concat(ls_gdf).reset_index(drop=True)
            if not points_all.empty:
                points_all = (
                    points_all.drop(columns=["cameraParameters", "geometry"])
                    .rename(columns={"lng": "lon"})
                    .join(
                        df_seqs[
                            [
                                "id",
                                "address",
                                "cameraParameters",
                                "countryCode",
                                "deviceName",
                                "distance",
                                "sequenceType",
                            ]
                        ]
                        .set_index("id")
                        .rename(columns={"distance": "distanceSeq"}),
                        on="sequenceId",
                        how="left",
                    )
                )  # append the sequence metadata to each point based on sequence ID
                points_all = points_all.drop_duplicates(subset=["id"])  # remove duplicated points, if any
            nSeqs = 0
            if not points_all.empty:
                nSeqs = points_all["sequenceId"].nunique()
            print(
                "Download complete, collected",
                nSeqs,
                "sequences",
                len(points_all),
                "points",
            )
            return points_all
    except Exception as e:
        print(f"Error: {e}")
