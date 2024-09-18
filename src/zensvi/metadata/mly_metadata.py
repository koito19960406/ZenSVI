from pathlib import Path
import pandas as pd
from typing import Union
import h3
import osmnx as ox
import numpy as np
from shapely.geometry import LineString, Polygon
import geopandas as gpd
from astral import LocationInfo, sun
import pytz
import datetime
from tqdm.auto import tqdm
from timezonefinder import TimezoneFinder
import polars as pl
from shapely.wkb import dumps as wkb_dumps, loads as wkb_loads

def _calculate_angle(line):
    if len(line) > 1:
        start, end = line[0], line[-1]
        angle = np.arctan2(end[1] - start[1], end[0] - start[0])
        return np.degrees(angle) % 360
    else:
        return None

def _create_hexagon(df, resolution=7):
    df = df.with_columns(
        pl.struct(["lat", "lon"])
        .apply(lambda x: h3.geo_to_h3(x["lat"], x["lon"], resolution))
        .alias("h3_id")
    )
    unique_h3_ids = df.select("h3_id").unique()
    hex_gdf = gpd.GeoDataFrame(
        unique_h3_ids.to_pandas(),
        geometry=[Polygon(h3.h3_to_geo_boundary(h, geo_json=True)) for h in unique_h3_ids["h3_id"]],
        crs=4326
    )
    return hex_gdf

def _day_or_night(df):
    df = df.with_columns(pl.col("local_datetime").dt.date().alias("date"))
    unique_locations_dates = df.select(["lat", "lon", "date", "timezone"]).unique()
    sun_times_list = []
    for row in unique_locations_dates.iter_rows():
        location = LocationInfo(latitude=row[0], longitude=row[1])
        s = sun.sun(location.observer, date=row[2], tzinfo=row[3])
        sun_times_list.append({
            "lat": row[0], "lon": row[1], "date": row[2],
            "sunrise": s["sunrise"].astimezone(pytz.UTC),
            "sunset": s["sunset"].astimezone(pytz.UTC),
        })
    sun_times_df = pl.DataFrame(sun_times_list)
    df = df.join(sun_times_df, on=["lat", "lon", "date"])
    df = df.with_columns(
        pl.when((pl.col("sunrise").cast(pl.Datetime("us")) <= pl.col("datetime_utc").cast(pl.Datetime("us"))) & 
                (pl.col("datetime_utc").cast(pl.Datetime("us")) <= pl.col("sunset").cast(pl.Datetime("us"))))
        .then(pl.lit("daytime"))
        .otherwise(pl.lit("nighttime"))
        .alias("daytime_nighttime")
    )
    return df.drop(["date", "sunrise", "sunset"])

def _compute_speed(df):
    # Convert DataFrame to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df.to_pandas(),
        geometry=gpd.points_from_xy(df["lon"], df["lat"], crs="EPSG:4326")
    )
    
    # Project to a local CRS once
    gdf = ox.project_gdf(gdf)
    
    # Convert projected coordinates to WKB
    gdf['geometry_wkb'] = gdf['geometry'].apply(lambda geom: wkb_dumps(geom, output_dimension=2))
    
    # Convert back to Polars DataFrame
    df = pl.DataFrame(gdf.drop(columns=['geometry']))
    
    # Sort the DataFrame by sequence_id and datetime_utc
    df = df.sort(["sequence_id", "datetime_utc"])
    
    # Convert datetime_utc to datetime if it's a string
    if df.schema["datetime_utc"] == pl.Utf8:
        df = df.with_columns(
            pl.col("datetime_utc")
            .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
            .alias("datetime_utc")
        )
    
    # Create shifted columns for sequence ID, geometry, and datetime
    df = df.with_columns([
        df["sequence_id"].shift(-1).alias("shifted_sequence_id"),
        df["geometry_wkb"].shift(-1).alias("shifted_geometry_wkb"),
        df["datetime_utc"].shift(-1).alias("shifted_datetime"),
    ])
    
    # Explicitly filter out null geometries and mismatched sequence IDs before any calculations
    valid_rows = df.filter(
        (df["sequence_id"] == df["shifted_sequence_id"]) &
        df["sequence_id"].is_not_null() &
        df["geometry_wkb"].is_not_null() &
        df["shifted_geometry_wkb"].is_not_null()
    )
    
    # Now compute distances and time differences on the pre-filtered data
    def compute_distance(row):
        if row["geometry_wkb"] is None or row["shifted_geometry_wkb"] is None:
            return None
        geom1 = wkb_loads(row["geometry_wkb"])
        geom2 = wkb_loads(row["shifted_geometry_wkb"])
        return geom1.distance(geom2)

    valid_rows = valid_rows.with_columns([
        pl.struct(["geometry_wkb", "shifted_geometry_wkb"])
        .map_elements(compute_distance, return_dtype=pl.Float64, skip_nulls=False)
        .alias("distance_m"),
        ((pl.col("shifted_datetime") - pl.col("datetime_utc")).dt.total_seconds() / 3600).alias("time_diff_hrs"),
    ])


    
    # Calculate speed in km/h
    valid_rows = valid_rows.with_columns(
        (pl.col("distance_m") / 1000 / pl.col("time_diff_hrs")).alias("speed_kmh")
    )
    
    # Drop unnecessary columns and combine with the original dataframe to retain all rows
    valid_rows = valid_rows.select(["sequence_id", "datetime_utc", "speed_kmh"])
    df = df.join(valid_rows, on=["sequence_id", "datetime_utc"], how="left")
    df = df.drop(["shifted_sequence_id", "shifted_geometry_wkb", "shifted_datetime", "geometry_wkb"])
    
    # Convert Inf, "", and NaN to None
    df = df.with_columns(
        pl.when(
            pl.col("speed_kmh").is_null() | 
            pl.col("speed_kmh").is_nan() | 
            pl.col("speed_kmh").is_infinite()
        )
        .then(None)
        .otherwise(pl.col("speed_kmh"))
        .alias("speed_kmh")
    )
    
    return df
    
def _datetime_to_season(local_datetime, lat):
    season_month_north = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
                          6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
    season_month_south = {12: "Summer", 1: "Summer", 2: "Summer", 3: "Autumn", 4: "Autumn", 5: "Autumn",
                          6: "Winter", 7: "Winter", 8: "Winter", 9: "Spring", 10: "Spring", 11: "Spring"}
    
    if isinstance(local_datetime, datetime.datetime):
        month = local_datetime.month
    elif isinstance(local_datetime, str):
        month = datetime.datetime.fromisoformat(local_datetime).month 
    else:
        raise ValueError("local_datetime must be either a datetime object or a string")
    
    return season_month_north[month] if lat >= 0 else season_month_south[month]


class MLYMetadata:
    """A class to compute metadata for the MLY dataset.

    :param path_input: path to the input CSV file (e.g., "mly_pids.csv"). The CSV file should contain the following columns: "id", "lat", "lon", "captured_at", "compass_angle", "creator_id", "sequence_id", "organization_id", "is_pano".
    :type path_input: Union[str, Path]
    """

    def __init__(self, path_input: Union[str, Path]):
        self._tf_instance = TimezoneFinder()
        self.path_input = Path(path_input)
        self.df = pl.read_csv(self.path_input)
        self.metadata = None
        # get street network in the extent of the dataset with OSMnx
        self.street_network = ox.graph_from_bbox(
            bbox=(
                self.df["lat"].min(),
                self.df["lat"].max(),
                self.df["lon"].min(),
                self.df["lon"].max(),
            ),
            network_type="all",
        )
        self.street_network = ox.convert.graph_to_gdfs(self.street_network, nodes=False)
        self.street_network = ox.projection.project_gdf(self.street_network)
        self.projected_crs = self.street_network.crs
        # calculate angle of the street segments
        # Prepare the data for the Polars DataFrame
        # Assuming self.street_network["geometry"] contains LineString objects
        geometry_list = [list(line.coords) for line in self.street_network["geometry"]]
        self.street_network["geometry_list"] = geometry_list
        street_network_data = self.street_network.drop(
            columns=[col for col in self.street_network.columns if col not in ["geometry_list", "geometry"]]
        )

        # Create the Polars DataFrame with specified data types
        self.street_network = pl.DataFrame({
            "geometry_list": pl.Series(street_network_data["geometry_list"], dtype=pl.List(pl.List(pl.Float64)))
        })

        # Continue with the rest of the initialization
        self.street_network = self.street_network.with_columns(
            pl.col("geometry_list").apply(lambda x: _calculate_angle(x)).alias("angle")
        )
        self.street_network = self.street_network.drop("geometry_list")
        self.street_network = gpd.GeoDataFrame(self.street_network.to_pandas(), geometry=street_network_data.reset_index().geometry)

        # create a dictionary of functions to compute metadata for each indicator at the image level
        self.indicator_metadata_image = {
            "year": self._compute_year_metadata_image,
            "month": self._compute_month_metadata_image,
            "day": self._compute_day_metadata_image,
            "hour": self._compute_hour_metadata_image,
            "day_of_week": self._compute_day_of_week_metadata_image,
            "daytime_nighttime": self._compute_daytime_nighttime_metadata_image,
            "season": self._compute_season_metadata_image,
            "relative_angle": self._compute_relative_angle_metadata_image,
            "h3_id": self._compute_h3_id_metadata_image,
            "speed_kmh": self._compute_speed_metadata_image,
        }
        # create a dictionary of functions to compute metadata for each indicator at the grid level
        self.indicator_metadata_grid_street = {
            "coverage": self._compute_coverage_metadata_grid_street,
            "count": self._compute_count_metadata_grid_street,
            "days_elapsed": self._compute_days_elapsed_metadata_grid_street,
            "most_recent_date": self._compute_most_recent_date_metadata_grid_street,
            "oldest_date": self._compute_oldest_date_metadata_grid_street,
            "number_of_years": self._compute_number_of_years_metadata_grid_street,
            "number_of_months": self._compute_number_of_months_metadata_grid_street,
            "number_of_days": self._compute_number_of_days_metadata_grid_street,
            "number_of_hours": self._compute_number_of_hours_metadata_grid_street,
            "number_of_days_of_week": self._compute_number_of_days_of_week_metadata_grid_street,
            "number_of_daytime": self._compute_number_of_daytime_metadata_grid_street,
            "number_of_nighttime": self._compute_number_of_nighttime_metadata_grid_street,
            "number_of_spring": self._compute_number_of_spring_metadata_grid_street,
            "number_of_summer": self._compute_number_of_summer_metadata_grid_street,
            "number_of_autumn": self._compute_number_of_autumn_metadata_grid_street,
            "number_of_winter": self._compute_number_of_winter_metadata_grid_street,
            "average_compass_angle": self._compute_average_compass_angle_metadata_grid_street,
            "average_relative_angle": self._compute_average_relative_angle_metadata_grid_street,
            "average_is_pano": self._compute_average_is_pano_metadata_grid_street,
            "number_of_users": self._compute_number_of_users_metadata_grid_street,
            "number_of_sequences": self._compute_number_of_sequences_metadata_grid_street,
            "number_of_organizations": self._compute_number_of_organizations_metadata_grid_street,
            "average_speed_kmh": self._compute_speed_metadata_grid_street,
        }
        
    def _compute_timezones(self, df, lat_col, lon_col):
        # Extract latitude and longitude data as lists for batch processing
        lats = df[lat_col].to_list()
        lons = df[lon_col].to_list()
        
        # Calculate timezones (batch operation)
        timezones = [self._tf_instance.timezone_at(lat=lat, lng=lon) for lat, lon in tqdm(zip(lats, lons), total=len(lats), desc="Computing timezones")]
        
        # Add timezones as a new column
        df = df.with_columns(pl.Series("timezone", timezones))
        return df

    def _compute_datetimes(self, df, timestamp_col, timezone_col):
        # Extract timestamps and timezones
        timestamps = df[timestamp_col].to_list()
        timezones = df[timezone_col].to_list()
        
        # Compute UTC and local datetimes
        datetimes_utc = []
        datetimes_local = []
        for timestamp, timezone_str in tqdm(zip(timestamps, timezones), total=len(timestamps), desc="Computing datetimes"):
            dt_utc = datetime.datetime.fromtimestamp(timestamp / 1000, tz=datetime.timezone.utc)
            dt_local = dt_utc.astimezone(pytz.timezone(timezone_str))
            datetimes_utc.append(dt_utc)
            datetimes_local.append(dt_local.isoformat())
        
        # Add new columns to the DataFrame
        df = df.with_columns(
            pl.Series("datetime_utc", datetimes_utc),
            pl.Series("local_datetime", datetimes_local)
        )
        return df

    def _compute_year_metadata_image(self):
        self.metadata = self.metadata.with_columns(
            # First remove the timezone information, then parse the datetime
           pl.col("local_datetime").dt.year().alias("year")
        )

    def _compute_month_metadata_image(self):
        self.metadata = self.metadata.with_columns(
            pl.col("local_datetime").dt.month().alias("month")
        )

    def _compute_day_metadata_image(self):
        self.metadata = self.metadata.with_columns(
            pl.col("local_datetime").dt.day().alias("day")
        )

    def _compute_hour_metadata_image(self):
        self.metadata = self.metadata.with_columns(
            pl.col("local_datetime").dt.hour().alias("hour")
        )

    def _compute_day_of_week_metadata_image(self):
        self.metadata = self.metadata.with_columns(
            pl.col("local_datetime").dt.weekday().alias("day_of_week")
        )

    def _compute_daytime_nighttime_metadata_image(self):
        self.metadata = _day_or_night(self.metadata)

    def _compute_season_metadata_image(self):
        # place holder for season metadata because it is already computed in the compute_metadata function
        pass

    def _compute_relative_angle_metadata_image(self):
        # Assuming nearest_line and self.metadata are Polars DataFrames
        nearest_line = pl.DataFrame(self.nearest_line[["id", "relative_angle"]])
        # Perform the left join using Polars
        self.metadata = self.metadata.join(nearest_line, on="id", how="left")

    def _compute_speed_metadata_image(self):
        # place holder for speed metadata because it is already computed in the compute_metadata function
        pass

    def _compute_h3_id_metadata_image(self):
        ls_res = [x for x in range(16)]
        for res in ls_res:
            self.metadata = self.metadata.with_columns(
                pl.struct(["lat", "lon"])
                .map_elements(
                    lambda x: h3.geo_to_h3(x["lat"], x["lon"], res),
                    return_dtype=pl.String,
                )
                .alias(f"h3_{res}")
            )

    def _compute_coverage_metadata_grid_street(self):
        # check if self.metadata is geopandas
        if not isinstance(self.metadata_geometry, gpd.GeoDataFrame):
            self.metadata_geometry = gpd.GeoDataFrame(geometry=self.metadata_geometry)
        # Calculate the unified buffer from each point in the GeoDataFrame
        buffer = self.gdf.buffer(self.coverage_buffer).unary_union
        # If possible, use spatial index to improve intersection checks
        if self.metadata_geometry.sindex:
            possible_matches_index = list(
                self.metadata_geometry.sindex.intersection(buffer.bounds)
            )
            relevant_geometries = self.metadata_geometry.iloc[possible_matches_index].copy()
        else:
            relevant_geometries = self.metadata_geometry.copy()

        # Compute intersections with buffer only for potentially intersecting geometries
        relevant_geometries["intersection"] = relevant_geometries[
            "geometry"
        ].intersection(buffer)

        # Calculate coverage based on the type of the geometries
        if isinstance(relevant_geometries["geometry"].iloc[0], LineString):
            relevant_geometries["coverage"] = (
                relevant_geometries["intersection"].length
                / relevant_geometries["geometry"].length
            )
        else:
            relevant_geometries["coverage"] = (
                relevant_geometries["intersection"].area
                / relevant_geometries["geometry"].area
            )

        # Store the results back into the original metadata DataFrame
        self.metadata = self.metadata.with_columns(pl.Series("coverage", relevant_geometries["coverage"]))

    def _compute_count_metadata_grid_street(self):
        # Group by the specified columns and count the rows in each group
        grouped = self.joined.group_by(self.columns_to_group_by).agg(
            pl.len().alias("count")
        )
        # join self.metadata with join on index and index_right
        self.metadata = self.metadata.join(grouped, on=self.columns_to_group_by, how="left")

    def _compute_days_elapsed_metadata_grid_street(self):
        # number of days between the most recent and oldest point (self.gdf) within self.metadata
        # calculate the time elapsed
        # Group by the specified columns and calculate the max and min datetimes
        grouped = self.joined.group_by(self.columns_to_group_by).agg(
            [
                pl.max("local_datetime").alias("max_datetime"),
                pl.min("local_datetime").alias("min_datetime"),
            ]
        )
        # Calculate the time difference in days between the most recent and oldest points
        grouped = grouped.with_columns(
            ((pl.col("max_datetime") - pl.col("min_datetime")).dt.total_days()).alias(
                "days_elapsed"
            )
        )
        # Join self.metadata with grouped on the specified columns, only including the 'days_elapsed' column
        self.metadata = self.metadata.join(grouped.select([self.columns_to_group_by, "days_elapsed"]), on=self.columns_to_group_by, how="left")

    def _compute_most_recent_date_metadata_grid_street(self):
        # number of days between the most recent point (self.gdf) and the most recent point within self.metadata
        # calculate the age of the most recent point
        # Group by the specified columns and calculate the max datetime
        grouped = self.joined.group_by(self.columns_to_group_by).agg(
            pl.max("local_datetime").alias("most_recent_date")
        )
        # Convert the most recent datetime to date (string format)
        grouped = grouped.with_columns(
            pl.col("most_recent_date").dt.strftime("%Y-%m-%d").alias("most_recent_date")
        )
        # Join self.metadata with grouped on the specified columns, only including the 'most_recent_date' column
        self.metadata = self.metadata.join(grouped.select([self.columns_to_group_by, "most_recent_date"]), on=self.columns_to_group_by, how="left")

    def _compute_oldest_date_metadata_grid_street(self):
        # number of days between the oldest point (self.gdf) and the oldest point within self.metadata
        # calculate the age of the oldest point
        # Group by the specified columns and calculate the min datetime
        grouped = self.joined.group_by(self.columns_to_group_by).agg(
            pl.min("local_datetime").alias("oldest_date")
        )
        # Convert the oldest datetime to date (string format)
        grouped = grouped.with_columns(
            pl.col("oldest_date").dt.strftime("%Y-%m-%d").alias("oldest_date")
        )
        # Join self.metadata with grouped on the specified columns, only including the 'oldest_date' column
        self.metadata = self.metadata.join(grouped.select([self.columns_to_group_by, "oldest_date"]), on=self.columns_to_group_by, how="left")

    def _compute_number_of_years_metadata_grid_street(self):
        # number of unique years in the dataset
        # spaital join self.gdf to self.metadata
        # Extract the year from 'local_datetime' and create a new column
        grouped = self.joined.with_columns(
            pl.col("local_datetime").dt.year().alias("year")
        )
        # Group by the specified columns and calculate the number of unique years
        grouped = grouped.group_by(self.columns_to_group_by).agg(
            pl.col("year").n_unique().alias("number_of_years")
        )
        # Join self.metadata with grouped on the specified columns, only including the 'number_of_years' column
        self.metadata = self.metadata.join(grouped.select([self.columns_to_group_by, "number_of_years"]), on=self.columns_to_group_by, how="left")

    def _compute_number_of_months_metadata_grid_street(self):
        # number of unique months in the dataset
        # spaital join self.gdf to self.metadata
        # Extract the month from 'local_datetime' and create a new column
        grouped = self.joined.with_columns(
            pl.col("local_datetime").dt.month().alias("month")
        )
        # Group by the specified columns and calculate the number of unique months
        grouped = grouped.group_by(self.columns_to_group_by).agg(
            pl.col("month").n_unique().alias("number_of_months")
        )
        # Join self.metadata with grouped on the specified columns, only including the 'number_of_months' column
        self.metadata = self.metadata.join(grouped.select([self.columns_to_group_by, "number_of_months"]), on=self.columns_to_group_by, how="left")

    def _compute_number_of_days_metadata_grid_street(self):
        # number of unique days in the dataset
        # spaital join self.gdf to self.metadata
        # Extract the day from 'local_datetime' and create a new column
        grouped = self.joined.with_columns(
            pl.col("local_datetime").dt.day().alias("day")
        )
        # Group by the specified columns and calculate the number of unique days
        grouped = grouped.group_by(self.columns_to_group_by).agg(
            pl.col("day").n_unique().alias("number_of_days")
        )
        # Join self.metadata with grouped on the specified columns, only including the 'number_of_days' column
        self.metadata = self.metadata.join(grouped.select([self.columns_to_group_by, "number_of_days"]), on=self.columns_to_group_by, how="left")

    def _compute_number_of_hours_metadata_grid_street(self):
        # number of unique hours in the dataset
        # spaital join self.gdf to self.metadata
        # Extract the hour from 'local_datetime' and create a new column
        grouped = self.joined.with_columns(
            pl.col("local_datetime").dt.hour().alias("hour")
        )
        # Group by the specified columns and calculate the number of unique hours
        grouped = grouped.group_by(self.columns_to_group_by).agg(
            pl.col("hour").n_unique().alias("number_of_hours")
        )
        # Join self.metadata with grouped on the specified columns, only including the 'number_of_hours' column
        self.metadata = self.metadata.join(grouped.select([self.columns_to_group_by, "number_of_hours"]), on=self.columns_to_group_by, how="left")

    def _compute_number_of_days_of_week_metadata_grid_street(self):
        # number of unique days of week in the dataset
        # spaital join self.gdf to self.metadata
        # Extract the day of week from 'local_datetime' and create a new column
        grouped = self.joined.with_columns(
            pl.col("local_datetime").dt.weekday().alias("day_of_week")
        )
        # Group by the specified columns and calculate the number of unique days of week
        grouped = grouped.group_by(
            self.columns_to_group_by
        ).agg(pl.col("day_of_week").n_unique().alias("number_of_days_of_week"))
        # Join self.metadata with grouped on the specified columns, only including the 'number_of_days_of_week' column
        self.metadata = self.metadata.join(grouped.select([self.columns_to_group_by, "number_of_days_of_week"]), on=self.columns_to_group_by, how="left")

    def _compute_number_of_daytime_metadata_grid_street(self):
        # number of points captured during daytime
        # calculate the number of points captured during daytime using _day_or_night function
        grouped = _day_or_night(self.joined)
        grouped = grouped.with_columns(
            pl.col("daytime_nighttime").eq(pl.lit("daytime")).alias("daytime")
        )
        # Group by specified columns and sum the daytime values
        grouped = grouped.group_by(self.columns_to_group_by).agg(
            pl.col("daytime").sum().alias("number_of_daytime")
        )
        # Join self.metadata with grouped on the specified columns, only including the 'number_of_daytime' column
        self.metadata = self.metadata.join(grouped.select([self.columns_to_group_by, "number_of_daytime"]), on=self.columns_to_group_by, how="left")

    def _compute_number_of_nighttime_metadata_grid_street(self):
        # number of points captured during nighttime
        grouped = _day_or_night(self.joined)
        grouped = grouped.with_columns(
            pl.col("daytime_nighttime").eq(pl.lit("nighttime")).alias("nighttime")
        )
        # Group by specified columns and sum the nighttime values
        grouped = grouped.group_by(self.columns_to_group_by).agg(
            pl.col("nighttime").sum().alias("number_of_nighttime")
        )
        # Join self.metadata with grouped on the specified columns, only including the 'number_of_nighttime' column
        self.metadata = self.metadata.join(grouped.select([self.columns_to_group_by, "number_of_nighttime"]), on=self.columns_to_group_by, how="left")

    def _compute_number_of_spring_metadata_grid_street(self):
        # number of points captured during spring
        # calculate the number of points captured during spring using _datetime_to_season function
        grouped = self.joined
        grouped = grouped.with_columns(
            pl.struct(["local_datetime", "lat"]).
            map_elements(lambda x: _datetime_to_season(x["local_datetime"], x["lat"]), return_dtype=pl.String).
            eq(pl.lit("Spring")).
            alias("spring")
        ).group_by(self.columns_to_group_by).agg(pl.col("spring").sum().alias("number_of_spring"))
        # Join self.metadata with grouped on the specified columns, only including the 'number_of_spring' column
        self.metadata = self.metadata.join(grouped.select([self.columns_to_group_by, "number_of_spring"]), on=self.columns_to_group_by, how="left")

    def _compute_number_of_summer_metadata_grid_street(self):
        # number of points captured during summer
        # calculate the number of points captured during summer using _datetime_to_season function
        grouped = self.joined
        grouped = grouped.with_columns(
            pl.struct(["local_datetime", "lat"]).
            map_elements(lambda x: _datetime_to_season(x["local_datetime"], x["lat"]), return_dtype=pl.String).
            eq(pl.lit("Summer")).
            alias("summer")
        ).group_by(self.columns_to_group_by).agg(pl.col("summer").sum().alias("number_of_summer"))
        # Join self.metadata with grouped on the specified columns, only including the 'number_of_summer' column
        self.metadata = self.metadata.join(grouped.select([self.columns_to_group_by, "number_of_summer"]), on=self.columns_to_group_by, how="left")

    def _compute_number_of_autumn_metadata_grid_street(self):
        # number of points captured during autumn
        # calculate the number of points captured during autumn using _datetime_to_season function
        grouped = self.joined
        grouped = grouped.with_columns(
            pl.struct(["local_datetime", "lat"]).
            map_elements(lambda x: _datetime_to_season(x["local_datetime"], x["lat"]), return_dtype=pl.String).
            eq(pl.lit("Autumn")).
            alias("autumn")
        ).group_by(self.columns_to_group_by).agg(pl.col("autumn").sum().alias("number_of_autumn"))
        # Join self.metadata with grouped on the specified columns, only including the 'number_of_autumn' column
        self.metadata = self.metadata.join(grouped.select([self.columns_to_group_by, "number_of_autumn"]), on=self.columns_to_group_by, how="left")

    def _compute_number_of_winter_metadata_grid_street(self):
        # number of points captured during winter
        # calculate the number of points captured during winter using _datetime_to_season function
        grouped = self.joined
        grouped = grouped.with_columns(
            pl.struct(["local_datetime", "lat"]).
            map_elements(lambda x: _datetime_to_season(x["local_datetime"], x["lat"]), return_dtype=pl.String).
            eq(pl.lit("Winter")).
            alias("winter")
        ).group_by(self.columns_to_group_by).agg(pl.col("winter").sum().alias("number_of_winter"))
        # Join self.metadata with grouped on the specified columns, only including the 'number_of_winter' column
        self.metadata = self.metadata.join(grouped.select([self.columns_to_group_by, "number_of_winter"]), on=self.columns_to_group_by, how="left")

    def _compute_average_compass_angle_metadata_grid_street(self):
        # average compass angle of the points within self.metadata
        # calculate the average compass angle
        grouped = self.joined.group_by(self.columns_to_group_by).agg(
            pl.col("compass_angle").mean().alias("average_compass_angle")
        )
        # Join self.metadata with grouped on the specified columns, only including the 'average_compass_angle' column
        self.metadata = self.metadata.join(grouped.select([self.columns_to_group_by, "average_compass_angle"]), on=self.columns_to_group_by, how="left")

    def _compute_average_relative_angle_metadata_grid_street(self):
        # average relative angle of the points within self.metadata
        nearest_line = pl.DataFrame(self.nearest_line[["id", "relative_angle"]])
        grouped = self.joined.join(nearest_line, on="id", how="left")
        # calculate the average relative angle
        grouped = grouped.group_by(
            self.columns_to_group_by
        ).agg(pl.col("relative_angle").mean().alias("average_relative_angle"))
        # Join self.metadata with grouped on the specified columns, only including the 'average_relative_angle' column
        self.metadata = self.metadata.join(grouped.select([self.columns_to_group_by, "average_relative_angle"]), on=self.columns_to_group_by, how="left")

    def _compute_average_is_pano_metadata_grid_street(self):
        # average is_pano of the points within self.metadata
        # calculate the average is_pano
        grouped = self.joined.group_by(
            self.columns_to_group_by
        ).agg(pl.col("is_pano").mean().alias("average_is_pano"))
        # Join self.metadata with grouped on the specified columns, only including the 'average_is_pano' column
        self.metadata = self.metadata.join(grouped.select([self.columns_to_group_by, "average_is_pano"]), on=self.columns_to_group_by, how="left")

    def _compute_number_of_users_metadata_grid_street(self):
        # number of unique users in the dataset
        # calculate the number of unique users
        grouped = self.joined.group_by(
            self.columns_to_group_by
        ).agg(pl.col("creator_id").n_unique().alias("number_of_users"))
        # Join self.metadata with grouped on the specified columns, only including the 'number_of_users' column
        self.metadata = self.metadata.join(grouped.select([self.columns_to_group_by, "number_of_users"]), on=self.columns_to_group_by, how="left")

    def _compute_number_of_sequences_metadata_grid_street(self):
        # number of unique sequences in the dataset
        # calculate the number of unique sequences
        grouped = self.joined.group_by(
            self.columns_to_group_by
        ).agg(pl.col("sequence_id").n_unique().alias("number_of_sequences"))
        # Join self.metadata with grouped on the specified columns, only including the 'number_of_sequences' column
        self.metadata = self.metadata.join(grouped.select([self.columns_to_group_by, "number_of_sequences"]), on=self.columns_to_group_by, how="left")

    def _compute_number_of_organizations_metadata_grid_street(self):
        # number of unique organizations in the dataset
        # calculate the number of unique organizations
        grouped = self.joined.group_by(
            self.columns_to_group_by
        ).agg(pl.col("organization_id").n_unique().alias("number_of_organizations"))
        # Join self.metadata with grouped on the specified columns, only including the 'number_of_organizations' column
        self.metadata = self.metadata.join(grouped.select([self.columns_to_group_by, "number_of_organizations"]), on=self.columns_to_group_by, how="left")

    def _compute_speed_metadata_grid_street(self):
        # average speed of the points within self.metadata
        # calculate the average speed
        grouped = self.joined.group_by(
            self.columns_to_group_by
        ).agg(pl.col("speed_kmh").mean().alias("average_speed_kmh"))
        # Join self.metadata with grouped on the specified columns, only including the 'average_speed_kmh' column
        self.metadata = self.metadata.join(grouped.select([self.columns_to_group_by, "average_speed_kmh"]), on=self.columns_to_group_by, how="left")

    def _compute_image_metadata(self, indicator_list):
        # define self.metadata as a copy of the input DataFrame with only "id" column
        self.metadata = self.df
        if indicator_list == "all":
            indicator_list = self.indicator_metadata_image.keys()
        else:
            # split string of indicators into a list
            indicator_list = indicator_list.split(" ")
        for indicator in indicator_list:
            self.indicator_metadata_image[indicator]()
        # make sure that columns do not include those that are in self.indicator_metadata_image (key) but not in the indicator_list in this run
        for key in self.indicator_metadata_image.keys():
            if key not in indicator_list and key in self.metadata.columns:
                self.metadata = self.metadata.drop(key)
        return self.metadata

    def _compute_grid_metadata(self, indicator_list):
        if indicator_list == "all":
            indicator_list = self.indicator_metadata_grid_street.keys()
        else:
            # split string of indicators into a list
            indicator_list = indicator_list.split(" ")
        for indicator in indicator_list:
            self.indicator_metadata_grid_street[indicator]()
        # make sure that columns do not include those that are in self.indicator_metadata_grid_street (key) but not in the indicator_list in this run
        for key in self.indicator_metadata_grid_street.keys():
            if key not in indicator_list and key in self.metadata.columns:
                self.metadata = self.metadata.drop(key)
        return self.metadata

    def _compute_street_metadata(self, indicator_list):
        if indicator_list == "all":
            indicator_list = self.indicator_metadata_grid_street.keys()
        else:
            # split string of indicators into a list
            indicator_list = indicator_list.split(" ")
        for indicator in indicator_list:
            self.indicator_metadata_grid_street[indicator]()
        return self.metadata

    def compute_metadata(
        self,
        unit: str = "image",
        grid_resolution: int = 7,
        coverage_buffer: int = 50,
        indicator_list: str = "all",
        path_output: Union[str, Path] = None,
        max_distance: int = 50,
    ):
        """
        Compute metadata for the dataset.

        :param unit: The unit of analysis. Defaults to "image".
        :type unit: str
        :param grid_resolution: The resolution of the H3 grid. Defaults to 7.
        :type grid_resolution: int
        :param indicator_list: List of indicators to compute metadata for. Use space-separated string of indicators or "all". Options for image-level metadata: "year", "month", "day", "hour", "day_of_week", "relative_angle", "h3_id", "speed_kmh". Options for grid-level metadata: "coverage", "count", "days_elapsed", "most_recent_date", "oldest_date", "number_of_years", "number_of_months", "number_of_days", "number_of_hours", "number_of_days_of_week", "number_of_daytime", "number_of_nighttime", "number_of_spring", "number_of_summer", "number_of_autumn", "number_of_winter", "average_compass_angle", "average_relative_angle", "average_is_pano", "number_of_users", "number_of_sequences", "number_of_organizations", "average_speed_kmh". Defaults to "all".
        :type indicator_list: str
        :param path_output: Path to save the output metadata. Defaults to None.
        :type path_output: Union[str, Path]
        :param max_distance: The maximum distance to search for the nearest street segment. Defaults to 50.
        :type max_distance: int

        :return: A DataFrame containing the computed metadata.
        :rtype: pd.DataFrame
        """
        # set coverage buffer as a class attribute
        self.coverage_buffer = coverage_buffer

        # check indicator_list and pre-compute metadata, e.g., timezone and local datetime
        if (
            "all" in indicator_list
            or "year" in indicator_list
            or "month" in indicator_list
            or "day" in indicator_list
            or "hour" in indicator_list
            or "time" in indicator_list
            or "speed" in indicator_list
        ):
            # calculate local datetime for each point
            self.df = self._compute_timezones(self.df, "lat", "lon")
            self.df = self._compute_datetimes(self.df, "captured_at", "timezone")

        # check indicator_list and pre-compute metadata, e.g., season
        if (
            "all" in indicator_list
            or "season" in indicator_list
            or "spring" in indicator_list
            or "summer" in indicator_list
            or "autumn" in indicator_list
            or "winter" in indicator_list
        ):
            self.df = self.df.with_columns(
                pl.struct(["local_datetime", "lat"])
                .map_elements(
                    lambda x: _datetime_to_season(x["local_datetime"], x["lat"]),
                    return_dtype=pl.String,
                )
                .alias("season")
            )

        # check indicator_list and pre-compute metadata, e.g., speed
        if "all" in indicator_list or "speed" in indicator_list:
            self.df = _compute_speed(self.df)

        # create self.gdf as a GeoDataFrame for spatial operations
        self.gdf = gpd.GeoDataFrame(
            self.df.to_pandas(),
            geometry=gpd.points_from_xy(self.df["lon"], self.df["lat"]),
            crs=4326,
        )
        self.gdf = self.gdf.to_crs(crs=self.projected_crs)

        # check indicator_list and pre-compute metadata, e.g., relative_angle
        if "all" in indicator_list or "relative_angle" in indicator_list:
            # Perform the nearest join with the street network
            nearest_line = self.gdf.sjoin_nearest(
                self.street_network[
                    ["geometry", "angle"]
                ],  # Ensure only necessary columns are loaded
                how="left",
                max_distance=max_distance,
                distance_col="dist",  # Save distances to avoid recomputing
            )

            # Calculate the relative angle and ensure it is within 0-180 degrees because we don't know the direction of the street
            nearest_line["relative_angle"] = (
                (nearest_line["angle"] - nearest_line["compass_angle"]) % 180
            ).abs()

            # Reduce to the essential data and perform a group_by operation to find the minimum angle for each 'id'
            min_angle = (
                nearest_line[["id", "relative_angle"]]
                .groupby("id", as_index=False)
                .min()
            )

            # Merge back to the original GeoDataFrame to include all original data
            self.nearest_line = self.gdf.merge(min_angle, on="id", how="left")

        # run the appropriate function to compute metadata based on the unit of analysis
        if unit == "image":
            # after here, there's no gpd.GeoDataFrame conversion, so let's convert local_datetime to datetime for Polars
            self.df = self.df.with_columns(
                pl.col("local_datetime").str.replace(r"([+-]\d{2}:\d{2})$", "").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.f").alias("local_datetime")
            )
            df = self._compute_image_metadata(indicator_list)
        elif unit == "grid":
            self.metadata_geometry = _create_hexagon(pl.from_dataframe(self.gdf[["lat", "lon"]]), resolution=grid_resolution)
            # reproject self.metadata_geometry to the same crs as self.street_network
            self.metadata_geometry = self.metadata_geometry.to_crs(self.projected_crs)
            # create key column by counting
            self.metadata_geometry["key_col"] = range(len(self.metadata_geometry))
            # convert key_col to string
            self.metadata_geometry["key_col"] = self.metadata_geometry["key_col"].astype(str)
            # spatial join self.gdf to self.metadata_geometry
            self.joined = self.gdf.sjoin(self.metadata_geometry)
            self.joined = pl.DataFrame(self.joined.drop(columns="geometry"))
            self.metadata = pl.DataFrame(self.metadata_geometry.drop(columns="geometry"))
            # after here, there's no gpd.GeoDataFrame conversion, so let's convert local_datetime to datetime for Polars
            self.joined = self.joined.with_columns(
                pl.col("local_datetime").str.replace(r"([+-]\d{2}:\d{2})$", "").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.f").alias("local_datetime")
            )
            # Get a list of all columns that contain 'index_right'
            self.columns_to_group_by = "key_col"
            df = self._compute_grid_metadata(indicator_list)
            # drop key_col column
            df = df.drop("key_col")
        elif unit == "street":
            self.metadata_geometry = self.street_network.copy()
            # create key column by counting
            self.metadata_geometry["key_col"] = range(len(self.metadata_geometry))
            # convert key_col to string
            self.metadata_geometry["key_col"] = self.metadata_geometry["key_col"].astype(str)
            # spatial join self.gdf to self.metadata_geometry
            self.joined = self.gdf.sjoin_nearest(self.metadata_geometry)
            self.joined = pl.DataFrame(self.joined.drop(columns="geometry"))
            self.metadata = pl.DataFrame(self.metadata_geometry.drop(columns="geometry"))
            # after here, there's no gpd.GeoDataFrame conversion, so let's convert local_datetime to datetime for Polars
            self.joined = self.joined.with_columns(
                pl.col("local_datetime").str.replace(r"([+-]\d{2}:\d{2})$", "").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.f").alias("local_datetime")
            )
            self.columns_to_group_by = "key_col"
            df = self._compute_street_metadata(indicator_list)
            # drop key_col column
            df = df.drop("key_col")
        else:
            raise ValueError("Invalid unit of analysis provided.")

        # save the output metadata to a file
        if path_output:
            if unit == "image":
                df.write_csv(path_output)
            else:
                if Path(path_output).suffix == ".geojson":
                    gdf = gpd.GeoDataFrame(df.to_pandas(), geometry=self.metadata_geometry.geometry)
                    gdf = gdf.set_crs(self.projected_crs)
                    gdf = gdf.to_crs("EPSG:4326")
                    gdf.to_file(path_output, driver="GeoJSON")
                elif Path(path_output).suffix == ".shp":
                    gdf = gpd.GeoDataFrame(df.to_pandas(), geometry=self.metadata_geometry.geometry)
                    gdf = gdf.set_crs(self.projected_crs)
                    gdf = gdf.to_crs("EPSG:4326")
                    gdf.to_file(path_output, driver="ESRI Shapefile")
                elif Path(path_output).suffix == ".csv":
                    df.write_csv(path_output)
                else:
                    raise ValueError(
                        "Invalid file format provided. Please provide a .csv, .shp, or .geojson file."
                    )
        df = df.to_pandas()
        return df
