import datetime
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import h3
import numpy as np
import osmnx as ox
import pandas as pd
import polars as pl
from astral import LocationInfo, sun
from shapely.geometry import LineString, MultiPoint, Polygon
from timezonefinder import TimezoneFinder
from tqdm.auto import tqdm

from zensvi.utils.log import Logger


def _calculate_angle(line: list) -> Optional[float]:
    if len(line) > 1:
        start, end = line[0], line[-1]
        angle = np.arctan2(end[1] - start[1], end[0] - start[0])
        return np.degrees(angle) % 360
    else:
        return None


def _latlng_to_h3(lat: float, lon: float, resolution: int) -> str:
    """Convert lat/lon to H3 cell, compatible with h3 v3 and v4."""
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(lat, lon, resolution)
    return h3.geo_to_h3(lat, lon, resolution)


def _h3_to_geo_boundary(hex_id: str) -> list:
    """Get boundary coordinates for an H3 cell in GeoJSON order (lng, lat)."""
    if hasattr(h3, "cell_to_boundary"):
        boundary = h3.cell_to_boundary(hex_id)
        return [(lng, lat) for lat, lng in boundary]
    return h3.h3_to_geo_boundary(hex_id, geo_json=True)


def _create_hexagon(df: pl.DataFrame, resolution: int = 7) -> gpd.GeoDataFrame:
    df = df.with_columns(
        pl.struct(["lat", "lon"])
        .map_elements(lambda x: _latlng_to_h3(x["lat"], x["lon"], resolution), return_dtype=pl.String)
        .alias("h3_id")
    )
    unique_h3_ids = df.select("h3_id").unique()
    hex_gdf = gpd.GeoDataFrame(
        unique_h3_ids.to_pandas(),
        geometry=[Polygon(_h3_to_geo_boundary(h)) for h in unique_h3_ids["h3_id"]],
        crs=4326,
    )
    return hex_gdf


def _day_or_night(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(pl.col("local_datetime").dt.date().alias("date"))
    unique_locations_dates = df.select(["lat", "lon", "date", "timezone"]).unique()
    sun_times_list = []
    for row in unique_locations_dates.iter_rows():
        location = LocationInfo(latitude=row[0], longitude=row[1])
        s = sun.sun(location.observer, date=row[2], tzinfo=row[3])
        sun_times_list.append(
            {
                "lat": row[0],
                "lon": row[1],
                "date": row[2],
                "sunrise": s["sunrise"].astimezone(datetime.timezone.utc),
                "sunset": s["sunset"].astimezone(datetime.timezone.utc),
            }
        )
    sun_times_df = pl.DataFrame(sun_times_list)
    df = df.join(sun_times_df, on=["lat", "lon", "date"])
    df = df.with_columns(
        pl.when(
            (pl.col("sunrise").cast(pl.Datetime("us")) <= pl.col("datetime_utc").cast(pl.Datetime("us")))
            & (pl.col("datetime_utc").cast(pl.Datetime("us")) <= pl.col("sunset").cast(pl.Datetime("us")))
        )
        .then(pl.lit("daytime"))
        .otherwise(pl.lit("nighttime"))
        .alias("daytime_nighttime")
    )
    return df.drop(["date", "sunrise", "sunset"])


def _compute_speed(df: pl.DataFrame) -> pl.DataFrame:
    # Convert DataFrame to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df.to_pandas(),
        geometry=gpd.points_from_xy(df["lon"], df["lat"], crs="EPSG:4326"),
    )

    # Project to a local CRS once
    gdf = ox.projection.project_gdf(gdf)

    # Sort by sequence_id and datetime_utc for consecutive-point distance calc
    gdf = gdf.sort_values(["sequence_id", "datetime_utc"]).reset_index(drop=True)

    # Vectorized distance: shift geometry within each sequence, compute distance
    shifted_geom = gdf.groupby("sequence_id")["geometry"].shift(-1)
    shifted_dt = gdf.groupby("sequence_id")["datetime_utc"].shift(-1)

    # Vectorized distance between consecutive points (GeoSeries.distance)
    distance_m = gdf["geometry"].distance(shifted_geom)

    # Time difference in hours
    time_diff_hrs = (shifted_dt - gdf["datetime_utc"]).dt.total_seconds() / 3600

    # Speed in km/h
    gdf["speed_kmh"] = (distance_m / 1000) / time_diff_hrs

    # Convert back to Polars, dropping geometry column
    result = pl.DataFrame(gdf.drop(columns=["geometry"]))

    # Convert Inf and NaN to None
    result = result.with_columns(
        pl.when(pl.col("speed_kmh").is_null() | pl.col("speed_kmh").is_nan() | pl.col("speed_kmh").is_infinite())
        .then(None)
        .otherwise(pl.col("speed_kmh"))
        .alias("speed_kmh")
    )

    return result


def _datetime_to_season(local_datetime: Union[datetime.datetime, str], lat: float) -> str:
    season_month_north = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Autumn",
        10: "Autumn",
        11: "Autumn",
    }
    season_month_south = {
        12: "Summer",
        1: "Summer",
        2: "Summer",
        3: "Autumn",
        4: "Autumn",
        5: "Autumn",
        6: "Winter",
        7: "Winter",
        8: "Winter",
        9: "Spring",
        10: "Spring",
        11: "Spring",
    }

    if isinstance(local_datetime, datetime.datetime):
        month = local_datetime.month
    elif isinstance(local_datetime, str):
        month = datetime.datetime.fromisoformat(local_datetime).month
    else:
        raise ValueError("local_datetime must be either a datetime object or a string")

    return season_month_north[month] if lat >= 0 else season_month_south[month]


def _season_expr(datetime_col: str = "local_datetime", lat_col: str = "lat") -> pl.Expr:
    """Vectorized Polars expression that maps (month, hemisphere) to season string."""
    month = pl.col(datetime_col).dt.month()
    is_north = pl.col(lat_col) >= 0
    # Northern hemisphere mapping
    north = (
        pl.when(month.is_in([12, 1, 2]))
        .then(pl.lit("Winter"))
        .when(month.is_in([3, 4, 5]))
        .then(pl.lit("Spring"))
        .when(month.is_in([6, 7, 8]))
        .then(pl.lit("Summer"))
        .otherwise(pl.lit("Autumn"))
    )
    # Southern hemisphere mapping (shifted 6 months)
    south = (
        pl.when(month.is_in([12, 1, 2]))
        .then(pl.lit("Summer"))
        .when(month.is_in([3, 4, 5]))
        .then(pl.lit("Autumn"))
        .when(month.is_in([6, 7, 8]))
        .then(pl.lit("Winter"))
        .otherwise(pl.lit("Spring"))
    )
    return pl.when(is_north).then(north).otherwise(south)


class BaseMetadata:
    """Base class to compute metadata for a street-view dataset.

    Subclasses provide the input by overriding :meth:`_load_dataframe` and
    :meth:`_normalize_columns` so that ``self.df`` exposes the normalized schema the
    compute methods expect: ``"id", "lat", "lon", "captured_at"`` (milliseconds since the
    Unix epoch), ``"compass_angle", "creator_id", "sequence_id", "organization_id",
    "is_pano"``.

    Args:
        path_input (Union[str, Path]): path to the input CSV file.
    """

    def __init__(self, path_input: Union[str, Path], log_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize the metadata class.

        Args:
            path_input (Union[str, Path]): Path to the input CSV file.
            log_path (Union[str, Path], optional): Path to the log file. Defaults to None.

        Raises:
            ValueError: If the input CSV file does not contain required columns.

        This method initializes the class by:
        1. Setting up logging if a log path is provided.
        2. Loading and normalizing the input CSV file.
        3. Setting up dictionaries of metadata computation functions for image and grid/street levels.
        The street network is loaded lazily the first time it is needed.
        """
        self._tf_instance = TimezoneFinder()
        self.path_input = Path(path_input)
        if log_path is not None:
            self.logger = Logger(log_path)
        else:
            self.logger = None
        self.df = self._normalize_columns(self._load_dataframe())
        self.metadata = None
        # Street network is loaded lazily via _ensure_street_network()
        self.street_network = None
        self.projected_crs = None
        # Per-call caches cleared at the start of compute_metadata
        self._joined_daynight_cache = None
        self._seasons_grouped_cache = None

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

    def _load_dataframe(self) -> pl.DataFrame:
        """Load the raw input into a polars DataFrame. Override for non-CSV sources."""
        return pl.read_csv(self.path_input)

    def _normalize_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Rename/derive columns to the normalized schema. Default is a no-op.

        Subclasses map their provider-specific columns to: ``id, lat, lon, captured_at``
        (ms since Unix epoch), ``compass_angle, creator_id, sequence_id, organization_id,
        is_pano``.
        """
        return df

    def _ensure_street_network(self) -> None:
        """Lazily load and prepare the street network. No-op if already loaded."""
        if self.street_network is not None:
            return
        lats = self.df["lat"].to_list()
        lons = self.df["lon"].to_list()
        # Build a tight query polygon: convex hull of all points + ~200 m buffer.
        # This avoids downloading the huge empty area that a raw bounding box would cover
        # when input points are sparse or spread across non-contiguous areas.
        buffer_deg = 200 / 111_000  # ~200 m expressed in degrees latitude
        points_geom = MultiPoint(list(zip(lons, lats)))
        query_polygon = points_geom.convex_hull.buffer(buffer_deg)
        print(
            f"Downloading OSM street network for {len(lats)} points "
            f"(convex hull + 200 m buffer). This may take a while for large areas…"
        )
        graph = ox.graph_from_polygon(query_polygon, network_type="all")
        sn = ox.convert.graph_to_gdfs(graph, nodes=False)
        sn = ox.projection.project_gdf(sn)
        self.projected_crs = sn.crs
        geometry_list = [list(line.coords) for line in sn["geometry"]]
        sn["geometry_list"] = geometry_list
        sn_data = sn.drop(columns=[col for col in sn.columns if col not in ["geometry_list", "geometry"]])
        sn_pl = pl.DataFrame({"geometry_list": pl.Series(sn_data["geometry_list"], dtype=pl.List(pl.List(pl.Float64)))})
        sn_pl = sn_pl.with_columns(
            pl.col("geometry_list").map_elements(lambda x: _calculate_angle(x), return_dtype=pl.Float64).alias("angle")
        ).drop("geometry_list")
        self.street_network = gpd.GeoDataFrame(sn_pl.to_pandas(), geometry=sn_data.reset_index().geometry)

    def _compute_timezones(self, df: pl.DataFrame, lat_col: str, lon_col: str) -> pl.DataFrame:
        # Deduplicate by rounding to 0.01° (~1km) — points this close share a timezone
        unique_coords = df.select(
            pl.col(lat_col).round(2).alias("_lat_r"),
            pl.col(lon_col).round(2).alias("_lon_r"),
        ).unique()

        # Lookup timezone only for unique rounded coordinates
        tz_list = [
            self._tf_instance.timezone_at(lat=row[0], lng=row[1])
            for row in tqdm(unique_coords.iter_rows(), total=len(unique_coords), desc="Computing timezones")
        ]
        unique_coords = unique_coords.with_columns(pl.Series("timezone", tz_list))

        # Map back to original rows via rounded coordinates
        df = df.with_columns(
            pl.col(lat_col).round(2).alias("_lat_r"),
            pl.col(lon_col).round(2).alias("_lon_r"),
        )
        df = df.join(unique_coords, on=["_lat_r", "_lon_r"], how="left", coalesce=True).drop(["_lat_r", "_lon_r"])
        return df

    def _compute_datetimes(self, df: pl.DataFrame, timestamp_col: str, timezone_col: str) -> pl.DataFrame:
        # Convert millisecond timestamps to UTC datetimes using Polars native ops
        df = df.with_columns(
            pl.from_epoch(pl.col(timestamp_col), time_unit="ms").dt.replace_time_zone("UTC").alias("datetime_utc"),
            pl.arange(0, pl.len()).alias("_row_idx"),
        )

        # Convert to local datetimes by processing each timezone group
        local_parts = []
        for (tz_str,), group in df.group_by([timezone_col]):
            local_parts.append(
                group.with_columns(pl.col("datetime_utc").dt.convert_time_zone(tz_str).alias("local_datetime"))
            )
        df = pl.concat(local_parts).sort("_row_idx").drop("_row_idx")

        return df

    def _compute_year_metadata_image(self) -> None:
        self.metadata = self.metadata.with_columns(
            # First remove the timezone information, then parse the datetime
            pl.col("local_datetime")
            .dt.year()
            .alias("year")
        )

    def _compute_month_metadata_image(self) -> None:
        self.metadata = self.metadata.with_columns(pl.col("local_datetime").dt.month().alias("month"))

    def _compute_day_metadata_image(self) -> None:
        self.metadata = self.metadata.with_columns(pl.col("local_datetime").dt.day().alias("day"))

    def _compute_hour_metadata_image(self) -> None:
        self.metadata = self.metadata.with_columns(pl.col("local_datetime").dt.hour().alias("hour"))

    def _compute_day_of_week_metadata_image(self) -> None:
        self.metadata = self.metadata.with_columns(pl.col("local_datetime").dt.weekday().alias("day_of_week"))

    def _compute_daytime_nighttime_metadata_image(self) -> None:
        self.metadata = _day_or_night(self.metadata)

    def _compute_season_metadata_image(self) -> None:
        # place holder for season metadata because it is already computed in the compute_metadata function
        pass

    def _compute_relative_angle_metadata_image(self) -> None:
        # Assuming nearest_line and self.metadata are Polars DataFrames
        nearest_line = pl.DataFrame(self.nearest_line[["id", "relative_angle"]])
        # Perform the left join using Polars
        self.metadata = self.metadata.join(nearest_line, on="id", how="left")

    def _compute_speed_metadata_image(self) -> None:
        # place holder for speed metadata because it is already computed in the compute_metadata function
        pass

    def _compute_h3_id_metadata_image(self) -> None:
        # Compute H3 IDs on unique (lat, lon) pairs to avoid N×16 Python UDF calls.
        unique_ll = self.metadata.select(["lat", "lon"]).unique()
        for res in range(16):
            r = res  # avoid late-binding in lambda
            unique_ll = unique_ll.with_columns(
                pl.struct(["lat", "lon"])
                .map_elements(
                    lambda x, r=r: _latlng_to_h3(x["lat"], x["lon"], r),
                    return_dtype=pl.String,
                )
                .alias(f"h3_{res}")
            )
        self.metadata = self.metadata.join(unique_ll, on=["lat", "lon"], how="left")

    def _compute_coverage_metadata_grid_street(self) -> None:
        # check if self.metadata is geopandas
        if not isinstance(self.metadata_geometry, gpd.GeoDataFrame):
            self.metadata_geometry = gpd.GeoDataFrame(geometry=self.metadata_geometry)
        # Calculate the unified buffer from each point in the GeoDataFrame
        buffer = self.gdf.buffer(self.coverage_buffer).unary_union
        # If possible, use spatial index to improve intersection checks
        if self.metadata_geometry.sindex:
            possible_matches_index = list(self.metadata_geometry.sindex.intersection(buffer.bounds))
            relevant_geometries = self.metadata_geometry.iloc[possible_matches_index].copy()
        else:
            relevant_geometries = self.metadata_geometry.copy()

        # Compute intersections with buffer only for potentially intersecting geometries
        if len(relevant_geometries) == 0:
            self.metadata = self.metadata.with_columns(pl.lit(0.0).alias("coverage"))
            return

        relevant_geometries["intersection"] = relevant_geometries["geometry"].intersection(buffer)

        # Calculate coverage based on the type of the geometries
        if isinstance(relevant_geometries["geometry"].iloc[0], LineString):
            relevant_geometries["coverage"] = (
                relevant_geometries["intersection"].length / relevant_geometries["geometry"].length
            )
        else:
            relevant_geometries["coverage"] = (
                relevant_geometries["intersection"].area / relevant_geometries["geometry"].area
            )

        # Store the results back into the original metadata DataFrame.
        # Only relevant_geometries (spatial-index hits) have coverage values; all
        # other streets get 0.0.  Build a full-length array before converting to a
        # Polars Series so the lengths always match self.metadata.
        coverage_values = [0.0] * len(self.metadata)
        for arr_pos, orig_idx in enumerate(possible_matches_index):
            coverage_values[orig_idx] = relevant_geometries["coverage"].iloc[arr_pos]
        self.metadata = self.metadata.with_columns(pl.Series("coverage", coverage_values))

    def _compute_count_metadata_grid_street(self) -> None:
        # Group by the specified columns and count the rows in each group
        grouped = self.joined.group_by(self.columns_to_group_by).agg(pl.len().alias("count"))
        # join self.metadata with join on index and index_right
        self.metadata = self.metadata.join(grouped, on=self.columns_to_group_by, how="left")

    def _compute_days_elapsed_metadata_grid_street(self) -> None:
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
            ((pl.col("max_datetime") - pl.col("min_datetime")).dt.total_days()).alias("days_elapsed")
        )
        # Join self.metadata with grouped on the specified columns, only including the 'days_elapsed' column
        self.metadata = self.metadata.join(
            grouped.select([self.columns_to_group_by, "days_elapsed"]),
            on=self.columns_to_group_by,
            how="left",
        )

    def _compute_most_recent_date_metadata_grid_street(self) -> None:
        # number of days between the most recent point (self.gdf) and the most recent point within self.metadata
        # calculate the age of the most recent point
        # Group by the specified columns and calculate the max datetime
        grouped = self.joined.group_by(self.columns_to_group_by).agg(pl.max("local_datetime").alias("most_recent_date"))
        # Convert the most recent datetime to date (string format)
        grouped = grouped.with_columns(pl.col("most_recent_date").dt.strftime("%Y-%m-%d").alias("most_recent_date"))
        # Join self.metadata with grouped on the specified columns, only including the 'most_recent_date' column
        self.metadata = self.metadata.join(
            grouped.select([self.columns_to_group_by, "most_recent_date"]),
            on=self.columns_to_group_by,
            how="left",
        )

    def _compute_oldest_date_metadata_grid_street(self) -> None:
        # number of days between the oldest point (self.gdf) and the oldest point within self.metadata
        # calculate the age of the oldest point
        # Group by the specified columns and calculate the min datetime
        grouped = self.joined.group_by(self.columns_to_group_by).agg(pl.min("local_datetime").alias("oldest_date"))
        # Convert the oldest datetime to date (string format)
        grouped = grouped.with_columns(pl.col("oldest_date").dt.strftime("%Y-%m-%d").alias("oldest_date"))
        # Join self.metadata with grouped on the specified columns, only including the 'oldest_date' column
        self.metadata = self.metadata.join(
            grouped.select([self.columns_to_group_by, "oldest_date"]),
            on=self.columns_to_group_by,
            how="left",
        )

    def _compute_number_of_years_metadata_grid_street(self) -> None:
        # number of unique years in the dataset
        # spaital join self.gdf to self.metadata
        # Extract the year from 'local_datetime' and create a new column
        grouped = self.joined.with_columns(pl.col("local_datetime").dt.year().alias("year"))
        # Group by the specified columns and calculate the number of unique years
        grouped = grouped.group_by(self.columns_to_group_by).agg(pl.col("year").n_unique().alias("number_of_years"))
        # Join self.metadata with grouped on the specified columns, only including the 'number_of_years' column
        self.metadata = self.metadata.join(
            grouped.select([self.columns_to_group_by, "number_of_years"]),
            on=self.columns_to_group_by,
            how="left",
        )

    def _compute_number_of_months_metadata_grid_street(self) -> None:
        # number of unique months in the dataset
        # spaital join self.gdf to self.metadata
        # Extract the month from 'local_datetime' and create a new column
        grouped = self.joined.with_columns(pl.col("local_datetime").dt.month().alias("month"))
        # Group by the specified columns and calculate the number of unique months
        grouped = grouped.group_by(self.columns_to_group_by).agg(pl.col("month").n_unique().alias("number_of_months"))
        # Join self.metadata with grouped on the specified columns, only including the 'number_of_months' column
        self.metadata = self.metadata.join(
            grouped.select([self.columns_to_group_by, "number_of_months"]),
            on=self.columns_to_group_by,
            how="left",
        )

    def _compute_number_of_days_metadata_grid_street(self) -> None:
        # number of unique days in the dataset
        # spaital join self.gdf to self.metadata
        # Extract the day from 'local_datetime' and create a new column
        grouped = self.joined.with_columns(pl.col("local_datetime").dt.day().alias("day"))
        # Group by the specified columns and calculate the number of unique days
        grouped = grouped.group_by(self.columns_to_group_by).agg(pl.col("day").n_unique().alias("number_of_days"))
        # Join self.metadata with grouped on the specified columns, only including the 'number_of_days' column
        self.metadata = self.metadata.join(
            grouped.select([self.columns_to_group_by, "number_of_days"]),
            on=self.columns_to_group_by,
            how="left",
        )

    def _compute_number_of_hours_metadata_grid_street(self) -> None:
        # number of unique hours in the dataset
        # spaital join self.gdf to self.metadata
        # Extract the hour from 'local_datetime' and create a new column
        grouped = self.joined.with_columns(pl.col("local_datetime").dt.hour().alias("hour"))
        # Group by the specified columns and calculate the number of unique hours
        grouped = grouped.group_by(self.columns_to_group_by).agg(pl.col("hour").n_unique().alias("number_of_hours"))
        # Join self.metadata with grouped on the specified columns, only including the 'number_of_hours' column
        self.metadata = self.metadata.join(
            grouped.select([self.columns_to_group_by, "number_of_hours"]),
            on=self.columns_to_group_by,
            how="left",
        )

    def _compute_number_of_days_of_week_metadata_grid_street(self) -> None:
        # number of unique days of week in the dataset
        # spaital join self.gdf to self.metadata
        # Extract the day of week from 'local_datetime' and create a new column
        grouped = self.joined.with_columns(pl.col("local_datetime").dt.weekday().alias("day_of_week"))
        # Group by the specified columns and calculate the number of unique days of week
        grouped = grouped.group_by(self.columns_to_group_by).agg(
            pl.col("day_of_week").n_unique().alias("number_of_days_of_week")
        )
        # Join self.metadata with grouped on the specified columns, only including the 'number_of_days_of_week' column
        self.metadata = self.metadata.join(
            grouped.select([self.columns_to_group_by, "number_of_days_of_week"]),
            on=self.columns_to_group_by,
            how="left",
        )

    def _get_joined_daynight(self) -> pl.DataFrame:
        """Return self.joined with daytime_nighttime column, computing once and caching."""
        if self._joined_daynight_cache is None:
            self._joined_daynight_cache = _day_or_night(self.joined)
        return self._joined_daynight_cache

    def _compute_number_of_daytime_metadata_grid_street(self) -> None:
        grouped = self._get_joined_daynight()
        grouped = grouped.with_columns(pl.col("daytime_nighttime").eq(pl.lit("daytime")).alias("daytime"))
        grouped = grouped.group_by(self.columns_to_group_by).agg(pl.col("daytime").sum().alias("number_of_daytime"))
        self.metadata = self.metadata.join(
            grouped.select([self.columns_to_group_by, "number_of_daytime"]),
            on=self.columns_to_group_by,
            how="left",
        )

    def _compute_number_of_nighttime_metadata_grid_street(self) -> None:
        grouped = self._get_joined_daynight()
        grouped = grouped.with_columns(pl.col("daytime_nighttime").eq(pl.lit("nighttime")).alias("nighttime"))
        grouped = grouped.group_by(self.columns_to_group_by).agg(pl.col("nighttime").sum().alias("number_of_nighttime"))
        self.metadata = self.metadata.join(
            grouped.select([self.columns_to_group_by, "number_of_nighttime"]),
            on=self.columns_to_group_by,
            how="left",
        )

    def _get_seasons_grouped_cache(self) -> pl.DataFrame:
        """Compute all four season counts in one group_by, caching the result."""
        if self._seasons_grouped_cache is None:
            season_col = _season_expr("local_datetime", "lat")
            self._seasons_grouped_cache = (
                self.joined.with_columns(season_col.alias("_season"))
                .group_by(self.columns_to_group_by)
                .agg(
                    (pl.col("_season") == "Spring").sum().alias("number_of_spring"),
                    (pl.col("_season") == "Summer").sum().alias("number_of_summer"),
                    (pl.col("_season") == "Autumn").sum().alias("number_of_autumn"),
                    (pl.col("_season") == "Winter").sum().alias("number_of_winter"),
                )
            )
        return self._seasons_grouped_cache

    def _compute_number_of_spring_metadata_grid_street(self) -> None:
        grouped = self._get_seasons_grouped_cache()
        self.metadata = self.metadata.join(
            grouped.select([self.columns_to_group_by, "number_of_spring"]),
            on=self.columns_to_group_by,
            how="left",
        )

    def _compute_number_of_summer_metadata_grid_street(self) -> None:
        grouped = self._get_seasons_grouped_cache()
        self.metadata = self.metadata.join(
            grouped.select([self.columns_to_group_by, "number_of_summer"]),
            on=self.columns_to_group_by,
            how="left",
        )

    def _compute_number_of_autumn_metadata_grid_street(self) -> None:
        grouped = self._get_seasons_grouped_cache()
        self.metadata = self.metadata.join(
            grouped.select([self.columns_to_group_by, "number_of_autumn"]),
            on=self.columns_to_group_by,
            how="left",
        )

    def _compute_number_of_winter_metadata_grid_street(self) -> None:
        grouped = self._get_seasons_grouped_cache()
        self.metadata = self.metadata.join(
            grouped.select([self.columns_to_group_by, "number_of_winter"]),
            on=self.columns_to_group_by,
            how="left",
        )

    def _compute_average_compass_angle_metadata_grid_street(self) -> None:
        # average compass angle of the points within self.metadata
        # calculate the average compass angle
        grouped = self.joined.group_by(self.columns_to_group_by).agg(
            pl.col("compass_angle").mean().alias("average_compass_angle")
        )
        # Join self.metadata with grouped on the specified columns, only including the 'average_compass_angle' column
        self.metadata = self.metadata.join(
            grouped.select([self.columns_to_group_by, "average_compass_angle"]),
            on=self.columns_to_group_by,
            how="left",
        )

    def _compute_average_relative_angle_metadata_grid_street(self) -> None:
        # average relative angle of the points within self.metadata
        nearest_line = pl.DataFrame(self.nearest_line[["id", "relative_angle"]])
        grouped = self.joined.join(nearest_line, on="id", how="left")
        # calculate the average relative angle
        grouped = grouped.group_by(self.columns_to_group_by).agg(
            pl.col("relative_angle").mean().alias("average_relative_angle")
        )
        # Join self.metadata with grouped on the specified columns, only including the 'average_relative_angle' column
        self.metadata = self.metadata.join(
            grouped.select([self.columns_to_group_by, "average_relative_angle"]),
            on=self.columns_to_group_by,
            how="left",
        )

    def _compute_average_is_pano_metadata_grid_street(self) -> None:
        # average is_pano of the points within self.metadata
        # calculate the average is_pano
        grouped = self.joined.group_by(self.columns_to_group_by).agg(pl.col("is_pano").mean().alias("average_is_pano"))
        # Join self.metadata with grouped on the specified columns, only including the 'average_is_pano' column
        self.metadata = self.metadata.join(
            grouped.select([self.columns_to_group_by, "average_is_pano"]),
            on=self.columns_to_group_by,
            how="left",
        )

    def _compute_number_of_users_metadata_grid_street(self) -> None:
        # number of unique users in the dataset
        # calculate the number of unique users
        grouped = self.joined.group_by(self.columns_to_group_by).agg(
            pl.col("creator_id").n_unique().alias("number_of_users")
        )
        # Join self.metadata with grouped on the specified columns, only including the 'number_of_users' column
        self.metadata = self.metadata.join(
            grouped.select([self.columns_to_group_by, "number_of_users"]),
            on=self.columns_to_group_by,
            how="left",
        )

    def _compute_number_of_sequences_metadata_grid_street(self) -> None:
        # number of unique sequences in the dataset
        # calculate the number of unique sequences
        grouped = self.joined.group_by(self.columns_to_group_by).agg(
            pl.col("sequence_id").n_unique().alias("number_of_sequences")
        )
        # Join self.metadata with grouped on the specified columns, only including the 'number_of_sequences' column
        self.metadata = self.metadata.join(
            grouped.select([self.columns_to_group_by, "number_of_sequences"]),
            on=self.columns_to_group_by,
            how="left",
        )

    def _compute_number_of_organizations_metadata_grid_street(self) -> None:
        if "organization_id" not in self.joined.columns:
            self.metadata = self.metadata.with_columns(pl.lit(None).cast(pl.UInt32).alias("number_of_organizations"))
            return
        grouped = self.joined.group_by(self.columns_to_group_by).agg(
            pl.col("organization_id").n_unique().alias("number_of_organizations")
        )
        self.metadata = self.metadata.join(
            grouped.select([self.columns_to_group_by, "number_of_organizations"]),
            on=self.columns_to_group_by,
            how="left",
        )

    def _compute_speed_metadata_grid_street(self) -> None:
        # average speed of the points within self.metadata
        # calculate the average speed
        grouped = self.joined.group_by(self.columns_to_group_by).agg(
            pl.col("speed_kmh").mean().alias("average_speed_kmh")
        )
        # Join self.metadata with grouped on the specified columns, only including the 'average_speed_kmh' column
        self.metadata = self.metadata.join(
            grouped.select([self.columns_to_group_by, "average_speed_kmh"]),
            on=self.columns_to_group_by,
            how="left",
        )

    def _compute_image_metadata(self, indicator_list: str) -> pl.DataFrame:
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

    def _compute_grid_metadata(self, indicator_list: str) -> pl.DataFrame:
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

    def _compute_street_metadata(self, indicator_list: str) -> pl.DataFrame:
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
        path_output: Optional[Union[str, Path]] = None,
        max_distance: int = 50,
    ) -> "pd.DataFrame":
        """Compute metadata for the dataset.

        Args:
            unit (str): The unit of analysis. Defaults to "image".
            grid_resolution (int): The resolution of the H3 grid.
                Defaults to 7.
            indicator_list (str): List of indicators to compute metadata
                for. Use space- separated string of indicators or "all".
                Options for image-level metadata: "year", "month",
                "day", "hour", "day_of_week", "relative_angle", "h3_id",
                "speed_kmh". Options for grid-level metadata:
                "coverage", "count", "days_elapsed", "most_recent_date",
                "oldest_date", "number_of_years", "number_of_months",
                "number_of_days", "number_of_hours",
                "number_of_days_of_week", "number_of_daytime",
                "number_of_nighttime", "number_of_spring",
                "number_of_summer", "number_of_autumn",
                "number_of_winter", "average_compass_angle",
                "average_relative_angle", "average_is_pano",
                "number_of_users", "number_of_sequences",
                "number_of_organizations", "average_speed_kmh". Defaults
                to "all".
            path_output (Union[str, Path]): Path to save the output
                metadata. Defaults to None.
            max_distance (int): The maximum distance to search for the
                nearest street segment. Defaults to 50.

        Returns:
            pd.DataFrame: A DataFrame containing the computed metadata.
        """
        if self.logger is not None:
            # record the arguments
            self.logger.log_args(
                f"{type(self).__name__} compute_metadata",
                unit=unit,
                grid_resolution=grid_resolution,
                coverage_buffer=coverage_buffer,
                indicator_list=indicator_list,
                path_output=path_output,
                max_distance=max_distance,
            )

        # set coverage buffer as a class attribute
        self.coverage_buffer = coverage_buffer

        # Clear per-call caches so repeated compute_metadata calls don't use stale data
        self._joined_daynight_cache = None
        self._seasons_grouped_cache = None
        # Reset self.df from disk (through normalization) so timezone/other columns added in
        # a previous call don't cause DuplicateError when compute_metadata is called more than once.
        self.df = self._normalize_columns(self._load_dataframe())

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
            self.df = self.df.with_columns(_season_expr("local_datetime", "lat").alias("season"))

        # check indicator_list and pre-compute metadata, e.g., speed
        if "all" in indicator_list or "speed" in indicator_list:
            self.df = _compute_speed(self.df)

        # Determine which expensive resources are actually needed
        needs_street_network = (
            unit == "street"
            or "all" in indicator_list
            or "relative_angle" in indicator_list
            or "average_relative_angle" in indicator_list
        )
        needs_gdf = unit in ("grid", "street") or needs_street_network

        if needs_street_network:
            self._ensure_street_network()

        # create self.gdf as a GeoDataFrame for spatial operations (only when needed)
        if needs_gdf:
            self.gdf = gpd.GeoDataFrame(
                self.df.to_pandas(),
                geometry=gpd.points_from_xy(self.df["lon"], self.df["lat"]),
                crs=4326,
            )
            if self.projected_crs is not None:
                self.gdf = self.gdf.to_crs(crs=self.projected_crs)
            else:
                # No street network loaded — derive projection from the points themselves
                self.gdf = ox.projection.project_gdf(self.gdf)
                self.projected_crs = self.gdf.crs

        # check indicator_list and pre-compute metadata, e.g., relative_angle
        if "all" in indicator_list or "relative_angle" in indicator_list:
            # Perform the nearest join with the street network
            nearest_line = self.gdf.sjoin_nearest(
                self.street_network[["geometry", "angle"]],  # Ensure only necessary columns are loaded
                how="left",
                max_distance=max_distance,
                distance_col="dist",  # Save distances to avoid recomputing
            )

            # Calculate the relative angle and ensure it is within 0-180 degrees because we don't know the direction of the street
            nearest_line["relative_angle"] = ((nearest_line["angle"] - nearest_line["compass_angle"]) % 180).abs()

            # Reduce to the essential data and perform a group_by operation to find the minimum angle for each 'id'
            min_angle = nearest_line[["id", "relative_angle"]].groupby("id", as_index=False).min()

            # Merge back to the original GeoDataFrame to include all original data
            self.nearest_line = self.gdf.merge(min_angle, on="id", how="left")

        # run the appropriate function to compute metadata based on the unit of analysis
        if unit == "image":
            # after here, there's no gpd.GeoDataFrame conversion, so let's convert local_datetime to datetime for Polars
            self.df = self.df.with_columns(pl.col("local_datetime").dt.replace_time_zone(None).alias("local_datetime"))
            df = self._compute_image_metadata(indicator_list)
        elif unit == "grid":
            self.metadata_geometry = _create_hexagon(
                pl.from_dataframe(self.gdf[["lat", "lon"]]), resolution=grid_resolution
            )
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
                pl.col("local_datetime").dt.replace_time_zone(None).alias("local_datetime")
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
                pl.col("local_datetime").dt.replace_time_zone(None).alias("local_datetime")
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
                    raise ValueError("Invalid file format provided. Please provide a .csv, .shp, or .geojson file.")
        df = df.to_pandas()
        return df
