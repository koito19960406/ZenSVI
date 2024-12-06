import osmnx as ox


def standardize_column_names(df):
    """Standardizes column names for latitude and longitude in a DataFrame.

    Converts various common column names for latitude and longitude coordinates
    to standardized 'latitude' and 'longitude' names.

    Args:
        df (pandas.DataFrame): DataFrame containing coordinate columns.

    Returns:
        pandas.DataFrame: DataFrame with standardized column names.
    """
    longitude_variants = ["longitude", "long", "lon", "lng", "x"]
    latitude_variants = ["latitude", "lat", "lt", "y"]
    # convert all column names to lowercase
    df.columns = [col.lower() for col in df.columns]
    for col in df.columns:
        if col in longitude_variants:
            df.rename(columns={col: "longitude"}, inplace=True)
        elif col in latitude_variants:
            df.rename(columns={col: "latitude"}, inplace=True)
    return df


def create_buffer_gdf(gdf, buffer_distance):
    """Creates a buffer around geometries in a GeoDataFrame.

    Projects the GeoDataFrame if needed, creates a buffer of specified distance
    around the geometries, and reprojects back to WGS84.

    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame containing geometries.
        buffer_distance (float): Buffer distance in meters.

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame with buffered geometries.
    """
    if not gdf.crs:
        gdf = gdf.set_crs("EPSG:4326")

    if not gdf.crs.is_projected:
        gdf = ox.projection.project_gdf(gdf)

    # Buffer the points by buffer_distance
    gdf["geometry"] = gdf.buffer(buffer_distance)

    # Project the GeoDataFrame back to EPSG:4326
    gdf = gdf.to_crs("EPSG:4326")

    return gdf


def check_and_buffer(gdf, buffer):
    """Checks geometry type and creates buffer if appropriate.

    Validates that point and line geometries have non-zero buffers,
    then creates the buffer if validation passes.

    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame to check and buffer.
        buffer (float): Buffer distance in meters.

    Returns:
        geopandas.GeoDataFrame: Buffered GeoDataFrame if validation passes.

    Raises:
        ValueError: If buffer is 0 for point or line geometries.
    """
    # check geometry type
    geom_type = gdf.geom_type.unique()[0]
    # raise an error if the geometry is a Point/MultiPoint or a LineString/MultiLineString & buffer = 0
    if (
        geom_type == "Point" or geom_type == "MultiPoint" or geom_type == "LineString" or geom_type == "MultiLineString"
    ) and buffer == 0:
        raise ValueError(
            "Buffer cannot be 0 if the geometry is either a Point/MultiPoint or a LineString/MultiLineString"
        )
    else:
        gdf = create_buffer_gdf(gdf, buffer)
        return gdf
