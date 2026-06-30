# encoding: utf-8
# author: yujunhou
# contact: hou.yujun@u.nus.edu

from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gp
import pandas as pd
import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from zensvi.download.utils.geoprocess import GeoProcessor
from zensvi.utils.log import verbosity_tqdm

API_BASE = "https://api.openstreetcam.org/2.0"
MAX_ITEMS_PER_PAGE = 150  # KartaView 2.0 hard cap; larger values return apiCode 400.
DEFAULT_RADIUS = 50  # meters. KartaView proximity queries time out (408) at larger radii in dense areas.
MIN_RADIUS = 10  # meters. Floor when shrinking radius on a timeout.


# Function to determine whether the exception should trigger a retry
def is_retriable_exception(exception):
    """Determine if the exception should trigger a retry."""
    if isinstance(exception, requests.exceptions.RequestException):
        # Only retry for server errors (500, 503) and rate limits (429)
        if exception.response is not None:
            status_code = exception.response.status_code
            # Retry only on specific server-side issues
            if status_code in {500, 503, 429}:
                return True  # Trigger retry for these error codes
    return False  # Don't retry on other errors (like client-side 400)


# The main function to fetch data from a URL with retry logic
@retry(
    stop=stop_after_attempt(5),  # Retry up to 5 times
    wait=wait_exponential(multiplier=1, min=2, max=10),  # Exponential backoff: 2s, 4s, 8s... up to 10s
    retry=retry_if_exception(is_retriable_exception),  # Only retry if `is_retriable_exception` returns True
)
def get_result_from_url(url):
    """Query a KartaView 2.0 API URL and return its ``result`` object.

    Args:
        url (str): The KartaView API URL to query.

    Returns:
        dict or None: The ``result`` dict (containing ``data`` and ``hasMoreData``) on a
            successful query, or None when the API reports an empty response (apiCode 601).

    Raises:
        ValueError: On an API error (e.g. apiCode 400 bad request, 408 query timeout).
    """
    r = requests.get(url, timeout=30)

    try:
        response_json = r.json()  # Parse the response as JSON
    except ValueError as e:
        raise ValueError(f"Error processing response JSON: {e}")

    status = response_json.get("status", {})
    api_code = status.get("apiCode", 0)
    api_message = status.get("apiMessage", "Unknown API error")

    # apiCode 600 indicates a successful request with the expected data
    if api_code == 600:
        return response_json.get("result", {}) or {}
    # apiCode 601 indicates a successful but empty response (no data) — not an error
    if api_code == 601:
        return None
    # Any other apiCode (e.g. 400 bad request, 408 query timeout) is an error
    raise ValueError(f"API Error: {api_message}")


def get_all_pages(base_url, items_per_page=MAX_ITEMS_PER_PAGE):
    """Fetch every page of a KartaView 2.0 list endpoint.

    Pages with ``itemsPerPage<=150`` (the API cap) and loops until ``result.hasMoreData``
    is False.

    Args:
        base_url (str): The endpoint URL, with any filter parameters but without
            ``itemsPerPage``/``page``.
        items_per_page (int): Items per page, capped at 150. Defaults to 150.

    Returns:
        list: All items collected across pages.
    """
    items_per_page = min(items_per_page, MAX_ITEMS_PER_PAGE)
    all_data = []
    page = 1
    while True:
        sep = "&" if "?" in base_url else "?"
        url = f"{base_url}{sep}itemsPerPage={items_per_page}&page={page}"
        result = get_result_from_url(url)
        if not result:
            break
        data = result.get("data") or []
        all_data.extend(data)
        if not result.get("hasMoreData") or len(data) == 0:
            break
        page += 1
    return all_data


def get_photos_near_point(lat, lon, radius=DEFAULT_RADIUS, min_radius=MIN_RADIUS):
    """Fetch all KartaView photos near a point via the proximity endpoint.

    KartaView 2.0 no longer supports bounding-box queries; discovery is point + radius on
    ``/2.0/photo/``. The query times out (408) in dense areas at larger radii, so on a
    timeout the radius is halved and retried down to ``min_radius``.

    Args:
        lat (float): Latitude of the query point.
        lon (float): Longitude of the query point.
        radius (int): Search radius in meters. Defaults to 50.
        min_radius (int): Smallest radius to fall back to before giving up. Defaults to 10.

    Returns:
        list: Photo dicts near the point (each with ``id``, ``lat``, ``lng``, ``shotDate``,
            ``fileurlProc``, nested ``sequence``, etc.). Empty if none found or all retries time out.
    """
    r = radius
    while r >= min_radius:
        base_url = f"{API_BASE}/photo/?lat={lat}&lng={lon}&radius={r}" "&join=sequence&orderBy=id&orderDirection=desc"
        try:
            return get_all_pages(base_url)
        except ValueError as e:
            # Density-driven 408 timeout: shrink the radius and retry this point.
            if "timeout" in str(e).lower() or "408" in str(e):
                r = r // 2
                continue
            print(f"Request failed: {e}")
            return []
    print(f"Skipping point ({lat}, {lon}): KartaView still timed out at radius={min_radius}m")
    return []


def clip_points_with_shape(points, shape):
    """Clip points to within a shape boundary.

    Args:
        points (geopandas.GeoDataFrame): GeoDataFrame containing points.
        shape (geopandas.GeoDataFrame): GeoDataFrame containing boundary shape.

    Returns:
        geopandas.GeoDataFrame: Points clipped to shape boundary.
    """
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


def _flatten_photos(df):
    """Flatten the proximity-response photo frame to the downloader's schema.

    Extracts ``sequenceId``/``userId`` from the nested ``sequence`` object, derives an
    ``is_pano`` flag, and renames ``lng`` -> ``lon``.

    Args:
        df (pandas.DataFrame): Raw photos from the proximity endpoint.

    Returns:
        pandas.DataFrame: Photos with flat columns and no nested ``sequence`` dict.
    """
    if "sequence" in df.columns:
        df["sequenceId"] = df["sequence"].apply(lambda s: s.get("id") if isinstance(s, dict) else None)
        df["userId"] = df["sequence"].apply(lambda s: s.get("userId") if isinstance(s, dict) else None)
        df = df.drop(columns=["sequence"])
    if "fieldOfView" in df.columns:
        df["is_pano"] = df["fieldOfView"].apply(lambda v: str(v) == "360")
    elif "projection" in df.columns:
        df["is_pano"] = df["projection"].apply(lambda v: str(v).upper() == "SPHERE")
    df = df.rename(columns={"lng": "lon"})
    return df


def get_points_in_shape(
    shape,
    distance=DEFAULT_RADIUS,
    grid=False,
    grid_size=DEFAULT_RADIUS,
    radius=DEFAULT_RADIUS,
    verbosity=1,
    max_workers=None,
):
    """Discover all KartaView photo points within a shape.

    Generates sample points within the shape using :class:`GeoProcessor` (street-network
    sampling by default, with a grid fallback), runs a proximity query at each point,
    dedupes the collected photos by id, and clips them to the shape.

    Args:
        shape (geopandas.GeoDataFrame): Boundary shape (EPSG:4326) to search within.
        distance (float): Spacing in meters between sample points along the street network.
            Defaults to 50 (matched to the proximity radius). Defaults to 50.
        grid (bool): Use grid sampling instead of the street network. Defaults to False.
        grid_size (float): Grid cell size in meters when ``grid`` is True. Defaults to 50.
        radius (int): Proximity search radius in meters per sample point. Defaults to 50.
        verbosity (int): Progress-bar verbosity. Defaults to 1.
        max_workers (int, optional): Threads for concurrent proximity queries. Defaults to None.

    Returns:
        pandas.DataFrame or None: Photo points with columns including ``id``, ``lon``, ``lat``,
            ``sequenceId``, ``shotDate``, ``fileurlProc``; or None if no photos are found.
    """
    try:
        # 1. Generate sample points within the shape (same approach as GSVDownloader).
        geo_processor = GeoProcessor(
            shape.copy(), distance=distance, grid=grid, grid_size=grid_size, verbosity=verbosity
        )
        points_df = geo_processor.get_lat_lon()  # columns: longitude, latitude
        if points_df is None or points_df.empty:
            print("No data from KartaView.")
            return None

        # 2. Run a proximity query at each sample point (concurrently) and collect photos.
        photos = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(get_photos_near_point, row.latitude, row.longitude, radius)
                for row in points_df.itertuples(index=False)
            ]
            for future in verbosity_tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Querying KartaView",
                verbosity=verbosity,
                level=1,
            ):
                photos.extend(future.result())

        if not photos:
            print("No data from KartaView.")
            return None

        # 3. Flatten, dedupe by photo id, and clip to the shape.
        df = _flatten_photos(pd.DataFrame(photos))
        df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
        gdf = gp.GeoDataFrame(df, geometry=gp.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
        gdf = clip_points_with_shape(gdf, shape)
        if gdf is None or gdf.empty:
            print("No data from KartaView.")
            return None

        points_all = pd.DataFrame(gdf.drop(columns="geometry")).reset_index(drop=True)
        nSeqs = points_all["sequenceId"].nunique() if "sequenceId" in points_all.columns else 0
        print("Download complete, collected", nSeqs, "sequences", len(points_all), "points")
        return points_all
    except Exception as e:
        print(f"Error: {e}")
