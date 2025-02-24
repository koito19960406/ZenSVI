# encoding: utf-8
# author: yujunhou
# contact: hou.yujun@u.nus.edu

import geopandas as gp
import pandas as pd
import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


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
def get_data_from_url(url):
    """Get data from a KartaView API URL.

    Args:
        url (str): The KartaView API URL to query.

    Returns:
        dict: The JSON response data if successful, None otherwise.
    """
    r = requests.get(url, timeout=10)  # Send GET request with a 10-second timeout

    # Try to parse the JSON response and handle errors based on the API response
    try:
        response_json = r.json()  # Parse the response as JSON
        # Extract apiCode and apiMessage to check for errors
        api_code = response_json.get("status", {}).get("apiCode", 0)
        api_message = response_json.get("status", {}).get("apiMessage", "Unknown API error")

        # If the status code is 400 (Bad Request), print the error message and raise an exception
        if r.status_code == 400:
            print(f"HTTP 400 Error: {api_message}")  # Print the API-specific error message
            raise ValueError(f"API Error: {api_message}")  # Raise an exception to stop further processing

        # If apiCode is 600, this indicates a successful request with the expected data
        if api_code == 600:
            return response_json.get("result", {}).get("data", None)  # Return the data from the response

        # For any other error (apiCode other than 600), raise an exception
        raise ValueError(f"API Error: {api_message}")  # Raise exception with API message

    except ValueError as e:
        # Handle JSON parsing errors or other exceptions during the response processing
        raise ValueError(f"Error processing response JSON: {e}")


def data_to_dataframe(data):
    """Convert JSON data to a pandas DataFrame.

    Args:
        data (dict): JSON data from KartaView API.

    Returns:
        pd.DataFrame: DataFrame containing the data.
    """
    try:
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        print(f"Error: {e}")


def get_points_in_sequence(sequenceId):
    """Get all photo points in a KartaView sequence.

    Args:
        sequenceId (str): ID of the KartaView sequence.

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame containing photo points, or empty DataFrame if no data.
    """
    try:
        url = f"https://api.openstreetcam.org/2.0/sequence/{sequenceId}/photos?itemsPerPage=1000000&join=user"
        try:
            data = get_data_from_url(url)
        except Exception as e:
            print(f"Request failed: {e}")
            data = None  # Explicitly set to None if the request fails
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


def get_sequences_in_shape(shape):
    """Get all KartaView sequences within a shape boundary.

    Args:
        shape (geopandas.GeoDataFrame): GeoDataFrame containing boundary shape.

    Returns:
        pd.DataFrame: DataFrame containing sequence data.
    """
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
            try:
                data = get_data_from_url(url)
            except Exception as e:
                print(f"Request failed: {e}")
                data = None  # Explicitly set to None if the request fails
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
    """Get all KartaView photo points within a shape boundary.

    Args:
        shape (geopandas.GeoDataFrame): GeoDataFrame containing boundary shape.

    Returns:
        pd.DataFrame: DataFrame containing photo points and sequence metadata.
    """
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
