import random
import re

import requests


def _panoids_url(lat, lon):
    """Construct URL for Google Street View panorama metadata.

    Args:
        lat (float): Latitude coordinate
        lon (float): Longitude coordinate

    Returns:
        str: Formatted URL for the Google Street View metadata API
    """
    url = "https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{0:}!4d{1:}!2d50!3m10!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2!4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2&callback=_xdc_._v2mub5"
    return url.format(lat, lon)


def _panoids_data(lat, lon, proxies):
    """Fetch panorama metadata from Google Street View API using proxies.

    Args:
        lat (float): Latitude coordinate
        lon (float): Longitude coordinate
        proxies (list): List of proxy servers to use

    Returns:
        str: Raw response text from the API

    Note:
        Will retry with different proxies if a request fails
    """
    url = _panoids_url(lat, lon)
    while True:
        proxy = random.choice(proxies)
        try:
            resp = requests.get(url, proxies=proxy, timeout=5)
            return resp.text
        except Exception as e:
            print(f"Proxy {proxy} is not working. Exception: {e}")
            continue


def panoids(lat, lon, proxies):
    """Get panorama IDs and metadata for a location from Google Street View.

    Args:
        lat (float): Latitude coordinate
        lon (float): Longitude coordinate
        proxies (list): List of proxy servers to use

    Returns:
        list: List of dictionaries containing panorama metadata with keys:
            - panoid (str): Panorama ID
            - lat (float): Latitude of panorama
            - lon (float): Longitude of panorama
            - year (int): Year photo was taken (if available)
            - month (int): Month photo was taken (if available)
    """
    resp = _panoids_data(lat, lon, proxies)

    # Get all the panorama ids and coordinates
    pans = re.findall(r'\[[0-9]+,"(.+?)"\].+?\[\[null,null,(-?[0-9]+.[0-9]+),(-?[0-9]+.[0-9]+)', resp)
    pans = [{"panoid": p[0], "lat": float(p[1]), "lon": float(p[2])} for p in pans]

    # Get all the dates
    dates = re.findall(r"([0-9]?[0-9]?[0-9])?,?\[(20[0-9][0-9]),([0-9]+)\]", resp)
    dates = [list(d)[1:] for d in dates]

    if len(dates) > 0:
        # Convert all values to integers
        dates = [[int(v) for v in d] for d in dates]

        # Make sure the month value is between 1-12
        dates = [d for d in dates if d[1] <= 12 and d[1] >= 1]

        # The last date belongs to the first panorama
        year, month = dates.pop(-1)
        pans[0].update({"year": year, "month": month})

        # The dates then apply in reverse order to the bottom panoramas
        dates.reverse()
        for i, (year, month) in enumerate(dates):
            pans[-1 - i].update({"year": year, "month": month})

    return pans
