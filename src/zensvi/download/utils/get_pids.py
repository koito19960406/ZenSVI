import re
from datetime import datetime
import requests
import random

def _panoids_url(lat, lon):
    url = "https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{0:}!4d{1:}!2d50!3m10!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2!4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2&callback=_xdc_._v2mub5"
    return url.format(lat, lon)

def _panoids_data(lat, lon, proxies):
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
    resp = _panoids_data(lat, lon, proxies)

    # Get all the panorama ids and coordinates
    pans = re.findall('\[[0-9]+,"(.+?)"\].+?\[\[null,null,(-?[0-9]+.[0-9]+),(-?[0-9]+.[0-9]+)', resp)
    pans = [{"panoid": p[0], "lat": float(p[1]), "lon": float(p[2])} for p in pans]

    # Get all the dates
    dates = re.findall('([0-9]?[0-9]?[0-9])?,?\[(20[0-9][0-9]),([0-9]+)\]', resp)
    dates = [list(d)[1:] for d in dates]

    if len(dates) > 0:
        # Convert all values to integers
        dates = [[int(v) for v in d] for d in dates]

        # Make sure the month value is between 1-12
        dates = [d for d in dates if d[1] <= 12 and d[1] >= 1]

        # The last date belongs to the first panorama
        year, month = dates.pop(-1)
        pans[0].update({'year': year, "month": month})

        # The dates then apply in reverse order to the bottom panoramas
        dates.reverse()
        for i, (year, month) in enumerate(dates):
            pans[-1-i].update({'year': year, "month": month})

    return pans
