import aiohttp
import aiohttp_socks
import asyncio
import random
import re
import python_socks
from aiohttp_proxy import ProxyConnector, ProxyType

async def _panoids_url(lat, lon):
    url = "https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{0:}!4d{1:}!2d50!3m10!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2!4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2&callback=_xdc_._v2mub5"
    return url.format(lat, lon)

async def _panoids_data(lat, lon, proxies):
    sem = asyncio.Semaphore(1)  # Limit the number of concurrent connections
    while True:  # Loop will continue until a successful request
        proxy = random.choice(proxies)
        url = await _panoids_url(lat, lon)
        try:
            async with sem:  # Acquire the semaphore
                connector = ProxyConnector.from_url(proxy, limit=1)
                # connector = aiohttp.TCPConnector(limit=1)
                async with aiohttp.ClientSession(connector=connector) as session:
                    async with session.get(url) as response:
                        return await response.text()
                # if proxy.startswith('http'):
                #     async with aiohttp.ClientSession() as session:
                #         async with session.get(url, proxy=proxy, timeout=5) as resp:
                #             if resp.status == 200:  # Check if request is successful
                #                 return await resp.text()
                # elif proxy.startswith('socks'):
                #     session = aiohttp.ClientSession(connector=aiohttp_socks.SocksConnector.from_url(proxy))
                #     async with session:
                #         async with session.get(url, timeout=5) as resp:
                #             if resp.status == 200:  # Check if request is successful
                #                 return await resp.text()
        except (aiohttp.ClientHttpProxyError, python_socks._errors.ProxyConnectionError, asyncio.TimeoutError, ConnectionRefusedError, aiohttp.ClientOSError) as e:  # Catch proxy errors and timeout errors
            print(f"Proxy {proxy} failed, trying another one.... Error message: {e}")
            continue


async def panoids(lat, lon, proxies, closest=False, disp=False):
    resp = await _panoids_data(lat, lon, proxies)

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

    if disp:
        for pan in pans:
            print(pan)

    if closest and len(dates) > 0:
        return [pans[i] for i in range(len(dates))]
    else:
        return pans
