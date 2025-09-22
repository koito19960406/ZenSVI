import asyncio
import glob
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import aiohttp
import numpy as np
import requests
from PIL import Image
from requests.exceptions import ProxyError

from zensvi.utils.log import verbosity_tqdm


class ImageTool:
    """A class containing static methods for image manipulation and downloading."""

    @staticmethod
    def concat_horizontally(im1, im2):
        """Horizontally concatenates two images.

        Args:
            im1 (PIL.Image): First PIL image to concatenate.
            im2 (PIL.Image): Second PIL image to concatenate.

        Returns:
            PIL.Image: New image with im1 and im2 concatenated horizontally.
        """
        dst = Image.new("RGB", (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    @staticmethod
    def concat_vertically(im1, im2):
        """Vertically concatenates two images.

        Args:
            im1 (PIL.Image): First PIL image to concatenate.
            im2 (PIL.Image): Second PIL image to concatenate.

        Returns:
            PIL.Image: New image with im1 and im2 concatenated vertically.
        """
        dst = Image.new("RGB", (im1.width, im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        return dst

    @staticmethod
    def fetch_image_with_proxy(pano_id, zoom, x, y, ua, proxies):
        """Fetches a Google Street View image tile using a random proxy.

        Args:
            pano_id (str): Google Street View panorama ID.
            zoom (int): Zoom level for the image tile.
            x (int): X coordinate of the tile.
            y (int): Y coordinate of the tile.
            ua (str): User agent string for the HTTP request.
            proxies (list): List of proxy servers to use.

        Returns:
            PIL.Image: The fetched image tile.

        Raises:
            ProxyError: If the selected proxy fails to connect.
        """
        while True:
            # Choose a random proxy for each request
            proxy = random.choice(proxies)
            url_img = f"https://cbk0.google.com/cbk?output=tile&panoid={pano_id}&zoom={zoom}&x={x}&y={y}"
            try:
                image = Image.open(requests.get(url_img, headers=ua, proxies=proxy, stream=True).raw)
                return image
            except ProxyError as e:
                print(f"Proxy {proxy} is not working. Exception: {e}")
                continue

    @staticmethod
    async def fetch_image_with_proxy_async(session, semaphore, pano_id, zoom, x, y, ua, proxies):
        """Async version of fetch_image_with_proxy.

        Args:
            session (aiohttp.ClientSession): The aiohttp session for requests.
            semaphore (asyncio.Semaphore): Semaphore for controlling concurrency.
            pano_id (str): Google Street View panorama ID.
            zoom (int): Zoom level for the image tile.
            x (int): X coordinate of the tile.
            y (int): Y coordinate of the tile.
            ua (str): User agent string for the HTTP request.
            proxies (list): List of proxy servers to use.

        Returns:
            PIL.Image: The fetched image tile.
        """
        async with semaphore:
            for attempt in range(3):  # Max 3 attempts
                proxy = random.choice(proxies)
                url_img = f"https://cbk0.google.com/cbk?output=tile&panoid={pano_id}&zoom={zoom}&x={x}&y={y}"

                try:
                    proxy_url = f"http://{proxy['http']}" if proxy and "http" in proxy else None

                    async with session.get(
                        url_img, headers=ua, proxy=proxy_url, timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status == 200:
                            content = await response.read()
                            # Convert bytes to PIL Image
                            from io import BytesIO

                            image = Image.open(BytesIO(content))
                            return image
                        else:
                            if attempt == 2:  # Last attempt
                                raise Exception(
                                    f"Failed to fetch tile {x},{y} for pano {pano_id}: HTTP {response.status}"
                                )
                except Exception as e:
                    if attempt == 2:  # Last attempt
                        print(f"Proxy {proxy} failed for tile {x},{y} in pano {pano_id}. Exception: {e}")
                        raise
                    continue

            raise Exception(f"All attempts failed for tile {x},{y} in pano {pano_id}")

    @staticmethod
    def is_bottom_black(image, row_count=3, intensity_threshold=10):
        """Check if the bottom rows of an image are near black.

        Uses linear computation instead of nested loops for faster execution.

        Args:
            image (PIL.Image): The image to check.
            row_count (int, optional): Number of bottom rows to check. Defaults to 3.
            intensity_threshold (int, optional): Maximum pixel intensity to be considered black. Defaults to 10.

        Returns:
            bool: True if the bottom rows are near black, False otherwise.
        """
        # Convert the bottom rows to a numpy array for fast processing
        bottom_rows = np.array(image)[-row_count:, :]
        # Check if all pixels in the bottom rows are less than or equal to the intensity threshold
        return np.all(bottom_rows <= intensity_threshold)

    @staticmethod
    def process_image(image, zoom):
        """Process an image by cropping and resizing based on zoom level.

        Only processes the image if the bottom is black.

        Args:
            image (PIL.Image): The image to process.
            zoom (int): The zoom level used to determine dimensions.

        Returns:
            PIL.Image: The processed image, either cropped and resized or unchanged.
        """
        if ImageTool.is_bottom_black(image):
            # Compute the crop and resize dimensions based on zoom level
            crop_height, crop_width = 208 * (2**zoom), 416 * (2**zoom)
            resize_height, resize_width = 256 * (2**zoom), 512 * (2**zoom)

            # Crop the image
            image = image.crop((0, 0, crop_width, crop_height))

            # Resize the image
            image = image.resize((resize_width, resize_height), Image.LANCZOS)

        return image

    @staticmethod
    def get_and_save_image(
        pano_id,
        identif,
        zoom,
        vertical_tiles,
        horizontal_tiles,
        out_path,
        ua,
        proxies,
        cropped=False,
        full=True,
    ):
        """Download and compose a complete Street View image from individual tiles.

        Args:
            pano_id (str): Google Street View panorama ID.
            identif (str): Custom identifier for the saved image.
            zoom (int): Zoom level for image resolution.
            vertical_tiles (int): Number of vertical tiles to download.
            horizontal_tiles (int): Number of horizontal tiles to download.
            out_path (str): Directory path to save the images.
            ua (str): User agent string for HTTP requests.
            proxies (list): List of proxy servers to use.
            cropped (bool, optional): Whether to crop image horizontally in half. Defaults to False.
            full (bool, optional): Whether to save the complete composed image. Defaults to True.

        Returns:
            str: The identifier of the saved image.

        Raises:
            ValueError: If the final image dimensions are invalid.
        """
        for x in range(horizontal_tiles):
            for y in range(vertical_tiles):
                new_img = ImageTool.fetch_image_with_proxy(pano_id, zoom, x, y, ua, proxies)
                if not full:
                    new_img.save(f"{out_path}/{identif}_x{x}_y{y}.jpg")
                if y == 0:
                    first_slice = new_img
                else:
                    first_slice = ImageTool.concat_vertically(first_slice, new_img)

            if x == 0:
                final_image = first_slice
            else:
                final_image = ImageTool.concat_horizontally(final_image, first_slice)

        if full:
            name = f"{out_path}/{identif}"
            if cropped or zoom == 0:
                h_cropped = final_image.size[1] // 2
                final_image = final_image.crop((0, 0, final_image.size[0], h_cropped))

            # Validate image before saving
            if final_image.size[0] > 0 and final_image.size[1] > 0:
                final_image = ImageTool.process_image(final_image, zoom)
                final_image.save(f"{name}.jpg")
            else:
                raise ValueError(f"Invalid image for pano_id {pano_id}")

        return identif

    @staticmethod
    async def get_and_save_image_async(
        session,
        semaphore,
        pano_id,
        identif,
        zoom,
        vertical_tiles,
        horizontal_tiles,
        out_path,
        ua,
        proxies,
        cropped=False,
        full=True,
    ):
        """Async version of get_and_save_image.

        Args:
            session (aiohttp.ClientSession): The aiohttp session for requests.
            semaphore (asyncio.Semaphore): Semaphore for controlling concurrency.
            pano_id (str): Google Street View panorama ID.
            identif (str): Custom identifier for the saved image.
            zoom (int): Zoom level for image resolution.
            vertical_tiles (int): Number of vertical tiles to download.
            horizontal_tiles (int): Number of horizontal tiles to download.
            out_path (str): Directory path to save the images.
            ua (str): User agent string for HTTP requests.
            proxies (list): List of proxy servers to use.
            cropped (bool, optional): Whether to crop image horizontally in half. Defaults to False.
            full (bool, optional): Whether to save the complete composed image. Defaults to True.

        Returns:
            str: The identifier of the saved image.
        """
        # Use executor for CPU-intensive image processing
        loop = asyncio.get_event_loop()

        # Fetch all tiles asynchronously
        tile_tasks = []
        for x in range(horizontal_tiles):
            for y in range(vertical_tiles):
                task = ImageTool.fetch_image_with_proxy_async(session, semaphore, pano_id, zoom, x, y, ua, proxies)
                tile_tasks.append((x, y, task))

        # Download all tiles concurrently
        tiles = {}
        for x, y, task in tile_tasks:
            try:
                tile_image = await task
                tiles[(x, y)] = tile_image

                # Save individual tile if not full
                if not full:
                    await loop.run_in_executor(None, lambda: tile_image.save(f"{out_path}/{identif}_x{x}_y{y}.jpg"))
            except Exception as e:
                print(f"Failed to fetch tile ({x}, {y}) for pano {pano_id}: {e}")
                raise

        if full:
            # Compose final image using executor for CPU-intensive work
            final_image = await loop.run_in_executor(
                None, ImageTool._compose_image_from_tiles, tiles, horizontal_tiles, vertical_tiles
            )

            # Process and save final image
            name = f"{out_path}/{identif}"
            if cropped or zoom == 0:
                h_cropped = final_image.size[1] // 2
                final_image = final_image.crop((0, 0, final_image.size[0], h_cropped))

            # Validate image before saving
            if final_image.size[0] > 0 and final_image.size[1] > 0:
                final_image = ImageTool.process_image(final_image, zoom)
                await loop.run_in_executor(None, lambda: final_image.save(f"{name}.jpg"))
            else:
                raise ValueError(f"Invalid image for pano_id {pano_id}")

        return identif

    @staticmethod
    def _compose_image_from_tiles(tiles, horizontal_tiles, vertical_tiles):
        """Compose the final image from individual tiles (CPU-intensive, run in executor)."""
        for x in range(horizontal_tiles):
            for y in range(vertical_tiles):
                tile_image = tiles[(x, y)]
                if y == 0:
                    first_slice = tile_image
                else:
                    first_slice = ImageTool.concat_vertically(first_slice, tile_image)

            if x == 0:
                final_image = first_slice
            else:
                final_image = ImageTool.concat_horizontally(final_image, first_slice)

        return final_image

    @staticmethod
    def dwl_multiple(
        panoids,
        zoom,
        v_tiles,
        h_tiles,
        out_path,
        uas,
        proxies,
        cropped,
        full,
        batch_size=1000,
        logger=None,
        max_workers=None,
        verbosity=1,
    ):
        """Download multiple Street View images in parallel using batched processing.

        Args:
            panoids (list): List of Google Street View panorama IDs.
            zoom (int): Zoom level for image resolution.
            v_tiles (int): Number of vertical tiles per image.
            h_tiles (int): Number of horizontal tiles per image.
            out_path (str): Base directory to save images.
            uas (list): List of user agent strings.
            proxies (list): List of proxy servers to use.
            cropped (bool): Whether to crop images horizontally in half.
            full (bool): Whether to save complete composed images.
            batch_size (int, optional): Number of images to process per batch. Defaults to 1000.
            logger (Logger, optional): Logger object for recording progress. Defaults to None.
            max_workers (int, optional): Maximum number of concurrent download threads. Defaults to None.
            verbosity (int, optional): Level of verbosity for progress bars (0=no progress, 1=outer loops only, 2=all loops). Defaults to 1.

        Note:
            Creates subdirectories for each batch of downloads.
            Failed downloads are logged if a logger is provided.
        """
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        errors = 0

        # Calculate current highest batch number
        existing_batches = glob.glob(os.path.join(out_path, "batch_*"))
        existing_batch_numbers = [int(os.path.basename(batch).split("_")[-1]) for batch in existing_batches]
        start_batch_number = max(existing_batch_numbers, default=0)

        num_batches = (len(panoids) + batch_size - 1) // batch_size

        for i in verbosity_tqdm(
            range(num_batches),
            desc=f"Processing outer batches of size {min(batch_size, len(panoids))}",
            verbosity=verbosity,
            level=1,
        ):
            # Create a new sub-folder for each batch
            batch_out_path = os.path.join(out_path, f"batch_{start_batch_number + i + 1}")
            os.makedirs(batch_out_path, exist_ok=True)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                jobs = []
                batch_panoids = panoids[
                    start_batch_number * batch_size + i * batch_size : (start_batch_number + i + 1) * batch_size
                ]
                batch_uas = uas[
                    start_batch_number * batch_size + i * batch_size : (start_batch_number + i + 1) * batch_size
                ]
                for pano, ua in zip(batch_panoids, batch_uas):
                    kw = {
                        "pano_id": pano,
                        "identif": pano,
                        "ua": ua,
                        "proxies": proxies,
                        "zoom": zoom,
                        "vertical_tiles": v_tiles,
                        "horizontal_tiles": h_tiles,
                        "out_path": batch_out_path,  # Pass the new sub-folder path
                        "cropped": cropped,
                        "full": full,
                    }
                    jobs.append(executor.submit(ImageTool.get_and_save_image, **kw))

                for job in verbosity_tqdm(
                    as_completed(jobs),
                    total=len(jobs),
                    desc=f"Downloading images for batch #{start_batch_number + i + 1}",
                    verbosity=verbosity,
                    level=2,
                ):
                    try:
                        job.result()
                    except Exception as e:
                        print(e)
                        errors += 1
                        failed_panoid = batch_panoids[jobs.index(job)]
                        if logger:
                            logger.log_failed_pid(failed_panoid)

        print("Total images downloaded:", len(panoids) - errors, "Errors:", errors)
        if logger:
            logger.log_info(f"Total images downloaded: {len(panoids) - errors}, Errors: {errors}")

    @staticmethod
    async def dwl_multiple_async(
        panoids,
        zoom,
        v_tiles,
        h_tiles,
        out_path,
        uas,
        proxies,
        cropped,
        full,
        batch_size=1000,
        logger=None,
        max_concurrency=None,
        verbosity=1,
    ):
        """Async version of dwl_multiple for downloading multiple Street View images.

        Args:
            panoids (list): List of Google Street View panorama IDs.
            zoom (int): Zoom level for image resolution.
            v_tiles (int): Number of vertical tiles per image.
            h_tiles (int): Number of horizontal tiles per image.
            out_path (str): Base directory to save images.
            uas (list): List of user agent strings.
            proxies (list): List of proxy servers to use.
            cropped (bool): Whether to crop images horizontally in half.
            full (bool): Whether to save complete composed images.
            batch_size (int, optional): Number of images to process per batch. Defaults to 1000.
            logger (Logger, optional): Logger object for recording progress. Defaults to None.
            max_concurrency (int, optional): Maximum number of concurrent downloads. Defaults to None.
            verbosity (int, optional): Level of verbosity for progress bars. Defaults to 1.
        """
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        errors = 0
        if max_concurrency is None:
            max_concurrency = min(100, os.cpu_count() * 4)

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrency)

        # Calculate current highest batch number
        existing_batches = glob.glob(os.path.join(out_path, "batch_*"))
        existing_batch_numbers = [int(os.path.basename(batch).split("_")[-1]) for batch in existing_batches]
        start_batch_number = max(existing_batch_numbers, default=0)

        num_batches = (len(panoids) + batch_size - 1) // batch_size

        for i in verbosity_tqdm(
            range(num_batches),
            desc=f"Processing async batches of size {min(batch_size, len(panoids))}",
            verbosity=verbosity,
            level=1,
        ):
            # Create a new sub-folder for each batch
            batch_out_path = os.path.join(out_path, f"batch_{start_batch_number + i + 1}")
            os.makedirs(batch_out_path, exist_ok=True)

            batch_panoids = panoids[
                start_batch_number * batch_size + i * batch_size : (start_batch_number + i + 1) * batch_size
            ]
            batch_uas = uas[
                start_batch_number * batch_size + i * batch_size : (start_batch_number + i + 1) * batch_size
            ]

            async with aiohttp.ClientSession() as session:
                tasks = []
                for pano, ua in zip(batch_panoids, batch_uas):
                    task = ImageTool.get_and_save_image_async(
                        session=session,
                        semaphore=semaphore,
                        pano_id=pano,
                        identif=pano,
                        zoom=zoom,
                        vertical_tiles=v_tiles,
                        horizontal_tiles=h_tiles,
                        out_path=batch_out_path,
                        ua=ua,
                        proxies=proxies,
                        cropped=cropped,
                        full=full,
                    )
                    tasks.append((task, pano))

                # Process tasks with progress tracking
                completed = 0
                total_tasks = len(tasks)

                pbar = verbosity_tqdm(
                    range(total_tasks),
                    desc=f"Downloading images for async batch #{start_batch_number + i + 1}",
                    verbosity=verbosity,
                    level=2,
                )
                
                for coro in asyncio.as_completed([task for task, _ in tasks]):
                    try:
                        await coro
                        completed += 1
                        if hasattr(pbar, 'update'):
                            pbar.update(1)
                    except Exception as e:
                        errors += 1
                        # Find the failed panorama ID
                        failed_pano = None
                        for task, pano in tasks:
                            if task == coro:
                                failed_pano = pano
                                break

                        print(f"Error downloading pano {failed_pano}: {e}")
                        if logger:
                            logger.log_failed_pid(failed_pano)
                        if hasattr(pbar, 'update'):
                            pbar.update(1)

        print("Total images downloaded:", len(panoids) - errors, "Errors:", errors)
        if logger:
            logger.log_info(f"Total async images downloaded: {len(panoids) - errors}, Errors: {errors}")
