import os
import aiohttp
from PIL import Image
import numpy as np
from io import BytesIO
from tqdm import tqdm
import asyncio
import io

class ImageTool():
    @staticmethod
    def concat_horizontally(im1, im2):
        """
        Description of concat_horizontally
        Horizontally concatenates two images

        Args:
            im1 (undefined): first PIL image
            im2 (undefined): second PIL image

        """
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    @staticmethod
    def concat_vertically(im1, im2):
        """
        Description of concat_vertically
        Vertically concatenates two images

        Args:
            im1 (undefined): first PIL image
            im2 (undefined): second PIL image

        """
        dst = Image.new('RGB', (im1.width, im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        return dst
    
    @staticmethod
    def compose_folder(in_path, out_path, how='horizontally'):
        """
        Description of compose_folder
        concatenates all the images in a folder; the second part of each 
        image must follow the first once sorted by filename.

        Args:
            in_path (undefined): input folder path
            out_path (undefined): output folder path
            how='horizontally' (undefined): concatenation direction

        """
        images = sorted(os.listdir(in_path))
            
        for name1, name2 in zip(images[0::2], images[1::2]):
            
            im1 = Image.open(in_path + name1)
            im2 = Image.open(in_path + name2)
            
            if how == 'horizontally':
                ImageTool.concat_horizontally(im1, im2).save(out_path + '_'.join(name1.split('_')[1:]))
            else:
                ImageTool.concat_vertically(im1, im2).save(out_path + '_'.join(name1.split('_')[1:]))

    @staticmethod
    async def get_and_save_image(session, sem, pano_id, identif, zoom, vertical_tiles, horizontal_tiles, out_path, ua, cropped=False, full=True):
        async def get_image(url):
            async with sem, session.get(url, headers=ua) as response:
                image_data = await response.read()
                return Image.open(io.BytesIO(image_data))


        first_url_img = f'https://cbk0.google.com/cbk?output=tile&panoid={pano_id}&zoom={zoom}&x={0}&y={0}'

        first = await get_image(first_url_img)

        for y in range(1, vertical_tiles):
            url_new_img = f'https://cbk0.google.com/cbk?output=tile&panoid={pano_id}&zoom={zoom}&x={0}&y={y}'
            new_img = await get_image(url_new_img)
            first = ImageTool.concat_vertically(first, new_img)

        first_slice = first

        for x in range(1, horizontal_tiles):
            first_url_img = f'https://cbk0.google.com/cbk?output=tile&panoid={pano_id}&zoom={zoom}&x={x}&y={0}'
            first = await get_image(first_url_img)
            
            for y in range(1, vertical_tiles):
                url_new_img = f'https://cbk0.google.com/cbk?output=tile&panoid={pano_id}&zoom={zoom}&x={x}&y={y}'
                new_img = await get_image(url_new_img)
                first = ImageTool.concat_vertically(first, new_img)

            first_slice = ImageTool.concat_horizontally(first_slice, first)

        name = f'{out_path}/{identif}'
        if full:
            image = np.array(first_slice)
            sun_i = sum(image[-5, :, 1])
            h, w, c = image.shape
            h_c = int(h * 0.812)
            w_c = int(w * 0.812)
            if sun_i == 0:
                pre_image = image[0:h_c, 0:w_c]
            else:
                pre_image = image
            pillow_image = Image.fromarray(pre_image)
            pillow_image.save(f'{name}.jpg')
        
        return identif

    @staticmethod
    async def dwl_multiple(panoids, zoom, v_tiles, h_tiles, out_path, uas, cropped=True, full=False, log_path=None):
        """
        Description of dwl_multiple

        Calls the get_and_save_image function using multiple tasks.

        Args:
            panoids (undefined): GSV anorama id
            zoom (undefined):    image resolution
            v_tiles (undefined): number of vertical tiles
            h_tiles (undefined): number of horizontal tiles
            out_path (undefined): output path
            cropped=False (undefined): set True if the image split horizontally in half is needed
            full=True (undefined): set to True if the full image is needed
            log_path=None (undefined): path to a log file
        """

        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        def log_error(panoid):
            nonlocal log_path
            if log_path is not None:
                with open(log_path, 'a') as log_file:
                    log_file.write(f"{panoid}\n")

        errors = 0
        sem = asyncio.Semaphore(100)  # Limit the number of simultaneous connections

        async with aiohttp.ClientSession() as session:
            tasks = []
            for pano, ua in zip(panoids, uas):
                kw = {
                    "session": session,
                    "sem": sem,
                    "pano_id": pano,
                    "identif": pano,
                    "ua": {"User-Agent": ua},
                    "zoom": zoom,
                    "vertical_tiles": v_tiles,
                    "horizontal_tiles": h_tiles,
                    "out_path": out_path,
                    "cropped": cropped,
                    "full": full
                }
                tasks.append(asyncio.create_task(ImageTool.get_and_save_image(**kw)))

            for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Downloading images"):
                try:
                    await task
                except Exception as e:
                    print(e)
                    errors += 1
                    failed_panoid = panoids[tasks.index(task)]
                    log_error(failed_panoid)

        print("Total images downloaded:", len(tasks) - errors, "Errors:", errors)
