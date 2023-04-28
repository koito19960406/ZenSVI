import os
import sys
import requests
from PIL import Image
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
import numpy as np

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
    def get_and_save_image(pano_id, identif, zoom, vertical_tiles, horizontal_tiles, out_path, ua, cropped=False, full=True):
        """
        Description of get_and_save_image
        
        Downloads an image tile by tile and composes them together.

        Args:
            pano_id (undefined): GSV anorama id
            identif (undefined): custom identifier
            size (undefined):    image resolution
            vertical_tiles (undefined): number of vertical tiles
            horizontal_tiles (undefined): number of horizontal tiles
            out_path (undefined): output path
            cropped=False (undefined): set True if the image split horizontally in half is needed
            full=True (undefined): set to True if the full image is needed

        """

        first_url_img = f'https://cbk0.google.com/cbk?output=tile&panoid={pano_id}&zoom={zoom}&x={0}&y={0}'

        first = Image.open(requests.get(first_url_img, headers=ua, stream=True).raw)
        # first_vert = False

        for y in range(1, vertical_tiles):
            #new_img = Image.open(f'./images/test_x0_y{y}.png')
            url_new_img = f'https://cbk0.google.com/cbk?output=tile&panoid={pano_id}&zoom={zoom}&x={0}&y={y}'
            new_img = Image.open(requests.get(url_new_img, headers=ua, stream=True).raw)
            first = ImageTool.concat_vertically(first, new_img)
        first_slice = first

        for x in range(1, horizontal_tiles):

            first_url_img = f'https://cbk0.google.com/cbk?output=tile&panoid={pano_id}&zoom={zoom}&x={x}&y={0}'
            first = Image.open(requests.get(first_url_img, headers=ua, stream=True).raw)
            
            for y in range(1, vertical_tiles):
                #new_img = Image.open(f'./images/test_x{x}_y{y}.png')
                url_new_img = f'https://cbk0.google.com/cbk?output=tile&panoid={pano_id}&zoom={zoom}&x={x}&y={y}'
                new_img = Image.open(requests.get(url_new_img, headers=ua, stream=True).raw)
                first = ImageTool.concat_vertically(first, new_img)

            new_slice = first
            first_slice = ImageTool.concat_horizontally(first_slice, new_slice)

        # first_slice.thumbnail(size, Image.ANTIALIAS)
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
        # if cropped:
        #     first_slice.crop((0, 0, size[1], size[1])).save(f'{name}_p1.jpg')
        #     first_slice.crop((size[1], 0, size[0], size[1])).save(f'{name}_p2.jpg')
        
        return identif

    @staticmethod
    def dwl_multiple(panoids, identifiers, nthreads, zoom, v_tiles, h_tiles, out_path, uas, cropped=True, full=False):
        """
        Description of get_and_save_image
        
        Calls the get_and_save_image function using multiple threads.
        
        Args:
            panoids (undefined): GSV anorama id
            identifiers (undefined): custom identifier
            nthreads (undefined): number of threads
            zoom (undefined):    image resolution
            v_tiles (undefined): number of vertical tiles
            h_tiles (undefined): number of horizontal tiles
            out_path (undefined): output path
            cropped=False (undefined): set True if the image split horizontally in half is needed
            full=True (undefined): set to True if the full image is needed

        """
        
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            
        workers_range = nthreads

        with ThreadPoolExecutor(max_workers = workers_range) as executor:

            jobs = []
            results_done = []
            for pano, identif, ua in zip(panoids, identifiers, uas):

                kw = {
                    "pano_id" : pano, 
                    "identif" : identif,
                    "ua": ua,
                    "zoom" : zoom,
                    "vertical_tiles" : v_tiles, 
                    "horizontal_tiles" : h_tiles,
                    "out_path" : out_path,
                    "cropped" : cropped,
                    "full" : full
                }
                
                jobs.append(executor.submit(ImageTool.get_and_save_image, **kw))


            for job in futures.as_completed(jobs):
                result_done = job.result()
                results_done.append(result_done)


