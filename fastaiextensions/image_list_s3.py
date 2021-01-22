# modified from https://forums.fast.ai/t/my-s3-imagelist-version/75923
from fastai.core import PathOrStr
from fastai.vision import ImageList, Image
from typing import Callable
from s3ls import s3ls
import warnings
import PIL
from io import BytesIO
from fastai.vision.image import pil2tensor
import numpy as np

# DEBUG: Trying this to fix a broken pipe issue, but not sure it will help.
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True


# # No include/exclude but possible to add a custom filtering function
# def get_s3_files(s3_client, bucket, extensions=None, prefix=""):
#     res = []
#     for object_ in s3_resource.Bucket(bucket).objects.all():
#         if os.path.splitext(object_.key)[1] in extensions:
#             res.append((bucket,object_.key))
#     if filter_func: 
#         res = [i for i in res if filter_func(i)]
#     return res

# Extend the feature by inheritance
# We extend it by ovewrite a few methods
class ImageListS3(ImageList):
    def __init__(self, *args, bucket=None, **kwargs):
        """
        bucket: Bucket resource, eg from boto3.resource(...).Bucket(bucketname)
        """
        super().__init__(*args,**kwargs)
        if bucket is None:
            raise ValueError("Required argument 'bucket' not defined")
            
        self.bucket = bucket
        self.copy_new.append('bucket')  # ?
        
    # New class method that allows you to create an ImageList out of s3 files
    @classmethod  
    def from_s3_files(cls, bucket=None, extensions=['.jpg','.png'], prefix=None, recursive=False, list_dirs=False, **kwargs):
        file_list = s3ls(bucket, path=prefix, recursive=recursive, list_dirs=list_dirs)  # Need to add filtering into this api
        print(file_list)
        return cls(file_list, bucket=bucket, path='', **kwargs)  # what is path?
    
    # Overwrites original open() method
    # Just calls the method self.open_image() rather than the function open_image()
    # Otherwise it call the function open_image() of the library, probably du to scope logics
    # So that's just a workaround, could use the default if it mapped directly to our own open_image()
    def open(self,s3_fn):
        return self.open_image(s3_fn, convert_mode=self.convert_mode, after_open=self.after_open)
    
    # Our own open_image() as a method, rather than the original function
    def open_image(self, fn:PathOrStr, div:bool=True, convert_mode:str='RGB', cls:type=Image,
        after_open:Callable=None)->Image:
        "Return `Image` object created from image in file `fn`."
        return open_image_from_s3(bucket=self.bucket, fn=fn, div=div, convert_mode=convert_mode, cls=cls)
    
def open_image_from_s3(bucket, fn:PathOrStr, div:bool=True, convert_mode:str='RGB', cls:type=Image,
    after_open:Callable=None)->Image:
    "Return `Image` object created from image in file `fn`."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning) # EXIF warning from TiffPlugin
        obj = bucket.Object(str(fn))

#         file_byte_stream = BytesIO(obj.get()['Body'].read())
        file_byte_stream = BytesIO()
        obj.download_fileobj(file_byte_stream)

        x = PIL.Image.open(file_byte_stream)
        x = x.convert(convert_mode)
    if after_open: 
        x = after_open(x)
    x = pil2tensor(x, np.float32)
    if div: 
        x.div_(255)
    return cls(x)
