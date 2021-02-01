# Helpers to overcome differences between fastai v2 and v1 apis
from fastai.basic_data import DataBunch
from fastai.vision import NormType, SplitFuncOrIdxList, nn
from typing import Callable, Optional, Tuple, Union, Any

from fastai.vision.learner import cnn_config, create_body, to_device, apply_init
from fastai import distributed
from fastai.core import ifnone
def unet_learner_distributed(data:DataBunch, arch:Callable, pretrained:bool=True, blur_final:bool=True,
                 norm_type:Optional[NormType]=None, split_on:Optional[SplitFuncOrIdxList]=None, blur:bool=False,
                 self_attention:bool=False, y_range:Optional[Tuple[float,float]]=None, last_cross:bool=True,
                 bottle:bool=False, cut:Union[int,Callable]=None, **learn_kwargs:Any)->distributed.Learner:
    """
    Build Unet learner from `data` and `arch` for Data Distributed Parallel (DDP) training
    
    Modified from fastai.vision.learner to use distributed.Learner instead.
    """
    
    meta = cnn_config(arch)
    body = create_body(arch, pretrained, cut)
    try:    size = data.train_ds[0][0].size
    except: size = next(iter(data.train_dl))[0].shape[-2:]
    model = to_device(models.unet.DynamicUnet(body, n_classes=data.c, img_size=size, blur=blur, blur_final=blur_final,
          self_attention=self_attention, y_range=y_range, norm_type=norm_type, last_cross=last_cross,
          bottle=bottle), data.device)
    learn = distributed.Learner(data, model, **learn_kwargs)
    learn.split(ifnone(split_on, meta['split']))

    if pretrained: learn.freeze()
    apply_init(model[2], nn.init.kaiming_normal_)

    # distributed.Learner has extra .to_distributed() method, but IS NOT distributed yet
    # We can notice this in the worker stdout logs.  if we don't call .to_distributed(),
    # both worker and master will show loss values, etc, per epoch.  If we call this here, 
    # the detailed epoch statements are only written on the master logs
    # WARNING: Uses hard coded device (eg: expects config of 1 GPU per node)
    print("Converting Learner to distributed Learner", flush=True)
    learn = learn.to_distributed(0)

    return learn

