from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS

@DATASETS.register_module()
class TurtleDataset(BaseSegDataset):
    CLASSES = ('background', 'head', 'turtle', 'flipper')
    PALETTE = [[0, 0, 0],    # 背景-黑色
               [255, 0, 0],   # 头-红色
               [0, 0, 255],   # 身体-蓝色
               [0, 255, 0]]   # 鳍-绿色
    
    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.jpg',      # 或 '.JPG'
            seg_map_suffix='.png',  # 你的掩码文件后缀
            reduce_zero_label=True, # 如果背景标签是0
            **kwargs
        )