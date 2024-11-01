# mask_generator.py
import os
import numpy as np
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

class MaskGenerator:
    def __init__(self, ann_file, img_dir, mask_dir):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.category_color_map = {
            "turtle": (0, 0, 255),  # 蓝色
            "flipper": (0, 255, 0), # 绿色
            "head": (255, 0, 0)     # 红色
        }
        os.makedirs(mask_dir, exist_ok=True)
        
    def generate_masks(self):
        plt.switch_backend('Agg')
        
        for image_id in self.coco.imgs.keys():
            self._process_single_image(image_id)
            
    def _process_single_image(self, image_id):
        img = self.coco.imgs[image_id]
        image_file = img['file_name']
        image_path = os.path.join(self.img_dir, image_file)
        
        if not os.path.exists(image_path):
            print(f"not found: {image_path}")
            return
            
        cat_ids = self.coco.getCatIds()
        anns_ids = self.coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = self.coco.loadAnns(anns_ids)

        mask = np.zeros((img['height'], img['width'], 3), dtype=np.uint8)
        
        for ann in anns:
            category_name = self.coco.loadCats(ann['category_id'])[0]['name']
            color_value = self.category_color_map.get(category_name, (0, 0, 0))
            mask += np.dstack([self.coco.annToMask(ann) * color for color in color_value])
            
        self._save_mask(mask, image_file)
        
    def _save_mask(self, mask, image_file):
        masked_image_path = os.path.join(self.mask_dir, f"colored_mask_{image_file}")
        os.makedirs(os.path.dirname(masked_image_path), exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axis('off')
        ax.imshow(mask, cmap='nipy_spectral')
        fig.savefig(masked_image_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"saving: {masked_image_path}")

if __name__ == "__main__":
    generator = MaskGenerator(
        ann_file='data/annotations.json',
        img_dir='data/',
        mask_dir='data/mask'
    )
    generator.generate_masks()