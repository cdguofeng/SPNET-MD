import os
from glob import glob
from typing import Optional
import time
import cv2
import numpy as np
import torch
import yaml
from fire import Fire
from tqdm import tqdm
from PIL import Image
from aug import get_normalize
from models.networks import get_generator


class Predictor:
    def __init__(self, weights_path: str, model_name: str = ''):
        with open('config/config.yaml') as cfg:
            config = yaml.load(cfg)
        model = get_generator(model_name or config['model'])
        model.load_state_dict(torch.load(weights_path)['model'])
        self.model = model.cuda()
        self.model.train(True)
        # GAN inference should be in train mode to use actual stats in norm layers,
        # it's not a bug
        self.normalize_fn = get_normalize()

    @staticmethod
    def _array_to_batch(x):
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x)

    def _preprocess(self, x: np.ndarray, mask: Optional[np.ndarray]):
        x, _ = self.normalize_fn(x, x)
        if mask is None:
            mask = np.ones_like(x, dtype=np.float32)
        else:
            mask = np.round(mask.astype('float32') / 255)

        h, w, _ = x.shape
        block_size = 32
        min_height = (h // block_size + 1) * block_size
        min_width = (w // block_size + 1) * block_size

        pad_params = {'mode': 'constant',
                      'constant_values': 0,
                      'pad_width': ((0, min_height - h), (0, min_width - w), (0, 0))
                      }
        x = np.pad(x, **pad_params)
        mask = np.pad(mask, **pad_params)

        return map(self._array_to_batch, (x, mask)), h, w

    @staticmethod
    def _postprocess(x: torch.Tensor) -> np.ndarray:
        x, = x
        x = x.detach().cpu().float().numpy()
        x = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 * 255.0
        return x.astype('uint8')

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray], ignore_mask=True) -> np.ndarray:
        (img, mask), h, w = self._preprocess(img, mask)
        with torch.no_grad():
            inputs = [img.cuda()]
            if not ignore_mask:
                inputs += [mask]
            pred = self.model(*inputs)
        return self._postprocess(pred)[:h, :w, :]

def main(img_pattern: str,
         mask_pattern: Optional[str] = None,
         weights_path='last_2023-2-17.h5',
         out_dir='submit/'):
    #start = time.time()
    predictor = Predictor(weights_path=weights_path)

    os.makedirs(out_dir, exist_ok=True)

    start=time.time()
    img_file = "F:/GF/dataset4/train/male/deblur/"
    k=0
    for imgpath in glob.glob("F:/GF/dataset4/train/male/blured_rgb/*.jpg"):
        img= cv2.imread(imgpath)
        LaplacePic = cv2.Sobel(img, cv2.CV_8U, 1, 1, ksize=3)
        add = np.sum(LaplacePic)
        mean, std = cv2.meanStdDev(LaplacePic)
        if mean[0] <1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            a = os.path.basename(imgpath)
            #print(a.split(".")[0])
            b = a.split(".")[0]
            pred = predictor(img,mask=None)
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_file + "/" + b + '.jpg', pred)
            k=k+1
            #cv2.imwrite('./submit/model/62-2-17.jpg',pred)
            #cv2.imwrite('F:/GF/manuscript/deblur/deblurring/v2/1437-b2-v2-2.jpg', pred)
           # end=time.time()
            # print("time ",end-start)
            # cv2.imshow("model",pred)
            # cv2.waitKey()
        else:
            print(mean[0])
    print(k)
if __name__ == '__main__':
    #imgpath = './19.jpg'
    main()
