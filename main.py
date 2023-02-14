import numpy as np
import cv2
import tqdm
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", help="path to the image", required=True)
    parser.add_argument("--patch_size", default="15", help="patch size")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    img = cv2.imread(args.image_path, 0)

    patch_size = int(args.patch_size)
    img_wave = np.ones(img.shape) * 255

    e = 0.0000000001
    for i in tqdm.tqdm(range(0, img.shape[0]-patch_size, patch_size)):
        for j in range(0, img.shape[1]-patch_size, patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]

            blackness = np.sum(patch) / patch_size**2
            frequency = (2*np.pi)/patch_size * np.log(np.sqrt(blackness)+e)
            amplitude = (1*patch_size) * (1-blackness/255)

            x = np.arange(0, patch_size, 1)
            y = amplitude * np.sin(frequency * x) - patch_size

            x += j
            y = np.int32(np.abs(y/2)) + i

            for k in range(x.shape[0]-1):
                cv2.line(img_wave, (x[k], y[k]), (x[k+1], y[k+1]), 0, 1)

    extension = '.'+args.image_path.split(".")[-1]
    img_name = os.path.basename(args.image_path)
    save_path = os.path.join(os.path.dirname(args.image_path), img_name.replace(extension, "_waves.png"))
    cv2.imwrite(save_path, img_wave)