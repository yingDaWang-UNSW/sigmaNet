import numpy as np
import os
import glob
from PIL import Image
from tqdm import tqdm
import argparse
import h5py
import scipy as sp
'''

IMAGE CONVERSION ROUTINE

'''
parser = argparse.ArgumentParser(description='SR/GAN3D')
parser.add_argument('--indir', type=str, default='./DRSRD3/DRSRD3_3D/shuffled3D', help='path to DIV2K images')

parser.add_argument('--outdir', type=str, default='./shuffled3D_BIN', help='directory where converted image files are stored')

parser.add_argument('--inext', type=str, default='jpg', help='Input image file type')

parser.add_argument('--bitdepth', type=str, default='uint8', help='Input image file type')
args = parser.parse_args()

print('Running image conversion on input images')

''' start converter '''

input_path=args.indir
output_path=args.outdir
extension=args.inext

# generate the read paths
img_paths=[]
img_paths_ext = glob.glob(os.path.join(input_path, '**', f'*.{extension}'), recursive=False)
img_paths.extend(img_paths_ext)
preProcessingBitDepth=args.bitdepth
# convert and save
for img_path in tqdm(img_paths):
    img_dir, img_file = os.path.split(img_path)
    img_id, img_ext = os.path.splitext(img_file)

    rel_dir = os.path.relpath(img_dir, input_path)
    out_dir = os.path.join(output_path, rel_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if extension=='png' or  extension=='jpg':
        if preProcessingBitDepth == 'uint8':
            img = Image.open(img_path)
            if img.mode != 'RGB': #makes it triple channel
                img = img.convert('RGB')
            img = np.array(img, dtype='uint8')
            '''elif preProcessingBitDepth == 'uint16':
            reader = png.Reader(img_path) #input must be already triple channel, otherwise will save as single channel 16bit
            data = reader.asDirect()
            pixels = data[2]
            img = []
            for row in pixels:
                row = np.asarray(row)
                row = np.reshape(row, [-1, 3])
                img.append(row)
            img = np.stack(img, 1)
            img = np.rot90(img,-1)
            img = np.fliplr(img)'''
    
    elif extension=='mat':
        #img=octave.load(img_path)
        #img=img.temp
#                img=sio.loadmat(img_path)
#                img=img['temp']
        arrays = {}
        f = h5py.File(img_path)
        for k, v in f.items():
            arrays[k] = np.array(v)
        img=arrays['temp']
    arr_path = os.path.join(out_dir, f'{img_id}.npy')
    np.save(arr_path, np.array(img))
print('Done')
