from random import shuffle
import scipy.misc
import numpy as np

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    # crop the images to [crop_h,crop_w,3] then resize to [resize_h,resize_w,3]
    if crop_w is None:
        crop_w = crop_h # the width and height after cropped
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w]) 

def merge(images, size):
    # merge all output images(of sample size:8*8 output images of size 64*64) into one big image
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images): # idx=0,1,2,...,63
        i = idx % size[1] # column number
        j = idx // size[1] # row number
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def transform(image, npx=64, is_crop=True, resize_w=64):
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.  # change pixel value range from [0,255] to [-1,1] to feed into CNN

def inverse_transform(images):
    return (images+1.)/2. # change image pixel value(outputs from tanh in range [-1,1]) back to [0,1]

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float) # [width,height] flatten RGB image to grayscale image
    else:
        return scipy.misc.imread(path).astype(np.float) # [width,height,color_dim]

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    # size indicates how to arrange the images to form a big summary image
    # images: [batchsize,height,width,color]
    # example: save_images(img, [8, 8],'./{}/train_{:02d}_{:04d}.png'.format(FLAGS.sample_dir, epoch, idx))
    return imsave(inverse_transform(images), size, image_path)

def save_images_256(images, size, image_path):
    images = inverse_transform(images)
    h, w = 64, 64 # 256,256
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images): # idx=0,1,2,...,63
        image = scipy.misc.imresize(image,[h,w])
        i = idx % size[1] # column number
        j = idx // size[1] # row number
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return scipy.misc.imsave(image_path, img)
