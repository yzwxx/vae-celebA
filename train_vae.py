import os
import sys
import scipy.misc
import pprint
import numpy as np
import time
import math
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from glob import glob
from random import shuffle
from model_vae import *
from utils import *


pp = pprint.PrettyPrinter()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

'''
Tensorlayer implementation of VAE
'''

flags = tf.app.flags
flags.DEFINE_integer("epoch", 30, "Epoch to train [5]") 
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for adam [0.001]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The number of batch images [64]")
flags.DEFINE_integer("image_size", 148, "The size of image to use (will be center cropped) [108]")
# flags.DEFINE_integer("decoder_output_size", 64, "The size of the output images to produce from decoder[64]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("sample_size", 64, "The number of sample images [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("z_dim", 128, "Dimension of latent representation vector from. [2048]")
flags.DEFINE_integer("sample_step", 300, "The interval of generating sample. [300]")
flags.DEFINE_integer("save_step", 800, "The interval of saveing checkpoints. [500]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA]")
flags.DEFINE_string("test_number", "vae_0808", "The number of experiment [test2]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [False]")
# flags.DEFINE_integer("class_dim", 4, "class number for auxiliary classifier [5]") 
#flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean("load_pretrain",False, "Default to False;If start training on a pretrained net, choose True")
FLAGS = flags.FLAGS



def main(_):
    pp.pprint(FLAGS.__flags)

    tl.files.exists_or_mkdir(FLAGS.checkpoint_dir)
    tl.files.exists_or_mkdir(FLAGS.sample_dir)

    with tf.device("/gpu:0"):
        ##========================= DEFINE MODEL ===========================##
        # the input_imgs are input for both encoder and discriminator
        input_imgs = tf.placeholder(tf.float32,[FLAGS.batch_size, FLAGS.output_size, 
            FLAGS.output_size, FLAGS.c_dim], name='real_images')

        # normal distribution for GAN
        z_p = tf.random_normal(shape=(FLAGS.batch_size, FLAGS.z_dim), mean=0.0, stddev=1.0)
        # normal distribution for reparameterization trick
        eps = tf.random_normal(shape=(FLAGS.batch_size, FLAGS.z_dim), mean=0.0, stddev=1.0)
        lr_vae = tf.placeholder(tf.float32, shape=[])


        # ----------------------encoder----------------------
        net_out1, net_out2, z_mean, z_log_sigma_sq = encoder(input_imgs, is_train=True, reuse=False)

        # ----------------------decoder----------------------
        # decode z 
        # z = z_mean + z_sigma * eps
        z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps)) # using reparameterization tricks
        gen0, _ = generator(z, is_train=True, reuse=False)

        # ----------------------for samples----------------------
        gen2, gen2_logits = generator(z, is_train=False, reuse=True)
        gen3, gen3_logits = generator(z_p, is_train=False, reuse=True)

        ##========================= DEFINE TRAIN OPS =======================##
        ''''
        reconstruction loss:
        use the pixel-wise mean square error in image space
        '''
        SSE_loss = tf.reduce_mean(tf.square(gen0.outputs - input_imgs))# /FLAGS.output_size/FLAGS.output_size/3
        '''
        KL divergence:
        we get z_mean,z_log_sigma_sq from encoder, then we get z from N(z_mean,z_sigma^2)
        then compute KL divergence between z and standard normal gaussian N(0,I) 
        '''
        KL_loss = tf.reduce_mean(- 0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq),1))

        ### important points! ###
        # the weight between style loss(KLD) and contend loss(pixel-wise mean square error)
        VAE_loss = 0.005*KL_loss + SSE_loss # KL_loss isn't working well if the weight of SSE is too big

        e_vars = tl.layers.get_variables_with_name('encoder',True,True)
        g_vars = tl.layers.get_variables_with_name('generator', True, True)
        # d_vars = tl.layers.get_variables_with_name('discriminator', True, True)
        vae_vars = e_vars+g_vars

        print("-------encoder-------")
        net_out1.print_params(False)
        print("-------generator-------")
        gen0.print_params(False)


        # optimizers for updating encoder, discriminator and generator
        vae_optim = tf.train.AdamOptimizer(lr_vae, beta1=FLAGS.beta1) \
                           .minimize(VAE_loss, var_list=vae_vars)
    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)

    # prepare file under checkpoint_dir
    model_dir = "vae_0808"
    #  there can be many models under one checkpoine file
    save_dir = os.path.join(FLAGS.checkpoint_dir, model_dir) #'./checkpoint/vae_0808'
    tl.files.exists_or_mkdir(save_dir)
    # under current directory
    samples_1 = FLAGS.sample_dir + "/" + FLAGS.test_number
    # samples_1 = FLAGS.sample_dir + "/test2"
    tl.files.exists_or_mkdir(samples_1) 

    if FLAGS.load_pretrain == True:
        load_e_params = tl.files.load_npz(path=save_dir,name='/net_e.npz')
        tl.files.assign_params(sess, load_e_params[:24], net_out1)
        net_out1.print_params(True)
        tl.files.assign_params(sess, np.concatenate((load_e_params[:24], load_e_params[30:]), axis=0), net_out2)
        net_out2.print_params(True)

        load_g_params = tl.files.load_npz(path=save_dir,name='/net_g.npz')
        tl.files.assign_params(sess, load_g_params, gen0)
        gen0.print_params(True)
    
    # get the list of absolute paths of all images in dataset
    data_files = glob(os.path.join("./data", FLAGS.dataset, "*.jpg"))
    data_files = sorted(data_files)
    data_files = np.array(data_files) # for tl.iterate.minibatches


    ##========================= TRAIN MODELS ================================##
    iter_counter = 0

    training_start_time = time.time()
    # use all images in dataset in every epoch
    for epoch in range(FLAGS.epoch):
        ## shuffle data
        print("[*] Dataset shuffled!")

        minibatch = tl.iterate.minibatches(inputs=data_files, targets=data_files, batch_size=FLAGS.batch_size, shuffle=True)
        idx = 0
        batch_idxs = min(len(data_files), FLAGS.train_size) // FLAGS.batch_size

        while True:
            try:
                batch_files,_ = minibatch.next()
                batch = [get_image(batch_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale = 0) \
                        for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                start_time = time.time()
                vae_current_lr = FLAGS.learning_rate


                # update
                kl, sse, errE, _ = sess.run([KL_loss,SSE_loss,VAE_loss,vae_optim], feed_dict={input_imgs: batch_images, lr_vae:vae_current_lr})


                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, vae_loss:%.8f, kl_loss:%.8f, sse_loss:%.8f" \
                        % (epoch, FLAGS.epoch, idx, batch_idxs,
                            time.time() - start_time, errE, kl, sse))
                sys.stdout.flush()

                iter_counter += 1
                # save samples
                if np.mod(iter_counter, FLAGS.sample_step) == 0:
                    # generate and visualize generated images
                    img1, img2 = sess.run([gen2.outputs, gen3.outputs], feed_dict={input_imgs: batch_images})
                    save_images(img1, [8, 8],
                                './{}/train_{:02d}_{:04d}.png'.format(samples_1, epoch, idx))

                    # img2 = sess.run(gen3.outputs, feed_dict={input_imgs: batch_images})
                    save_images(img2, [8, 8],
                                './{}/train_{:02d}_{:04d}_random.png'.format(samples_1, epoch, idx))

                    # save input image for comparison
                    save_images(batch_images,[8, 8],'./{}/input.png'.format(samples_1))
                    print("[Sample] sample generated!!!")
                    sys.stdout.flush()

                # save checkpoint
                if np.mod(iter_counter, FLAGS.save_step) == 0:
                    # save current network parameters
                    print("[*] Saving checkpoints...")
                    net_e_name = os.path.join(save_dir, 'net_e.npz')
                    net_g_name = os.path.join(save_dir, 'net_g.npz')
                    # this version is for future re-check and visualization analysis
                    net_e_iter_name = os.path.join(save_dir, 'net_e_%d.npz' % iter_counter)
                    net_g_iter_name = os.path.join(save_dir, 'net_g_%d.npz' % iter_counter)


                    # params of two branches
                    net_out_params = net_out1.all_params + net_out2.all_params
                    # remove repeat params
                    net_out_params = tl.layers.list_remove_repeat(net_out_params)
                    tl.files.save_npz(net_out_params, name=net_e_name, sess=sess)
                    tl.files.save_npz(gen0.all_params, name=net_g_name, sess=sess)

                    tl.files.save_npz(net_out_params, name=net_e_iter_name, sess=sess)
                    tl.files.save_npz(gen0.all_params, name=net_g_iter_name, sess=sess)

                    print("[*] Saving checkpoints SUCCESS!")

                idx += 1
                # print idx
            except StopIteration:
                print 'one epoch finished'
                break
            except Exception as e:
                raise e
            


    training_end_time = time.time()
    print("The processing time of program is : {:.2f}mins".format((training_end_time-training_start_time)/60.0))


if __name__ == '__main__':
    tf.app.run()

