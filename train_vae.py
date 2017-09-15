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

# def balance(x,shift,mult):
#     """
#     Using this sigmoid to discourage one network overpowering the other
#     """
#     # return 1.0 
#     return 1.0 / (1 + math.exp(-(x+shift)*mult))

def main(_):
    pp.pprint(FLAGS.__flags)

    # prepare for the file directory
    # if not os.path.exists(FLAGS.checkpoint_dir):
    #     os.makedirs(FLAGS.checkpoint_dir)
    # if not os.path.exists(FLAGS.sample_dir):
    #     os.makedirs(FLAGS.sample_dir)
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
        # learning rates for e,g,d
        # lr_d = tf.placeholder(tf.float32, shape=[])
        # lr_g = tf.placeholder(tf.float32, shape=[])
        # lr_e = tf.placeholder(tf.float32, shape=[])
        lr_vae = tf.placeholder(tf.float32, shape=[])


        # ----------------------encoder----------------------
        net_out1, net_out2, z_mean, z_log_sigma_sq = encoder(input_imgs, is_train=True, reuse=False)

        # ----------------------decoder----------------------
        # decode z 
        # z = z_mean + z_sigma * eps
        z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps)) # using reparameterization tricks
        gen0, _ = generator(z, is_train=True, reuse=False)
        # as akara suggests (not working)
        # _, _, z_mean_fake, z_log_sigma_sq_fake = encoder(gen0.outputs, is_train=True, reuse=True)
        # z_fake = tf.add(z_mean_fake, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq_fake)), eps))
        # decode z_p
        # gen1, _ = generator(z_p, is_train=True, reuse=True)

        # ----------------------discriminator----------------------
        # for real images
        # dis, dis_real_logits, hidden_output_real = discriminator(input_imgs, is_train=True, reuse=False)
        # dis, dis_real_logits = discriminator(input_imgs, is_train=True, reuse=False)
        # for fake images decoded from z
        # _, dis_fake_z_logits, hidden_output_z = discriminator(gen0.outputs, is_train=True, reuse=True)
        # _, dis_fake_z_logits = discriminator(gen0.outputs, is_train=True, reuse=True)
        # for fake images decoded from z_p
        # _, dis_fake_z_p_logits, hidden_output_z_p = discriminator(gen1.outputs, is_train=True, reuse=True)

        # ----------------------for samples----------------------
        gen2, gen2_logits = generator(z, is_train=False, reuse=True)
        gen3, gen3_logits = generator(z_p, is_train=False, reuse=True)

        ##========================= DEFINE TRAIN OPS =======================##
        # GAN loss
        # d loss real with hard labels
        # d_loss_real = tl.cost.sigmoid_cross_entropy(dis_real_logits, 
        #                                                 tf.ones_like(dis_real_logits), name='d_real')

        # smooth label for d loss real
        # smoothing_mask_real = tf.truncated_normal(dis_real_logits.get_shape(), mean= 0.8, stddev=0.1)
        # d_loss_real = tl.cost.sigmoid_cross_entropy(dis_real_logits, 
        #                                     tf.ones_like(dis_real_logits) * smoothing_mask_real, name='d_real')


        # # smoothing only on positive samples(improved-gan-techniques)
        # d_loss_fake_z = tl.cost.sigmoid_cross_entropy(dis_fake_z_logits, 
        #                                     tf.zeros_like(dis_fake_z_logits), name='d_fake_z')
        # d_loss_fake_z_p = tl.cost.sigmoid_cross_entropy(dis_fake_z_p_logits,
        #                                     tf.zeros_like(dis_fake_z_p_logits), name='d_fake_z_p')

        # # hard labels for g loss
        # g_loss_fake_z = tl.cost.sigmoid_cross_entropy(dis_fake_z_logits, 
        #                                     tf.ones_like(dis_fake_z_logits), name='g_fake_z')
        # g_loss_fake_z_p = tl.cost.sigmoid_cross_entropy(dis_fake_z_p_logits, 
        #                                     tf.ones_like(dis_fake_z_p_logits), name='g_fake_z_p')

        # smooth labels for g loss
        # smoothing_mask_fake_g1 = tf.truncated_normal(dis_fake_z_logits.get_shape(), mean= 0.8, stddev=0.1)
        # g_loss_fake_z = tl.cost.sigmoid_cross_entropy(dis_fake_z_logits, 
        #                                     tf.ones_like(dis_fake_z_logits)*smoothing_mask_fake_g1, name='g_fake_z')
        # smoothing_mask_fake_g2 = tf.truncated_normal(dis_fake_z_logits.get_shape(), mean= 0.8, stddev=0.1)
        # g_loss_fake_z_p = tl.cost.sigmoid_cross_entropy(dis_fake_z_p_logits, 
        #                                     tf.ones_like(dis_fake_z_p_logits)*smoothing_mask_fake_g2, name='g_fake_z_p')

        # this maybe wrong
        # gan_d_loss = d_loss_real + d_loss_fake_z_p #+ d_loss_fake_z
        # gan_g_loss =  g_loss_fake_z_p #+ g_loss_fake_z  

        # as akara suggests
        # gan_d_loss = d_loss_real + d_loss_fake_z
        # gan_g_loss =   g_loss_fake_z 
        
        # # same as paper 
        # gan_d_loss = d_loss_real + 0.5*(d_loss_fake_z + d_loss_fake_z_p)
        # # gan_g_loss =  g_loss_fake_z #+ g_loss_fake_z_p
        # gan_g_loss =  0.5*(g_loss_fake_z + g_loss_fake_z_p)
        ''''
        reconstruction loss:
        use the learned similarity measurement in l-th layer of discriminator
        '''
        # l_layer_loss = tf.reduce_mean(tf.square(hidden_output_z - hidden_output_real)) 

        # hidden_loss = tf.reduce_mean(tf.square(z - z_fake))
        # SSE_loss = tf.reduce_mean(tf.square(gen0.outputs - input_imgs))
        SSE_loss = tf.reduce_mean(tf.square(gen0.outputs - input_imgs))# /FLAGS.output_size/FLAGS.output_size/3
        '''
        KL divergence:
        we get z_mean,z_log_sigma_sq from encoder, then we get z from N(z_mean,z_sigma^2)
        then compute KL divergence between z and standard normal gaussian N(0,I) 
        '''
        # train_vae
        # KL_loss = tf.reduce_mean(- 0.5 * tf.reduce_mean(1 + tf.clip_by_value(z_log_sigma_sq,-10.0,10.0) - 
        #                 tf.square(tf.clip_by_value(z_mean,-10.0,10.0)) - tf.exp(tf.clip_by_value(z_log_sigma_sq,-10.0,10.0)),1))
        # train_vae2
        KL_loss = tf.reduce_mean(- 0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq),1))

        ### important points! ###
        # KL_weight = 1.0
        # # LL_weight = 1.0 # 19th July 10:05
        # LL_weight = 0.5 #0.5
        # Loss_encoder = tf.clip_by_value(KL_weight * KL_loss + 0.5*(l_layer_loss + SSE_loss),-100.0,100.0)
        # Loss_generator = tf.clip_by_value(0.5*(l_layer_loss + SSE_loss) + gan_g_loss,-100.0,100.0)

        # VAE_loss = KL_loss + SSE_loss # train_vae
        VAE_loss = 0.005*KL_loss + SSE_loss # KL_loss isn't working well if the weight of SSE is too big

        # Loss_encoder = tf.clip_by_value(KL_weight * KL_loss + SSE_loss,-100.0,100.0)
        # Loss_generator = tf.clip_by_value(gan_g_loss + SSE_loss,-100.0,100.0)
        # Loss_discriminator = tf.clip_by_value(gan_d_loss,-100.0,100.0)

        e_vars = tl.layers.get_variables_with_name('encoder',True,True)
        g_vars = tl.layers.get_variables_with_name('generator', True, True)
        # d_vars = tl.layers.get_variables_with_name('discriminator', True, True)
        vae_vars = e_vars+g_vars

        print("-------encoder-------")
        net_out1.print_params(False)
        print("-------generator-------")
        gen0.print_params(False)
        # print("-------discriminator--------")
        # dis.print_params(False)
        # print("---------------")

        # optimizers for updating encoder, discriminator and generator
        # e_optim = tf.train.AdamOptimizer(lr_e, beta1=FLAGS.beta1) \
        #                   .minimize(Loss_encoder, var_list=e_vars)
        # d_optim = tf.train.AdamOptimizer(lr_d, beta1=FLAGS.beta1) \
        #                   .minimize(Loss_discriminator, var_list=d_vars)
        # g_optim = tf.train.AdamOptimizer(lr_g, beta1=FLAGS.beta1) \
        #                   .minimize(Loss_generator, var_list=g_vars)
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
    # errE = 0.0
    # errG = 0.0
    # errD = 0.0

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
                # e_current_lr = FLAGS.learning_rate * balance(np.mean(errE),-.5,15)
                # g_current_lr = FLAGS.learning_rate * balance(np.mean(errG),-.5,15)
                # d_current_lr = FLAGS.learning_rate * balance(np.mean(errD),-.5,15)
                # e_current_lr = FLAGS.learning_rate
                # g_current_lr = FLAGS.learning_rate
                # d_current_lr = FLAGS.learning_rate
                vae_current_lr = FLAGS.learning_rate


                # update
                # errE, _ = sess.run([Loss_encoder, e_optim], feed_dict={input_imgs: batch_images})
                # errD, _ = sess.run([Loss_discriminator, d_optim], feed_dict={z: batch_z, real_images: batch_images})
                # errG, _ = sess.run([Loss_generator, g_optim], feed_dict={z: batch_z})
                # A, B, C, D, E = sess.run([KL_loss, l_layer_loss, gan_d_loss, gan_g_loss, SSE_loss], feed_dict = {input_imgs: batch_images, lr_e:e_current_lr, 
                # 	lr_d:d_current_lr, lr_g:g_current_lr})
                # print('------------------------------------')
                # print("         KL_loss: %.8f, l_layer_loss: %.8f, gan_d_loss:%.8f, gan_g_loss:%.8f, SSE_loss:%.8f  " \
                #         % (A, B, C, D, E))

                # A, B, C, D, E = sess.run([KL_loss, hidden_loss, gan_d_loss, gan_g_loss, SSE_loss], feed_dict = {input_imgs: batch_images, lr_e:e_current_lr, 
                #     lr_d:d_current_lr, lr_g:g_current_lr})
                # print('------------------------------------')
                # print("         KL_loss: %.8f, hidden_loss: %.8f, gan_d_loss:%.8f, gan_g_loss:%.8f, SSE_loss:%.8f  " \
                #         % (A, B, C, D, E))

                # A, B, C, D = sess.run([KL_loss, gan_d_loss, gan_g_loss, SSE_loss], feed_dict = {input_imgs: batch_images, lr_e:e_current_lr, 
                #     lr_d:d_current_lr, lr_g:g_current_lr})
                # print('------------------------------------')
                # print("         KL_loss: %.8f, gan_d_loss:%.8f, gan_g_loss:%.8f, SSE_loss:%.8f  " \
                #         % (A, B, C, D))

                # errE, errG, errD, _, _, _ = sess.run([Loss_encoder, Loss_generator, Loss_discriminator, e_optim,
                # 	g_optim, d_optim], feed_dict = {input_imgs: batch_images, lr_e:e_current_lr, 
                # 	lr_d:d_current_lr, lr_g:g_current_lr})
                # errD, _ = sess.run([Loss_discriminator, d_optim], feed_dict={input_imgs: batch_images, lr_d:d_current_lr})
                # for i in range(2):
                #     errG, _ = sess.run([Loss_generator, g_optim], feed_dict={input_imgs: batch_images, lr_g:g_current_lr})
                # errE, _ = sess.run([Loss_encoder, e_optim], feed_dict={input_imgs: batch_images, lr_e:e_current_lr})

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

