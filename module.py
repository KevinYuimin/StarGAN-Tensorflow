import tensorflow as tf
from ops import conv2d, deconv2d, relu, lrelu, instance_norm, tanh
import numpy as np

def generator(images, options, reuse=False, name='gen'):
    # reuse or not
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "CONSTANT") #CONSTANT
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_in1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "CONSTANT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_in2')
            return y + x
            
        # down sampling
        c0 = tf.pad(images, [[0, 0], [3, 3], [3, 3], [0, 0]], "CONSTANT")
        c1 = relu(instance_norm(conv2d(c0,   options.nf, ks=7, s=1, padding='VALID', name='gen_ds_conv1'), 'in1_1'))
        c2 = relu(instance_norm(conv2d(c1, 2*options.nf, ks=4, s=2, name='gen_ds_conv2'), 'in1_2'))
        c3 = relu(instance_norm(conv2d(c2, 4*options.nf, ks=4, s=2, name='gen_ds_conv3'), 'in1_3'))
        
        # bottleneck
        r1 = residule_block(c3, options.nf*4, name='g_r1')
        r2 = residule_block(r1, options.nf*4, name='g_r2')
        r3 = residule_block(r2, options.nf*4, name='g_r3')
        r4 = residule_block(r3, options.nf*4, name='g_r4')
        r5 = residule_block(r4, options.nf*4, name='g_r5')
        r6 = residule_block(r5, options.nf*4, name='g_r6')

        # up sampling
        d1 = relu(instance_norm(deconv2d(r6, options.nf*2, 4, 2, name='g_us_dconv1'), 'g_d1_in'))
        d2 = relu(instance_norm(deconv2d(d1, options.nf  , 4, 2, name='g_us_dconv2'), 'g_d2_in'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "CONSTANT") #REFLECT instead of Padding with 0, [batch,h,w,c]
        pred = tf.nn.tanh(conv2d(d2, 3, 7, 1, padding='VALID', name='g_pred_c'))
        
        return pred

def discriminator(images, options, reuse=False, repeat_num=6 ,name='disc'): #original version
    # In StarGAN Discriminator do not use instance normalization
    # reuse or not
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
            
        # input & hidden layer
        h1 = lrelu(conv2d(images, options.nf, ks=4, s=2, name='disc_conv1'))
        h2 = lrelu(conv2d(h1, 2*options.nf, ks=4, s=2, name='disc_conv2'))
        h3 = lrelu(conv2d(h2, 4*options.nf, ks=4, s=2, name='disc_conv3'))
        h4 = lrelu(conv2d(h3, 8*options.nf, ks=4, s=2, name='disc_conv4'))
        h5 = lrelu(conv2d(h4, 16*options.nf, ks=4, s=2, name='disc_conv5'))
        h6 = lrelu(conv2d(h5, 32*options.nf, ks=4, s=2, name='disc_conv6'))
        # (batch, h/64, w/64, 2048)
        
        # output layer
        # (batch, h/64, w/64, 2048) ==> (batch, h/64, w/64, 1) #patch GAN
        src = conv2d(h6, 1, ks=3, s=1, name='disc_conv7_patch') # (batch, h/64, w/64, 1)
        # (batch, h/64, w/64, 2048) ==> (batch, 1, 1, num_cls) #big kernel size conv
        
        k_size = int(options.image_size / np.power(2, repeat_num))
        aux = conv2d(h6, options.n_label, ks=k_size, s=1, padding='VALID', name='disc_conv8_aux') # (batch, 1, 1, num_cls) 
        aux = tf.reshape(aux,[-1,options.n_label])
        return src, aux

def wgan_gp_loss(real_img, fake_img, options,mode=None): #gradient penalty
    alpha = tf.random_uniform(
        shape=[options.batch_size,1,1,1], 
        minval=0.,
        maxval=1.
    )


    hat_img = alpha * real_img + (1.-alpha) * fake_img
    gradients = tf.gradients(discriminator(hat_img, options, reuse=True, name='disc')[0], xs=[hat_img])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
    gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.))
    
    return gradient_penalty
        
def gan_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels))

def lsgan_loss(in_, target):
    return tf.reduce_mean((in_-target)**2)

def cls_loss(logits, labels):
    # sigmoid cross entropy return [batchsize,n_label]
    return tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels),axis=1))

def cls_loss_SoftCE(logits, labels):
    # softmax cross entropy return [batchsize]
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))


def recon_loss(image1, image2):
    return tf.reduce_mean(tf.abs(image1 - image2))
