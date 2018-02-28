import os
import numpy as np
import tensorflow as tf
from collections import namedtuple
from tqdm import tqdm
from glob import glob
import random

from module import * 
from util import * 

class stargan(object):
    def __init__(self,sess,args):

        self.sess = sess
        self.phase = args.phase # train or test
        self.data_dir = args.data_dir # ./data/celebA
        self.log_dir = args.log_dir # ./assets/log
        self.ckpt_dir = args.ckpt_dir # ./assets/checkpoint
        self.sample_dir = args.sample_dir # ./assets/sample
        self.test_dir = args.test_dir # ./assets/test
        self.epoch = args.epoch # 100
        self.batch_size = args.batch_size # 16
        self.image_size = args.image_size # 64
        self.image_channel = args.image_channel # 3
        self.nf = args.nf # 64
        self.n_label = args.n_label # 10
        self.lambda_adv= args.lambda_adv
        self.lambda_gp = args.lambda_gp
        self.lambda_cls = args.lambda_cls # 1
        self.lambda_rec = args.lambda_rec # 10
        self.lr = args.lr # 0.0001
        self.beta1 = args.beta1 # 0.5
        self.continue_train = args.continue_train # False
        self.snapshot = args.snapshot # 100
        self.adv_type = args.adv_type # WGAN or GAN
        self.binary_attrs = args.binary_attrs
        self.d_steps = args.d_steps
        self.c_method = args.c_method
        self.attr_keys = args.attr_keys
        
        
        # hyper-parameter for building the module
        OPTIONS = namedtuple('OPTIONS', ['batch_size', 'image_size', 'nf', 'n_label', 'lambda_gp'])
        self.options = OPTIONS(self.batch_size, self.image_size, self.nf, self.n_label, self.lambda_gp)
        
        # build model & make checkpoint saver 
        self.build_model()
        self.saver = tf.train.Saver()
        
    def build_model(self):
        
        self.real_img = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size, self.image_channel],
                                     name='input_images')
        self.real_atr = tf.placeholder(tf.float32,
                                     [None, self.n_label], name='input_images_attributes')
        self.fake_atr = tf.placeholder(tf.float32,
                                     [None, self.n_label], name='target_images_attributes')
        
        
        self.epsilon = tf.placeholder(tf.float32, [None,1,1,1], name='gp_random_num')
        self.lr_decay = tf.placeholder(tf.float32, None, name='lr_decay')
        
        # generate fake image base on target attributes
        fake_atr_tile   = tf.tile(tf.reshape(self.fake_atr, [-1,1,1,self.n_label]),[1,self.image_size,self.image_size,1])
        real_img_concat = tf.concat((self.real_img, fake_atr_tile), axis=3)
        self.fake_img = generator(real_img_concat, self.options, False, name='gen')
        
        # reconstruct image
        real_atr_tile   = tf.tile(tf.reshape(self.real_atr, [-1,1,1,self.n_label]),[1,self.image_size,self.image_size,1])
        fake_img_concat = tf.concat((self.fake_img, real_atr_tile), axis=3)
        self.recon_img = generator(fake_img_concat, self.options, True , name='gen')
        
        # discriminate image
        # src: real or fake, cls: domain classification 
        self.src_real_img, self.cls_real_img = discriminator(self.real_img, self.options, False, name='disc')
        self.src_fake_img, self.cls_fake_img = discriminator(self.fake_img, self.options, True , name='disc')
        
        ### adversarial loss
        if self.adv_type == 'WGAN':
            self.gp_loss = self.lambda_gp * wgan_gp_loss(self.real_img, self.fake_img, self.options) #self.epsilon
            self.d_loss_fake = tf.reduce_mean(self.src_fake_img)
            self.d_loss_real = -tf.reduce_mean(self.src_real_img)
            self.d_adv_loss = self.d_loss_fake + self.d_loss_real + self.gp_loss
        elif self.adv_type == 'GAN': # 'GAN'
            d_real_adv_loss = gan_loss(self.src_fake_img, tf.ones_like(self.src_fake_img))
            d_fake_adv_loss = gan_loss(self.src_fake_img, tf.zeros_like(self.src_fake_img))
            self.d_adv_loss = d_real_adv_loss + d_fake_adv_loss
        else: #LSGAN
            d_real_adv_loss = lsgan_loss(self.src_real_img, tf.ones_like(self.src_real_img))
            d_fake_adv_loss = lsgan_loss(self.src_fake_img, tf.zeros_like(self.src_fake_img))
            self.d_adv_loss = d_real_adv_loss + d_fake_adv_loss
        
        ### domain classification loss
        if self.c_method == 'Sigmoid':
            self.d_real_cls_loss = cls_loss(self.cls_real_img, self.real_atr)
        else:
            self.d_real_cls_loss = cls_loss_SoftCE(self.cls_real_img, self.real_atr)
        
        ### disc loss function
        self.d_loss = self.d_adv_loss + self.lambda_cls * self.d_real_cls_loss
        
        ## generator loss ##
        ### adv loss
        if self.adv_type == 'WGAN':
            self.g_adv_loss = -tf.reduce_mean(self.src_fake_img)
        elif self.adv_type == 'GAN' : # 'GAN'
            self.g_adv_loss = gan_loss(self.src_fake_img, tf.ones_like(self.src_fake_img))
        else: #LSGAN
            self.g_adv_loss = lsgan_loss(self.src_fake_img, tf.ones_like(self.src_fake_img))
        
        ### domain classificatioin loss
        if self.c_method == 'Sigmoid':
            self.g_fake_cls_loss = cls_loss(self.cls_fake_img, self.fake_atr)
        else:
            self.g_fake_cls_loss = cls_loss_SoftCE(self.cls_fake_img, self.fake_atr)
        
        ### reconstruction loss
        self.g_recon_loss = recon_loss(self.real_img, self.recon_img)
        ### gen loss function
        self.g_loss = self.g_adv_loss + self.lambda_cls * self.g_fake_cls_loss + self.lambda_rec * self.g_recon_loss
        
        # trainable variables
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'disc' in var.name]
        for var in self.d_vars: print(var.name)
        self.g_vars = [var for var in t_vars if 'gen' in var.name]
        for var in self.g_vars: print(var.name)
        # optimizer
        self.d_optim = tf.train.AdamOptimizer(self.lr * self.lr_decay, beta1=self.beta1).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr * self.lr_decay, beta1=self.beta1).minimize(self.g_loss, var_list=self.g_vars)
        
        self.acc = self.compute_accuracy(self.cls_real_img,self.real_atr,self.c_method)
    
    def train(self):
        # summary setting
        self.summary()
        
        # load train data list & load attribute data
        data_files = load_data_list(self.data_dir)
        self.attr_names, self.attr_list = attr_extract(self.data_dir)
        
        # variable initialize
        self.sess.run(tf.global_variables_initializer())
        
        # load or not checkpoint
        if self.continue_train and self.checkpoint_load():
            print(" [*] before training, Load SUCCESS ")
        else:
            print(" [!] before training, no need to Load ")
        
        batch_idxs = len(data_files) // self.batch_size # 182599
        count = 0
        #train
        for epoch in range(self.epoch):
            # get lr_decay
            if epoch < self.epoch / 2:
                lr_decay = 1.0
            else:
                lr_decay = float(self.epoch - epoch) / float(self.epoch / 2)
            
            # data shuffle at the begining of an epoch
            np.random.shuffle(data_files)
            
            for idx in tqdm(range(batch_idxs)):
                count += 1
                # 
                data_list  = data_files[idx * self.batch_size : (idx+1) * self.batch_size] #reading batch
                attr_list  = [self.attr_list[os.path.basename(val)] for val in data_list] #get basename of dataA_list (for getting attrs later)
                attr_list_ = np.copy(attr_list)
                np.random.shuffle(attr_list_)
                
                # get batch images and labels
                real_atr = preprocess_attr(self.attr_names, attr_list , self.attr_keys) # Only reserve attrs that is listed in attr_keys.
                fake_atr = preprocess_attr(self.attr_names, attr_list_, self.attr_keys) 
                real_img = preprocess_image(data_list, self.image_size, phase='train') # Read images
                
                # update D network for d_steps times
                epsilon = np.random.rand(self.batch_size,1,1,1)
                feed = { self.real_img: real_img, self.real_atr: np.array(real_atr), self.fake_atr: np.array(fake_atr), self.epsilon: epsilon, self.lr_decay: lr_decay }
                _, d_loss, d_summary,gp_loss = self.sess.run([self.d_optim, self.d_loss, self.d_sum,self.gp_loss], feed_dict = feed)

                # updatae G network for 1 time
                if (idx+1) % self.d_steps == 0:
                    feed = { self.real_img: real_img, self.real_atr: np.array(real_atr), self.fake_atr: np.array(fake_atr), self.lr_decay: lr_decay }
                    _, g_loss, g_summary = self.sess.run([self.g_optim, self.g_loss, self.g_sum],
                                                             feed_dict = feed)
                                
                # summary
                    self.writer.add_summary(g_summary, count)
                self.writer.add_summary(d_summary, count)
                
                # save checkpoint and samples
                if count % self.snapshot == 0:
                    print("Epoch:%02d, Iter: %06d, g_loss: %4.4f, d_loss: %4.4f, gp_loss: %4.4f" % (epoch, count, g_loss, d_loss, gp_loss))
                    # checkpoint
                    self.checkpoint_save(count)
                    
                    # save samples (from test dataset)
                    self.sample_save(count)
        
        
    def test(self):
        # check if attribute available
        if not len(self.binary_attrs) == self.n_label:
            print ("binary_attr length is wrong! The length should be {}".format(self.n_label))
            return
        
        
        # load or not checkpoint
        if self.phase=='test' and self.checkpoint_load():
            print(" [*] before training, Load SUCCESS ")
            
            self.attr_names, self.attr_list = attr_extract(self.data_dir)
            test_files = glob(os.path.join(self.data_dir, 'test', '*'))
            test_list = random.sample(test_files, 10)
            attr_list = [self.attr_list[os.path.basename(val)] for val in test_list]
            # get batch images and labels
            real_atr = preprocess_attr(self.attr_names, attr_list, self.attr_keys) # Only reserve attrs that is listed in attr_keys.
            fake_atr = [float(i) for i in list(self.binary_attrs)] * len(test_list)
            fake_atr = np.array(fake_atr)
            fake_atr = np.reshape(fake_atr,[-1,self.n_label])
            real_img = preprocess_image(test_list, self.image_size, phase='test')
            # generate fakeB
            feed = { self.real_img: real_img, self.real_atr: real_atr,self.fake_atr: fake_atr }
            fake_img,recon_img = self.sess.run([self.fake_img,self.recon_img], feed_dict = feed)
            
            # save samples
            test_file = os.path.join(self.test_dir, 'test'+str(self.binary_attrs)+'.jpg')
            save_images(real_img, fake_img, recon_img, self.image_size, test_file, num=10)

        else:
            print(" [!] before training, no need to Load ")
    
    def test_all(self,num_sample=100):
        # check if attribute available
        if self.phase=='test_all' and self.checkpoint_load():
            print(" [*] before training, Load SUCCESS ")            
            
            test_files = glob(os.path.join(self.data_dir, 'test', '*'))
            self.attr_names, self.attr_list = attr_extract(self.data_dir)
            # [5,6] with the sequnce of (realA, realB, fakeB), totally 10 set save
            test_list = test_files[:num_sample]
            attr_list = [self.attr_list[os.path.basename(val)] for val in test_list]
            #attr_list_ = np.copy(attr_list)
            #np.random.shuffle(attr_list_)
            
            fake_atr = np.identity(self.n_label)
            # get batch images and labels
            real_atr = preprocess_attr(self.attr_names, attr_list, self.attr_keys) # Only reserve attrs that is listed in attr_keys.
            real_img = preprocess_image(test_list, self.image_size, phase='test') # Read images
            
            for idx,img in enumerate(real_img):
                # generate fakeB
                org_img = img.copy()
                #img = np.array([img,img,img])
                img = np.reshape(img,[1,self.image_size,self.image_size,self.image_channel])
                #print(np.shape(img))
                img = np.repeat(img,self.n_label,axis=0)
                #print(np.shape(img))
                feed = { self.real_img: img, self.real_atr: np.array(real_atr), self.fake_atr: np.array(fake_atr) }
                #fake_img,recon_img = self.sess.run([self.fake_img_sample,self.recon_img_sample], feed_dict = feed)
                fake_img = self.sess.run(self.fake_img, feed_dict = feed)
                fake_img = list(fake_img)
                # save samples
                file_name = os.path.basename(test_list[idx])
                test_file = os.path.join(self.test_dir, file_name)
                #test_file = os.path.join(self.test_dir, 'test2.jpg')
                img_list = [org_img]
                img_list = img_list+fake_img
                save_images_test(img_list, self.image_size, test_file, num=1, col=self.n_label+1)

        else:
            print(" [!] before training, no need to Load ")        
    
    def test_aux_accuracy(self):
        if self.phase=='aux_test' and self.checkpoint_load():
            print(" [*] before training, Load SUCCESS ")
            # [5,6] with the seequnce of (realA, realB, fakeB), totally 10 set save
            
            self.attr_names, self.attr_list = attr_extract(self.data_dir)
            test_files = glob(os.path.join(self.data_dir, 'test', '*'))
            batch_idxs = len(test_files) // self.batch_size # 182599
            over_all_acc = 0
            for idx in tqdm(range(batch_idxs)):
                test_list  = test_files[idx * self.batch_size : (idx+1) * self.batch_size] #reading batch
                attr_list = [self.attr_list[os.path.basename(val)] for val in test_list]

                real_atr = preprocess_attr(self.attr_names, attr_list, self.attr_keys) # Only reserve attrs that is listed in attr_keys.
                real_img = preprocess_image(test_list, self.image_size, phase='test')
                feed = { self.real_img: real_img, self.real_atr: real_atr }
                batch_acc = self.sess.run(self.acc, feed_dict = feed)
                over_all_acc += batch_acc
            print('overall accuracy: %3.3f'%(over_all_acc/batch_idxs))
        else:
            print(" [!] before training, no need to Load ")

    def compute_accuracy(self, x, y, method='Sigmoid'):
        if method == 'Sigmoid':
            x = tf.nn.sigmoid(x)
            predicted = self.threshold(x)
            correct = tf.cast(tf.equal(predicted, y),tf.float32)
            accuracy = tf.reduce_mean(correct) * 100.0
        else:
            x = tf.argmax(x,axis=1)
            y = tf.argmax(y,axis=1)
            correct = tf.cast(tf.equal(x, y),tf.float32)
            accuracy = tf.reduce_mean(correct) * 100.0
        return accuracy

    def threshold(self,x):
        ans = tf.cast(tf.greater(x,0.5),tf.float32)
        return ans

    def summary(self):
        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        
        # session : discriminator
        sum_d_1 = tf.summary.scalar('D/adv_loss', self.d_adv_loss)
        sum_d_2 = tf.summary.scalar('D/real_cls_loss', self.d_real_cls_loss)
        sum_d_3 = tf.summary.scalar('D/d_loss', self.d_loss)
        sum_d_4 = tf.summary.scalar('D/d_cls_acc',self.acc)
        sum_d_5 = tf.summary.scalar('D/d_gp',self.gp_loss)
        sum_d_6 = tf.summary.scalar('D/fake_loss',self.d_loss_fake)
        sum_d_7 = tf.summary.scalar('D/real_loss',self.d_loss_real)
        self.d_sum = tf.summary.merge([sum_d_1, sum_d_2, sum_d_3,sum_d_4,sum_d_5,sum_d_6,sum_d_7])
        
        # session : generator
        sum_g_1 = tf.summary.scalar('G/adv_loss', self.g_adv_loss)
        sum_g_2 = tf.summary.scalar('G/fake_cls_loss', self.g_fake_cls_loss)
        sum_g_3 = tf.summary.scalar('G/recon_loss', self.g_recon_loss)
        sum_g_4 = tf.summary.scalar('G/g_loss', self.g_loss)
        self.g_sum = tf.summary.merge([sum_g_1, sum_g_2, sum_g_3, sum_g_4])
       
    
    def checkpoint_load(self):
        print(" [*] Reading checkpoint...")
        
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print ('found!',ckpt_name)
            self.saver.restore(self.sess, os.path.join(self.ckpt_dir, ckpt_name))
            return True
        else:
            return False    
        
        
    def checkpoint_save(self, step):
        model_name = "stargan.model"
        self.saver.save(self.sess,
                        os.path.join(self.ckpt_dir, model_name),
                        global_step=step)
        
    
    def sample_save(self, step):
        num_sample = self.n_label
        test_files = glob(os.path.join(self.data_dir, 'test', '*'))
        
        # [5,6] with the sequnce of (realA, realB, fakeB), totally 10 set save
        test_list = random.sample(test_files, num_sample)
        attr_list = [self.attr_list[os.path.basename(val)] for val in test_list]
        #attr_list_ = np.copy(attr_list)
        #np.random.shuffle(attr_list_)
        
        fake_atr = np.identity(num_sample)
        # get batch images and labels
        real_atr = preprocess_attr(self.attr_names, attr_list, self.attr_keys) # Only reserve attrs that is listed in attr_keys.
        real_img = preprocess_image(test_list, self.image_size, phase='test') # Read images
                        
        # generate fakeB
        feed = { self.real_img: real_img, self.real_atr: np.array(real_atr), self.fake_atr: np.array(fake_atr) }
        fake_img,recon_img = self.sess.run([self.fake_img,self.recon_img], feed_dict = feed)
        
        # save samples
        sample_file = os.path.join(self.sample_dir, '%06d.jpg'%(step))
        save_images(real_img, fake_img, recon_img, self.image_size, sample_file, num=num_sample)

