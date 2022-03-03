from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime

# set date and time to append to file names
import time
dt = datetime.now().strftime("%Y-%m-%d-%H%M")
start = int(time.time()) 

import argparse
from absl import app
import math
import numpy as np
import os
import sys
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import skimage
from skimage.io import imread
from skimage.transform import rotate
import yaml
import logging
# logging.getLogger('tensorflow').disabled = True
# tf.logging.set_verbosity(tf.logging.WARN)
import wandb
import warnings
warnings.filterwarnings('ignore')

FLAGS = None

# get GPU card as assigned
gpu = int(os.getenv('SGE_HGR_gpu_card'))

# set date and time to append to file names
dt = datetime.now().strftime("%Y-%m-%d-%H%M")

# load config file
cnf = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

# set necessary parameters
batch_size = cnf["TRAIN"]["batch_size"]
logdir = cnf["TRAIN"]["logdir"]
imgdir = cnf["TRAIN"]["imgdir"]
modeldir = cnf["TRAIN"]["modeldir"]
iter_num = cnf["TRAIN"]["iter_num"]
pretrained_model = cnf["TRAIN"]["pretrained_model"]
age_group = cnf["TRAIN"]["age_group"]
notes = cnf["TRAIN"]["notes"]

imageType = 1
ous = 512 #640
ins = ous
interv = 200
nclass = 2
# axis = 0 # which direction 
train_stacks = []

if not os.path.isdir(f'{modeldir}/{age_group}_{dt}'):
  os.makedirs(f'{modeldir}/{age_group}_{dt}')


def imread2D_dir2(d):
  imgs = []
  for file in os.listdir(d):
    if file.endswith(".png"):
      imgs.append(file)
  imgs.sort()
  tempI = imread(os.path.join(d,imgs[0]))
  z = len(imgs)
  h = tempI.shape[0]
  w = tempI.shape[1]
  c = 1
  if (len(tempI.shape)>2):
    c = tempI.shape[2]
  # print(d)
  # print(z,h,w,c)

  I = np.zeros((z,h,w,c))
  idx = 0
  for i_name in imgs:
    tempI = skimage.img_as_float(imread(os.path.join(d,i_name)))
    if (len(tempI.shape)==2):
      I[idx,:,:,0] = tempI
    else:
      I[idx,:,:,:] = tempI
    idx+=1
    # print('image up: ',i_name)
  meanI = np.mean(I)
  stdI = np.std(I)
  # print('Image bottom: ',I)
  # print('norm:',I.max(),I.min(),meanI,stdI)
  return z,h,w,c, imgs, meanI,stdI

def netpad(I):
  oh = I.shape[0]
  ow = I.shape[1]
  ph = max(oh,ous) - oh
  pw = max(ow,ous) - ow
  hph = math.floor(ph/2)
  hpw = math.floor(pw/2)
  if len(I.shape)==3:
    nI = np.zeros((max(oh,ous),max(ow,ous),imageType))
    nI[hph:hph+oh,hpw:hpw+ow,:] = I
    return nI
  nI = np.zeros((max(oh,ous),max(ow,ous)))
  nI[hph:hph+oh,hpw:hpw+ow] = I
  return nI

def load_data(file_dir):
  print('--'*35)
  print(f'Loading Training Data from: {file_dir}')
  # get the dimensionns of the 3D data
  cur_dir_prefix = []
  train_idx = []
  train_dims = []
  train_stats = []
  for cur_dir in os.listdir(os.path.join(file_dir,'mask/')):
    dm = os.path.join(file_dir, 'mask/', cur_dir)
    train_stacks.append(dm)

    if os.path.isdir(dm):
      di  = os.path.join(file_dir, 'img/', cur_dir)
      cur_dir_prefix.append(cur_dir)
      # get dimensions and statistics
      z,h,w,c,imgs, meanI,stdI = imread2D_dir2(di)
      train_dims.append((z,h,w,c))
      train_stats.append((meanI,stdI))

      # generate training set 
      cur_idx = []
      for j in range(z):
        i_name = imgs[j]
        tempI = skimage.img_as_float(imread(os.path.join(dm,i_name)))
        if (np.sum(tempI)>0):
          cur_idx.append(j)
      train_idx.append(cur_idx)

  for i in range(len(cur_dir_prefix)):
    print('--'*35)
    print(f'Training Stack: {cur_dir_prefix[i]}, # annotated slices: {len(train_idx[i])}')
    # print(f'Annotated Slice index: {train_idx[i]}')
    print('--'*35)


  # saving txt file with training stacks and model infos:
  f = open(f'{modeldir}/{age_group}_{dt}/info.txt', "w")
  f.write(notes+"\n")
  f.write("Training stacks:"+"\n")
  for stack in train_stacks:
    f.write(stack)
    f.write('\n')
  f.close

  print(f'Created a text file {modeldir}/{age_group}_{dt}/info.txt with infos about the model.')
  return cur_dir_prefix, train_dims, train_idx, train_stats


def get_image(cur_dir_prefix,train_dims,train_idx,train_stats):
  #  which stack?
  i_stk = random.randint(0, len(cur_dir_prefix)-1)
  cur_dir = cur_dir_prefix[i_stk]
  di = os.path.join(imgdir, 'img/', cur_dir)
  dm = os.path.join(imgdir, 'mask/', cur_dir)
  I_shape = train_dims[i_stk]
  cur_idx = train_idx[i_stk]
  meanI,stdI = train_stats[i_stk]
  #  which slice?
  i_idx = cur_idx[random.randint(0, len(cur_idx)-1)]
  flnm = "{}_{:04}.png".format(cur_dir, i_idx)
  # load img
  big_tI = np.zeros((I_shape[1],I_shape[2],imageType))
  big_tI[:,:,0] = skimage.img_as_float(imread(os.path.join(di,flnm)))
  #print(big_tI.shape)
  big_lI = skimage.img_as_float(imread(os.path.join(dm,flnm)))
  #print(big_lI.shape)
  xs = big_tI.shape[0]
  ys = big_tI.shape[1]
  #print(xs,ys)
  stx = random.randint(0,xs-ins)
  sty = random.randint(0,ys-ins)
  train_sample = np.zeros((ins,ins,imageType))
  label_sample = np.zeros((ins,ins))
  label_sample = big_lI[stx:stx+ins,sty:sty+ins]
  while np.sum(label_sample) == 0:
    p = random.uniform(0,1)
    if p < 0.85:
      stx = random.randint(0,xs-ins)
      sty = random.randint(0,ys-ins)
      label_sample = big_lI[stx:stx+ins,sty:sty+ins]
    else:
      break
  train_sample = big_tI[stx:stx+ins,sty:sty+ins,:]
  nrotate = random.randint(0, 3)
  train_sample = rotate(train_sample, 90*nrotate)
  label_sample = np.round(rotate(label_sample, 90*nrotate)*255).astype('uint8')
  if np.max(label_sample) < 200:
    label_sample = label_sample*2
  nflip = random.randint(0, 1)
  if nflip:
    #print('flip')
    train_sample = np.fliplr(train_sample)
    label_sample = np.fliplr(label_sample)

  train_sample = (train_sample-meanI)/stdI

  gstd = np.random.uniform(low=1/1.2,high=1.2)
  gmean = np.random.uniform(low=-0.2,high=0.2)
  train_sample = train_sample * gstd
  train_sample = train_sample + gmean
  return train_sample, label_sample

def get_batch(cur_dir_prefix,train_dims,train_idx,train_stats,batch_size):
  train_samples = np.zeros((batch_size,ins,ins,imageType))
  label_samples = np.zeros((batch_size,ins,ins))
  for i in range(batch_size):
    train_sample, label_sample = get_image(cur_dir_prefix,train_dims,train_idx,train_stats)
    train_samples[i,:,:,:] = train_sample
    label_samples[i,:,:] = label_sample/interv
  return train_samples, label_samples.astype('int32')

def bias_variable(shape):
  initial = tf.get_variable("bias", shape=shape, initializer=tf.constant_initializer(value=0.0))
  return initial

def conv_(x,k,nin,nout,phase,s=1,d=1):
  stdv = math.sqrt(2/(nin*k*k))
  return tf.layers.conv2d(x, nout, k, strides=[s,s], dilation_rate=[d,d], padding='same',
                                      kernel_initializer = tf.random_normal_initializer(stddev=stdv),
                                      use_bias=False)

def bn_(x,phase):
  bn_result = tf.layers.batch_normalization(x, momentum=0.9, epsilon = 1e-5,
                                            gamma_initializer = tf.random_uniform_initializer(minval=0.0, maxval=1.0,dtype=tf.float32),
                                            training = phase)
  return bn_result

def relu_(x):
  return tf.nn.relu(x)

def cbr_(x,k,nin,nout,phase,s=1,d=1):
  x_conv = conv_(x,k,nin,nout,phase,s,d)
  x_bn = bn_(x_conv,phase)
  x_relu = relu_(x_bn)
  return x_relu

def bottleneck(x, nin, nout, phase, s=1, d=1):
  if (nin != nout) or (s != 1):
    print('conv_skip')
    with tf.variable_scope('skip'):
      skip_conv = conv_(x,1,nin,nout,phase,s=s)
      skip = bn_(skip_conv,phase)
  else:
    skip = x

  with tf.variable_scope('conv1'):
    c1_conv = conv_(x,3,nin,nout,phase,s=s,d=d)
    c1_bn = bn_(c1_conv,phase)
    c1_relu = relu_(c1_bn)
    
  with tf.variable_scope('conv2'):
    c2_conv = conv_(c1_relu,3,nout,nout,phase,d=d)
    c2_bn = bn_(c2_conv,phase)

  out = skip + c2_bn
  out_relu = relu_(out)
  return out_relu

def stack(x, nin, nout, nblock, phase, s=1, d=1, new_level=True):
  for i in range(nblock):
    with tf.variable_scope('block%d' % (i)):
      if i==0:
        x = bottleneck(x,nin,nout,phase,s=s,d=d)
      else:
        x = bottleneck(x,nout,nout,phase,d=d)
  return x

def max_pool_2x2(x):
  print('new pool')
  return tf.layers.max_pooling2d(x,[2,2],[2,2])

def deconv_(x, nin, nout, phase):
  stdv = math.sqrt(2/(nin*2*2))
  conv_r = tf.layers.conv2d_transpose(x, nout, 4, strides=[2,2], padding='same',
                                      kernel_initializer = tf.random_normal_initializer(stddev=stdv),
                                      use_bias=False)
  bn_r = bn_(conv_r,phase)
  relu_r = relu_(bn_r)
  return relu_r

def deconvn_(x, nin, nout, num_up, phase):
  cur_in = nin
  cur_out=((2**(num_up-1))*nout)

  for i in range(num_up):
    with tf.variable_scope('up%d' % (i)):
      x = deconv_(x,cur_in,cur_out,phase)
    cur_in = cur_out
    cur_out = math.floor(cur_out/2)

  return x

def model_apply2d(sess,result,input_img,phase,tI,ins):
  otI = tI
  oh = otI.shape[0]
  ow = otI.shape[1]
  ph = max(oh,ous) - oh
  pw = max(ow,ous) - ow
  hph = math.floor(ph/2)
  hpw = math.floor(pw/2)

  tI = netpad(tI)

  wI=np.zeros([ins,ins])
  pmap=np.zeros([tI.shape[0],tI.shape[1],nclass-1])
  avI=np.zeros([tI.shape[0],tI.shape[1],nclass-1])
  for i in range(ins):
    for j in range(ins):
      dx=min(i,ins-1-i)
      dy=min(j,ins-1-j)
      d=min(dx,dy)+1
      wI[i,j]=d;
  wI = wI/wI.max()

  avk = 2
  nrotate = 4
  for i1 in range(math.ceil(float(avk)*(float(tI.shape[0])-float(ins))/float(ins))+1):
    for j1 in range(math.ceil(float(avk)*(float(tI.shape[1])-float(ins))/float(ins))+1):
      insti=math.floor(float(i1)*float(ins)/float(avk))
      instj=math.floor(float(j1)*float(ins)/float(avk))
      inedi=insti+ins
      inedj=instj+ins
      if inedi>tI.shape[0]:
        inedi=tI.shape[0]
        insti=inedi-ins
      if inedj>tI.shape[1]:
        inedj=tI.shape[1]
        instj=inedj-ins
      # print(insti,inedi,instj,inedj)
      small_pmap=np.zeros([1,ins,ins,nclass-1])
      feed_image=np.zeros([nrotate,ins,ins,imageType])

      for j in range(nrotate):
        small_in = tI[insti:inedi,instj:inedj]
        feed_image[j,:,:,:] = np.rot90(small_in, j)

      small_out = sess.run(result,feed_dict={input_img:feed_image, phase: False})

      for j in range(nrotate):
        small_pmap = small_pmap + np.rot90(small_out[j,:,:,:],-j)        
      small_pmap = small_pmap / nrotate

      for i in range(nclass-1):
        pmap[insti:inedi,instj:inedj,i] += np.multiply(small_pmap[0,:,:,i],wI)
        avI[insti:inedi,instj:inedj,i] += wI

  final = np.divide(pmap,avI)
  sfinal = final[hph:hph+oh,hpw:hpw+ow,:]
  return sfinal

def branch(c1_2_conv, c2, c3, c4, c5, c6, c7, c8, nc, phase, scope):
  dout = 16
  with tf.variable_scope(scope):
    with tf.variable_scope('up1'):
        up1_conv = conv_(c1_2_conv,3,nc,dout,phase)
        up1_bn = bn_(up1_conv, phase)
        up1 = relu_(up1_bn)
    
    with tf.variable_scope('up2'):
      up2 = deconvn_(c2,2*nc,dout,1,phase)
  
    with tf.variable_scope('up3'):
      up3 = deconvn_(c3,4*nc,dout,2,phase)

    with tf.variable_scope('up4'):
      up4 = deconvn_(c4,8*nc,dout,3,phase)

    with tf.variable_scope('up5'):
      up5 = deconvn_(c5,8*nc,dout,4,phase)

    with tf.variable_scope('up6'):
      up6 = deconvn_(c6,8*nc,dout,5,phase)

    with tf.variable_scope('up7'):
      up7 = deconvn_(c7,16*nc,dout,6,phase)

    with tf.variable_scope('up8'):
      up8 = deconvn_(c8,16*nc,dout,7,phase)

    with tf.variable_scope('final'):
      f1 = tf.concat([up1, up2, up3, up4, up5, up6, up7, up8],3)
      with tf.variable_scope('final_conv1'):
        f1_conv = conv_(f1,3,dout*8,dout*8,phase)
        f1_bn = bn_(f1_conv,phase)
        f1_relu = relu_(f1_bn)
      with tf.variable_scope('final_conv2'):
        print('6dout')
        output = conv_(f1_relu,1,dout*8,nclass,phase) + tf.concat([tf.constant([0],dtype=tf.float32),bias_variable([nclass-1])],axis = 0)
  return output

def model(input_img,phase,nc,id):
  with tf.device('/gpu:%d' % (gpu)):
    print(f'Using /gpu:{gpu}')
    print("--"*35)
    print("\t Loading Layers")
    with tf.variable_scope('model%d' % (id)):
      with tf.variable_scope('scale1'):
        with tf.variable_scope('conv1'):
          c1_1_conv = conv_(input_img,3,imageType,nc,phase)
          c1_1_bn = bn_(c1_1_conv,phase)
          c1_1_relu = relu_(c1_1_bn)
        with tf.variable_scope('conv2'):
          c1_2_conv = conv_(c1_1_relu,3,nc,nc,phase)
          c1_2_bn = bn_(c1_2_conv,phase)
          c1_2_relu = relu_(c1_2_bn)
    
      with tf.variable_scope('scale2'):
        pool1 = max_pool_2x2(c1_2_relu)     # 1/2, 2nc
        c2 = stack(pool1,nc,2*nc,2,phase)
      with tf.variable_scope('scale3'):
        pool2 = max_pool_2x2(c2)            # 1/4, 4nc
        c3 = stack(pool2,2*nc,4*nc,2,phase)
      with tf.variable_scope('scale4'):
        pool3 = max_pool_2x2(c3)            # 1/8, 8nc 
        c4 = stack(pool3,4*nc,8*nc,2,phase)
      with tf.variable_scope('scale5'):
        pool4 = max_pool_2x2(c4)            # 1/16, 8nc
        c5 = stack(pool4,8*nc,8*nc,2,phase) 

      with tf.variable_scope('scale6'):
        pool5 = max_pool_2x2(c5)            # 1/32, 8nc
        c6 = stack(pool5,8*nc,8*nc,2,phase)
      with tf.variable_scope('scale7'):
        pool6 = max_pool_2x2(c6)            # 1/64, 16nc
        c7 = stack(pool6,8*nc,16*nc,2,phase)
      with tf.variable_scope('scale8'):
        pool7 = max_pool_2x2(c7)            # 1/128, 16nc
        c8 = stack(pool7,16*nc,16*nc,2,phase) 
  
      scores = []
      results = []

      cur_score = branch(c1_2_relu, c2, c3, c4, c5, c6, c7, c8, nc, phase, 'branch%d' % (1))
      cur_result = tf.nn.softmax(cur_score)
      scores.append(cur_score)
      results.append(cur_result)

  return scores, results
  
def main(_):
   # check for wandb mode
  wb = FLAGS.wb

  # initialize W&B
  if wb != 'None':
    wandb.init(project=wb, entity="nsapkota")


  #cur_dir_prefix, train_dims, train_idx = load_data(imgdir)
  cur_dir_prefix, train_dims, train_idx, train_stats = load_data(imgdir)
  
  # pretrained_model = FLAGS.model
  nc = 32
  
  input_img = tf.placeholder(tf.float32, [None,ins,ins,imageType])
  phase = tf.placeholder(tf.bool, name='phase')
  output_gt = tf.placeholder(tf.int32, [None, ins, ins])
  lr = tf.placeholder(tf.float32, name='lr')

  with tf.variable_scope('dummy'):
    scores, results = model(input_img,phase,nc,0)

    with tf.device('/gpu:%d' % (gpu)):
      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = scores[0],labels = output_gt ,name="entropy"))
      #tf.summary.scalar('loss%d' %(i),loss)
  
      cross_entropy = loss

      #gradient = tf.gradients(cross_entropy, tf.trainable_variables())
      #optimizer = AMSGradOptimizer(learning_rate=lr)

      with tf.variable_scope('result'):
        result = results[0][:,:,:,1:nclass]
      for i in range(1,nclass):
        tf.summary.image('result_%d' % (i), tf.expand_dims(result[:,:,:,i-1],axis=-1))

      tf.summary.image('input',input_img,batch_size)
      tf.summary.image('output_gt',tf.expand_dims(tf.cast(output_gt,tf.float32),-1),batch_size)
      tf.summary.scalar('error',cross_entropy)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement=True
  sess = tf.Session(config=config)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(logdir,
                                       sess.graph)
  saver = tf.train.Saver(max_to_keep=100000000)

  if (len(pretrained_model)==0):
    print('--'*35)
    print('Initializing model from scratch')
    print('--'*35)
    sess.run(tf.global_variables_initializer())
  else:
    print('--'*35)
    print(f'Loading pretrained model: {pretrained_model}')
    print('--'*35)
    saver = tf.train.import_meta_graph(f'{pretrained_model}.meta')
    saver.restore(sess, pretrained_model)
    sess.run(tf.variables_initializer([x for x in tf.global_variables() if 'Adam' in x.name]))

  total_parameters = 0
  for variable in tf.trainable_variables():
    shape = variable.get_shape()
    variable_parametes = 1
    for dim in shape:
        variable_parametes *= dim.value
    total_parameters += variable_parametes
  print(f'total parameters of the model: {total_parameters}')
  print('--'*35)

  # training loop
  cur_lr = 5e-4
  for i in range(1, iter_num+1):
    train_batch, label_batch = get_batch(cur_dir_prefix,train_dims,train_idx,train_stats, batch_size)
    #if i==10000:
    #  cur_lr = 5e-5
    cur_lr = 5e-4 * math.pow(1-i/iter_num, 0.9)
    summary,error,lr_val,_ = sess.run([merged,cross_entropy,lr,train_step], feed_dict={
          input_img:train_batch, output_gt: label_batch, phase: True, lr: cur_lr})
    # print(i,error,lr_val)
    if i==1 or i%25==0: 
      print(f'iteration: {i}/{iter_num}, error: {error:.4f}, lr: {lr_val:.8f}')

    if wb != 'None':
      wandb.log({"error":error})

    # train_writer.add_summary(summary, i)
    #if (i==5000) or ((i > 10000-1) and (i%10000 == 0)):
    if (i%5000 == 0) or i == iter_num:                    # change to 5000
      # checkpoint_name = os.path.join(modeldir, str(i) + '.ckpt')
      checkpoint_name = f'{modeldir}/{age_group}_{dt}/{i}.ckpt'
      print('Saving model at: ' + checkpoint_name)
      saver.save(sess,checkpoint_name)


  end = int(time.time()) 
  d = divmod(end-start,86400) 
  h = divmod(d[1],3600) 
  m = divmod(h[1],60) 
  s = m[1] 

  print("--"*35)
  print("Training ended !! Total Time: %d days, %d hr, %d min, %d s" % (d[0],h[0],m[0],s))
  print("--"*35)

  
if __name__ == '__main__':  
  parser = argparse.ArgumentParser()
  parser.add_argument('-wb', type=str, default='None',
                      help='run experiments with wandb')
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
