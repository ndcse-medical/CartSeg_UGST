from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import math
import numpy as np
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#import tensorflow as tf
import tensorflow.compat.v1 as tf

# next two lines are needed if TF2.0 is 
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()



from absl import app

import skimage
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import rotate
from scipy import ndimage

from datetime import datetime
import yaml
import logging
import wandb
logging.getLogger('tensorflow').disabled = True
tf.logging.set_verbosity(tf.logging.WARN)

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
batch_size = cnf["TEST"]["batch_size"]
logdir = cnf["TEST"]["logdir"]
resultdir = cnf["TEST"]["resultdir"]
imgdir = cnf["TEST"]["imgdir"]
trained_model = cnf["TEST"]["trained_model"]
test_stack = cnf["TEST"]["test_stack"]
iter_num = cnf["TEST"]["iter_num"]
result_temp_dir = cnf["TEST"]["result_temp_dir"]
threshold = cnf["TEST"]["threshold"]
results_id = cnf["TEST"]['results_id']

imageType = 1
ous = 512 #640
ins = ous
interv = 200
nclass = 2
# axis = 0 # which direction   

if not os.path.isdir(result_temp_dir):
  os.makedirs(result_temp_dir)

def imread3Den(d):
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

  meanI = np.mean(I)
  stdI = np.std(I)
  I = (I-meanI)/stdI
  # print('norm:',I.max(),I.min(),np.mean(I),np.std(I))
  return I

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

def get_image(cur_dir_prefix,train_dims,train_idx,train_stats):
  #  which stack?
  i_stk = random.randint(0, len(cur_dir_prefix)-1)
  cur_dir = cur_dir_prefix[i_stk]
  di = os.path.join(imgdir, 'img/', cur_dir)
  dm = os.path.join(imgdir, 'mask_2/', cur_dir)
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
    print('--'*35)
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
      # print('good score')
      scores.append(cur_score)
      results.append(cur_result)

  return scores, results


def convert_():
    di = os.path.join(imgdir, test_stack)
    print('--'*35)
    print('Combining Results for: ', di)
    print("Reading intermediate results from: ",result_temp_dir)
    print('--'*35)

    imgs = []
    for file in os.listdir(di):
      if file.endswith(".png"):
        imgs.append(file)
    imgs.sort()
    tempI = imread(os.path.join(di,imgs[0]))
    z = len(imgs)
    h = tempI.shape[0]
    w = tempI.shape[1]
    #I_arg = np.zeros((z,h,w,3))
    I_fuse = np.zeros((z,h,w))
    newshape = (1.0 * np.array([z, h, w])).astype(int)
    newshape = tuple(newshape)


    d_a0 = f'id_{results_id}_dt_{dt}_iter{iter_num}_D0_c1_{test_stack}'
    d_a1 = f'id_{results_id}_dt_{dt}_iter{iter_num}_D1_c1_{test_stack}'
    d_a2 = f'id_{results_id}_dt_{dt}_iter{iter_num}_D2_c1_{test_stack}'

    for idx in range(z):
      flnm = test_stack + '_{:04}.png'.format(idx)
      print(f'\t converting to 2d image, step {idx+1}/{z}')
      I_fuse[idx,:,:] = np.maximum(skimage.img_as_float(imread(os.path.join(result_temp_dir,d_a0,flnm))), skimage.img_as_float(imread(os.path.join(result_temp_dir,d_a1,flnm))) , skimage.img_as_float(imread(os.path.join(result_temp_dir,d_a2,flnm))))
      #I_fuse[idx,:,:] = skimage.img_as_float(imread(os.path.join(result_temp_dir,d_a0,flnm))) + skimage.img_as_float(imread(os.path.join(result_temp_dir,d_a1,flnm))) + skimage.img_as_float(imread(os.path.join(result_temp_dir,d_a2,flnm)))

    img_save_dir = f'{result_temp_dir}/id_{results_id}_dt_{dt}_iter{iter_num}_{test_stack}'

    if not os.path.isdir(img_save_dir):
      os.makedirs(img_save_dir)
    for z in range(I_fuse.shape[0]):
      i_name = test_stack + '_{slice:04}.png'.format(slice = z)
      x = (I_fuse[z,:,:] * 255).astype('uint8')
      imsave(os.path.join(img_save_dir, i_name),x)
    print('--'*35)
    print('Combining completed for: ',test_stack)
    print('--'*35)

def remove_connected_components():
  di = os.path.join(imgdir, test_stack)
  # combined_res_dir = f'{result_temp_dir}/id_{results_id}_dt_{dt}_iter{iter_num}_{test_stack}'
  combined_res_dir = f'ACH-Results-phase1-sept2021/ACH-E14.5/{test_stack}'


  print('--'*35)
  print('Removing connected components for: ', test_stack)
  print("Reading combined results from: ",combined_res_dir)
  print('--'*35)

  imgs = []
  for file in os.listdir(di):
    if file.endswith(".png"):
      imgs.append(file)
  imgs.sort()
  tempI = imread(os.path.join(di,imgs[0]))
  z = len(imgs)
  h = tempI.shape[0]
  w = tempI.shape[1]
  I = np.zeros((z,h,w))
  
  print(f'Removing connected components . . . ')

  for idx in range(z):
    flnm = test_stack + '_{:04}.png'.format(idx)
    tempI = skimage.img_as_float(imread(os.path.join(combined_res_dir,flnm)))
    I[idx,:,:] = tempI

  I[I<threshold/255.0]=0
  I[I>0]=1

  #I = ndimage.morphology.binary_erosion(I).astype(I.dtype)

  label_img, cc_num = ndimage.label(I)
  CC = ndimage.find_objects(label_img)
  cc_areas = ndimage.sum(I, label_img, range(cc_num+1))

  area_mask = (cc_areas < 20000)
  label_img[area_mask[label_img]] = 0
  label_img[label_img>0] = 1

  #img_save_dir = os.path.join(resultdir,'iter{}_'.format(Iter*500)+ test_stack)
  output_dir = f'{resultdir}/{test_stack[0:4]}_id_{results_id}/id_{results_id}_dt_{dt}_iter{iter_num}_stack_{test_stack}'
  if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
  print("--"*35)
  print('Saving results in:', output_dir)
  print("--"*35)
  for z in range(label_img.shape[0]):
    i_name = test_stack + '_{slice:04}.png'.format(slice = z)
    x = (label_img[z,:,:] * 255).astype('uint8')
    print(f'\t Saving final image, step: {z+1}/{label_img.shape[0]}')
    imsave(os.path.join(output_dir, i_name),x)

  print('--'*35)
  print('All done for: ',test_stack)
  print('--'*35)

def main(_):

  # check for wandb mode
  wb = FLAGS.wb

  # initialize W&B
  if wb != 'None':
    wandb.init(project=wb, entity="nsapkota")

  nc = 32
  
  input_img = tf.placeholder(tf.float32, [None,ins,ins,imageType])
  phase = tf.placeholder(tf.bool, name='phase')
  output_gt = tf.placeholder(tf.int32, [None, ins, ins])
  lr = tf.placeholder(tf.float32, name='lr')

  with tf.variable_scope('dummy'):
    scores, results = model(input_img,phase,nc,0)

    with tf.device('/gpu:%d' % (gpu)):
      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = scores[0],labels = output_gt ,name="entropy"))  
      cross_entropy = loss

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
  # train_writer = tf.summary.FileWriter(logdir,
  #                                      sess.graph)
  saver = tf.train.Saver(max_to_keep=100000000)

  saver = tf.train.import_meta_graph(f'{trained_model}.meta') # edit
  saver.restore(sess, trained_model )
  sess.run(tf.variables_initializer([x for x in tf.global_variables() if 'Adam' in x.name]))
  print('--'*35)
  print(f'Loading trained model from {trained_model}')
  print('--'*35)

  total_parameters = 0
  for variable in tf.trainable_variables():
    shape = variable.get_shape()
    variable_parametes = 1
    for dim in shape:
        variable_parametes *= dim.value
    total_parameters += variable_parametes
  print('--'*35)
  print(f'total parameters of the modle: {total_parameters}')
  print('--'*35)

  if (len(imgdir)>0):
    #for cur_dir in os.listdir(imgdir):
      #iter_num = 20000    # change this
    cur_dir = test_stack # change this
    testd = os.path.join(imgdir, cur_dir)
    print('--'*35)
    print(f'loading pictures from {testd}')
    print('--'*35)

    if os.path.isdir(testd):
      print('--'*35)
      print(f'Processing Images ...')
      print('--'*35)

      testI3D = imread3Den(testd)
      pmap = np.zeros([testI3D.shape[0],testI3D.shape[1],testI3D.shape[2],nclass-1])
      
      # axis = 0 
      print("Operating on axis = 0")
      for z in range(testI3D.shape[0]):
        print(f'\t step: {z+1}/{testI3D.shape[0]}')
        pmap[z,:,:,:] = model_apply2d(sess,result,input_img,phase,testI3D[z,:,:,:] ,ins)  

      print('--'*35)
      print("Saving axis = 0 results")
      for z in range(testI3D.shape[0]): 
        for k in range(nclass-1):
          pmap_img = np.zeros([testI3D.shape[1],testI3D.shape[2]],dtype='uint8')
          pmap_img[:,:] = pmap[z,:,:,k]*255
          #print(pmap_img.max())
          #print(pmap_img.min())
          img_save_dir = f'{result_temp_dir}/id_{results_id}_dt_{dt}_iter{iter_num}_D0_c{k+1}_{cur_dir}'
          if not os.path.isdir(img_save_dir):
            os.makedirs(img_save_dir)
          imsave(os.path.join(img_save_dir, '{}_{:04}.png'.format(cur_dir, z)), pmap_img)
          print(f'\t step {z+1}/{testI3D.shape[0]}')

      # axis = 1
      print('--'*35)
      print("\nOperating on axis = 1")
      for z in range(testI3D.shape[1]):
        print(f'\t Processing, step: {z+1}/{testI3D.shape[1]}')
        pmap[:,z,:,:] = model_apply2d(sess,result,input_img,phase,testI3D[:,z,:,:] ,ins)  
    
      print('--'*35)
      print("Saving axis = 0 results")
      for z in range(testI3D.shape[0]):            
        for k in range(nclass-1):
          pmap_img = np.zeros([testI3D.shape[1],testI3D.shape[2]],dtype='uint8')
          pmap_img[:,:] = pmap[z,:,:,k]*255
          #print(pmap_img.max())
          #print(pmap_img.min())
          img_save_dir = f'{result_temp_dir}/id_{results_id}_dt_{dt}_iter{iter_num}_D1_c{k+1}_{cur_dir}'
          if not os.path.isdir(img_save_dir):
            os.makedirs(img_save_dir)
          imsave(os.path.join(img_save_dir, '{}_{:04}.png'.format(cur_dir, z)), pmap_img)
          print(f'\t step {z+1}/{testI3D.shape[0]}')

      # axis = 2
      print('--'*35)
      print("\nOperating on axis = 2")
      for z in range(testI3D.shape[2]):
        print(f'\t Processing, step: {z+1}/{testI3D.shape[2]}')
        pmap[:,:,z,:] = model_apply2d(sess,result,input_img,phase,testI3D[:,:,z,:] ,ins)  

      print('--'*35)
      print("Saving axis = 0 results")

      for z in range(testI3D.shape[0]):            
            for k in range(nclass-1):
              pmap_img = np.zeros([testI3D.shape[1],testI3D.shape[2]],dtype='uint8')
              pmap_img[:,:] = pmap[z,:,:,k]*255
              #print(pmap_img.max())
              #print(pmap_img.min())
              #img_save_dir = result_temp_dir + '/iter' + str(iter_num) + '_D' + str(2) + '_c' + str(k+1) + '_' + cur_dir
              img_save_dir = f'{result_temp_dir}/id_{results_id}_dt_{dt}_iter{iter_num}_D2_c{k+1}_{cur_dir}'
              if not os.path.isdir(img_save_dir):
                os.makedirs(img_save_dir)
              imsave(os.path.join(img_save_dir, '{}_{:04}.png'.format(cur_dir, z)), pmap_img)
              print(f'\t step {z+1}/{testI3D.shape[0]}')

  print('--'*35)
  print("\n--------------- 3 Directional Segmentation Completed -----------------")    
  print('--'*35)
 
  # call convert to 2d images function
  convert_()

  # remove connected components
  remove_connected_components()
  
if __name__ == '__main__':  
  parser = argparse.ArgumentParser()
  parser.add_argument('-wb', type=str, default='None',
                      help='run experiments with wandb')
  FLAGS, unparsed = parser.parse_known_args()
  # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
