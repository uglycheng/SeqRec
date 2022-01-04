import tensorflow as tf
import numpy as np
from collections import defaultdict
import random
from tqdm import tqdm
import time
import json


f = open('./ml-1m/ratings.dat')
ui_dic = defaultdict(set)
for l in f:
    l_list = l.strip().split('::')
    ui_dic[int(l_list[0])].add(((int(l_list[1])),int(l_list[3])))
f.close()


# user_place = json.loads(open('./user_place.json').read())
# place_user = json.loads(open('./place_user.json').read())
# user_place_pmulti = {}
# p5_set = set([])
# for u,v in user_place.items():
#     valid_ps = []
#     for p in v:
#         if len(place_user[p[0]])>=10:   
#             valid_ps.append(p)
#     if len(valid_ps)>=10: 
#         user_place_pmulti[u] = sorted(valid_ps,key=lambda x:x[1])
#         p5_set = p5_set.union(set([pp[0] for pp in valid_ps]))
# ui_dic = user_place_pmulti

target_len = 15
iset = set([])
new_ui_ic = {}
for k,v in ui_dic.items():
    v = sorted(list(v),key=lambda x:x[1],reverse=True)
    newv = [[-1,-1]]
    for pp in v:
        if pp[1]!=newv[-1][1]:
            newv.append(pp)
        if len(newv[1:]) == target_len+3:
            break
    if len(newv[1:])>=4:
        newv = newv[1:]
        newv.reverse()
        new_ui_ic[k] = newv
        iset = iset.union(set([i[0] for i in newv]))
print(len(new_ui_ic),len(iset),sum([len(i) for i in new_ui_ic.values()]))


def construct_data(iset,ui_dic,neg_sample_num,neg_sample_num_val,neg_sample_num_test):
    iset = list(iset)
    idic = {}
    for i in iset:
        idic[i] = len(idic)+1
    udic = {}
    for u in ui_dic:
        udic[u] = len(udic)+1
    user_train = []
    last_train = []
    next_train = []
    user_val = []
    last_val = []
    next_val = []
    user_test = []
    last_test = []
    next_test = []

    for u,v in tqdm(ui_dic.items()):
        for j in range(len(v)-3):
            i = v[j][0]
            user_train.append([udic[u]])
            last_train.append([idic[i]])
            if neg_sample_num==None:
                neg_rand = iset
            else:
                neg_rand = random.sample(iset,min(neg_sample_num+1,len(iset)))
                
            n = list(set(neg_rand)-set([v[j+1][0]]))[:neg_sample_num]+[v[j+1][0]]
            n = [idic[k] for k in n]
            next_train.append(n)

        user_val.append([udic[u]])
        last_val.append([idic[v[-3][0]]])
        if neg_sample_num_val==None:
            neg_rand = iset
        else:
            neg_rand = random.sample(iset,min(neg_sample_num_val+1,len(iset)))
        n = list(set(neg_rand)-set([v[-2][0]]))[:neg_sample_num_val]+[v[-2][0]]
        n = [idic[k] for k in n]
        next_val.append(n)

        user_test.append([udic[u]])
        last_test.append([idic[v[-2][0]]])
        if neg_sample_num_test==None:
            neg_rand = iset
        else:
            neg_rand = random.sample(iset,min(neg_sample_num_test+1,len(iset)))
        n = list(set(neg_rand)-set([v[-1][0]]))[:neg_sample_num_test]+[v[-1][0]]
        n = [idic[k] for k in n]
        next_test.append(n)

    return tf.constant(user_train), tf.constant(last_train), tf.constant(next_train),\
           tf.constant(user_val), tf.constant(last_val), tf.constant(next_val),\
           tf.constant(user_test), tf.constant(last_test), tf.constant(next_test)



def tfshuffle(u,l,n):
    x = tf.concat([u,l,n],axis=-1)
    print(x.shape)
    x = tf.random.shuffle(x)
    print(x.shape)
    return x[:,:1],x[:,1:2],x[:,2:]


class FPMC(tf.keras.layers.Layer):
    def __init__(self,user_size,item_size,d_model):
        super(FPMC, self).__init__()
        self.user_emb = tf.keras.layers.Embedding(user_size, d_model,embeddings_regularizer='l2')
        self.last_item_emb = tf.keras.layers.Embedding(item_size, d_model,embeddings_regularizer='l2')
        self.next_item_emb = tf.keras.layers.Embedding(item_size, d_model,embeddings_regularizer='l2')
        
    def call(self,user,last,candidates):
        u = self.user_emb(user)
        last = self.last_item_emb(last)
        candidates = self.next_item_emb(candidates)
        score = tf.matmul(u,last,transpose_b=True)+tf.matmul(u,candidates,transpose_b=True)+tf.matmul(last,candidates,transpose_b=True)
        return tf.squeeze(score)
    
def loss(pred):
    real = pred[:,-1]
    real = real[:,tf.newaxis]
    neg = pred[:,:-1]
    l = tf.math.log(1+tf.math.exp(neg-real))
    return tf.reduce_mean(l)

def accuracy_hitk(pred,k):
    correct_index = pred.shape[-1]-1
    acc = [int(correct_index in i) for i in tf.nn.top_k(pred,k).indices]
    return acc 

tf.random.set_seed(29)
user_train,last_train,next_train,\
user_val,last_val,next_val,\
user_test,last_test,next_test = construct_data(iset,new_ui_ic,None,None,100)

user_val = tf.concat([user_train,user_val],axis=0)
last_val = tf.concat([last_train,last_val],axis=0)
next_val = tf.concat([next_train,next_val],axis=0)
user_val,last_val,next_val = tfshuffle(user_val,last_val,next_val)
print(user_val.shape,last_val.shape,next_val.shape)

d = 16
num_epoch = 1000
batch_size = 128
k = 10
drop_rate = 0.5
num_layers = 4

tf.random.set_seed(29)
fpmc = FPMC(user_size=len(new_ui_ic)+1,item_size=len(iset)+1,d_model = d)

# optimizer = tf.keras.optimizers.SGD(learning_rate)
# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
optimizer = tf.keras.optimizers.Adam()
checkpoint_path = "./FMPC-ckpt-%s/train"%(int(time.time()))
# checkpoint_path = './SASR-movie-ckpt-1638148318/train'
f = open(checkpoint_path.split('/')[1]+'.txt','a')

ckpt = tf.train.Checkpoint(fpmc=fpmc,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')
    
for epoch in range(num_epoch):
    num_batch = (user_val.shape[0]//batch_size)+int(user_val.shape[0]%batch_size != 0)
    for batch in range(num_batch):
        with tf.GradientTape() as tape:
            pred = fpmc(user_val[batch*batch_size:(batch+1)*batch_size],last_val[batch*batch_size:(batch+1)*batch_size],next_val[batch*batch_size:(batch+1)*batch_size])
            l = loss(pred)
        gradients = tape.gradient(l, fpmc.trainable_variables)
        optimizer.apply_gradients(zip(gradients, fpmc.trainable_variables))
        print('Epoch: %s, Batch: %s/%s, loss: %s'%(epoch+1,batch+1,num_batch,float(l)))
        f.write('Epoch: %s, Batch: %s/%s, loss: %s\n'%(epoch+1,batch+1,num_batch,float(l)))
    pred_test = fpmc(user_test,last_test,next_test)
    acc_test = accuracy_hitk(pred_test,k) 
    print('Epoch: %s, Test top%s acc:%s'%(epoch+1,k,sum(acc_test)/len(acc_test)))
    f.write('Epoch: %s, Test top%s acc:%s\n'%(epoch+1,k,sum(acc_test)/len(acc_test)))
    ckpt_save_path = ckpt_manager.save()
    print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')
    f.write(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}\n')

f.close()


