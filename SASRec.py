import json
import tensorflow as tf
import random
import numpy as np
from tqdm import tqdm
import time
from collections import defaultdict


time.sleep(3600)

def process_seq(seq,neg_sample_seq,seq_len,neg_sample_num,place_idx):
    inp = ([place_idx[i[0]] for i in seq[:-1]] + [0]*max(0,seq_len-len(seq)+1))[-seq_len:]
    real = ([place_idx[i[0]] for i in seq[1:]] + [0]*max(0,seq_len-len(seq)+1))[-seq_len:]
    if neg_sample_seq != None:
        neg = list(set(random.sample(list(place_idx.values()),neg_sample_num*seq_len+len(neg_sample_seq)))-set([place_idx[i[0]] for i in neg_sample_seq]))[:neg_sample_num*seq_len]
        random.shuffle(neg)
        neg = [neg[i*neg_sample_num:(i+1)*neg_sample_num] for i in range(seq_len)]
    else:               
        neg = list(set(random.sample(list(place_idx.values()),(1+neg_sample_num)*seq_len)))[:(1+neg_sample_num)*seq_len]
        random.shuffle(neg)
        neg = [list(set(neg[i*(neg_sample_num+1):(i+1)*(neg_sample_num+1)])-set([real[i]]))[:neg_sample_num] for i in range(seq_len)]
    neg = [neg[j]+[real[j]] for j in range(seq_len)]
    pos = (([neg_sample_num]*(len(seq)-1))+([0]*max(0,seq_len-len(seq)+1)))[-seq_len:]
    return inp,real,neg,pos

def process_seq2(seq,neg_sample_seq,seq_len,place_idx):
    inp = ([place_idx[i[0]] for i in seq[:-1]] + [0]*max(0,seq_len-len(seq)+1))[-seq_len:]
    real = ([place_idx[i[0]] for i in seq[1:]] + [0]*max(0,seq_len-len(seq)+1))[-seq_len:]
    neg_all = list(set(place_idx.values())-set([place_idx[i[0]] for i in seq]))
    neg_all = [0]*(len(place_idx)-len(neg_all)-1)+neg_all
    neg = [neg_all+[real[j]] for j in range(seq_len)]
    pos = (([1]*(len(seq)-1))+([0]*max(0,seq_len-len(seq)+1)))[-seq_len:]
    return inp,real,neg,pos

def process_seq_test(seq,neg_sample_seq,seq_len,neg_sample_num,place_idx):
    inp = ([place_idx[i[0]] for i in seq[:-1]] + [0]*max(0,seq_len-len(seq)+1))[-seq_len:]
    real = ([place_idx[i[0]] for i in seq[1:]] + [0]*max(0,seq_len-len(seq)+1))[-seq_len:]
    neg = [[1 for i in range(neg_sample_num)]+[real[j]] for j in range(seq_len)]
    neg[min(seq_len-1,len(seq)-2)] = random.sample(list(set(list(place_idx.values()))-set([real[min(seq_len-1,len(seq)-2)]])),neg_sample_num) + [real[min(seq_len-1,len(seq)-2)]]
    pos = (([1]*(len(seq)-1))+([0]*max(0,seq_len-len(seq)+1)))[-seq_len:]
    return inp,real,neg,pos



def constract_dataset(user_place_multiple,seq_len,neg_sample_num_train,neg_sample_num_val,neg_sample_num_test):
    place_idx = {}
    for v in user_place_multiple.values():
        for p in v:
            if p[0] not in place_idx:
                place_idx[p[0]] = len(place_idx)+1
    train_set = []
    train_candidate = []
    train_real = []
    train_pos = []
    train_target = []

    val_set = []
    val_candidate = []
    val_real = []
    val_pos = []
    val_target = []

    test_set = []
    test_candidate = []
    test_real = []
    test_target = []

    random.seed(29)
    for user,seq in tqdm(user_place_multiple.items()):
        if neg_sample_num_test!= None:
            inp,real,neg,pos = process_seq_test(seq,None,seq_len,neg_sample_num_test,place_idx)
        else:
            inp,real,neg,pos = process_seq2(seq,None,seq_len,place_idx)
        test_set.append(inp)
        test_real.append(real)
        test_candidate.append(neg)
        test_target.append(min(seq_len-1,len(seq)-2))
        
        if neg_sample_num_val!= None:
            inp,real,neg,pos = process_seq(seq[:-1],seq[:-1],seq_len,neg_sample_num_val,place_idx)
        else:
            inp,real,neg,pos = process_seq2(seq[:-1],seq[:-1],seq_len,place_idx)
        val_set.append(inp)
        val_real.append(real)
        val_candidate.append(neg)
        val_pos.append(pos)
        val_target.append(min(seq_len-1,len(seq)-3))
        
        if neg_sample_num_train!= None:
            inp,real,neg,pos = process_seq(seq[:-2],seq[:-2],seq_len,neg_sample_num_train,place_idx)
        else:
            inp,real,neg,pos = process_seq2(seq[:-2],seq[:-2],seq_len,place_idx)
        train_set.append(inp)
        train_real.append(real)
        train_candidate.append(neg)
        train_pos.append(pos)
        train_target.append(min(seq_len-1,len(seq)-4))
        
    return tf.constant(train_set), tf.constant(train_candidate),tf.constant(train_real),tf.constant(train_pos),\
           tf.constant(val_set), tf.constant(val_candidate),tf.constant(val_real),tf.constant(val_pos),\
           tf.constant(test_set), tf.constant(test_candidate),tf.constant(test_real),place_idx,\
           test_target,val_target,train_target


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  

def scaled_dot_product_attention(q, k, v, pad_mask,look_ahead_mask):

    matmul_qk = tf.matmul(q, k, transpose_b=True) 

    
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    
    if pad_mask is not None:
        scaled_attention_logits += (pad_mask * -1e9)
    if look_ahead_mask is not None:
        scaled_attention_logits += (look_ahead_mask * -1e9)
    

   
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  

    output = tf.matmul(attention_weights, v)  
    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model) 
    ])

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, pad_mask,look_ahead_mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  
        k = self.wk(k) 
        v = self.wv(v)  

        q = self.split_heads(q, batch_size)  
        k = self.split_heads(k, batch_size)  
        v = self.split_heads(v, batch_size) 

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, pad_mask,look_ahead_mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model)) 

        output = self.dense(concat_attention) 

        return output, attention_weights



class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads,dff, rate=0.2):
        super(AttentionLayer, self).__init__()

        self.mha = SelfAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, pad_mask,look_ahead_mask):

        attn_output, _ = self.mha(x, x, x, pad_mask,look_ahead_mask) 
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output) 

        ffn_output = self.ffn(out1) 
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output) 

        return out2
    
class StackedLayers(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.2):
        super(StackedLayers, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.maximum_position_encoding = maximum_position_encoding
        self.input_vocab_size = input_vocab_size

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = tf.keras.layers.Embedding(maximum_position_encoding, d_model)

        self.enc_layers = [AttentionLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, pad_mask,look_ahead_mask,candidates):

        seq_len = tf.shape(x)[1]

        x = self.embedding(x)  
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding(tf.range(self.maximum_position_encoding))
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, pad_mask,look_ahead_mask)
        logits = tf.squeeze(tf.linalg.matmul(x[:,:,tf.newaxis,:],self.embedding(candidates),transpose_b=True),axis=-2)
        return tf.math.sigmoid(logits)  
    
    def pred(self, x,pad_mask,look_ahead_mask,candidates,target_pos):
        x = self.embedding(x)  
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding(tf.range(self.maximum_position_encoding))
        x = self.dropout(x, training=False)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, False, pad_mask,look_ahead_mask)
            
        target_pos = [target_pos[i]+x.shape[1]*i for i in range(len(target_pos))]
        x = tf.reshape(x,[-1,x.shape[-1]])
        x = tf.nn.embedding_lookup(x,target_pos)
        
        candidates = tf.reshape(candidates,[-1,candidates.shape[-1]])
        candidates = tf.nn.embedding_lookup(candidates,target_pos)
        
        candidate_mask = tf.math.logical_not(tf.math.equal(candidates, 0))
        logits = tf.squeeze(tf.linalg.matmul(x[:,tf.newaxis,:],self.embedding(candidates),transpose_b=True),axis=-2)
        candidate_mask = tf.cast(candidate_mask,dtype=logits.dtype)
        return tf.math.sigmoid(logits)*candidate_mask   
    

loss_pos = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')  
loss_neg = tf.keras.losses.CategoricalCrossentropy(reduction='none') 

def loss(pred,real_pos,candidates):
    term1 = loss_pos(real_pos,pred)
    mask = tf.math.logical_not(tf.math.equal(real_pos, 0))
    term1 = -tf.math.log(pred[:,:,-1])

    mask = tf.cast(mask, dtype=term1.dtype)
    term1 *= mask
    
    
    candidate_mask = tf.math.logical_not(tf.math.equal(candidates[:,:,:-1], 0))
    candidate_mask = tf.cast(candidate_mask,dtype=pred.dtype)
    term2 = -tf.reduce_sum(candidate_mask*tf.math.log((tf.constant(1,dtype=pred.dtype)-pred)[:,:,:-1]),axis=-1)
    term2 *= mask
    
    return tf.reduce_sum(term1+term2)/tf.reduce_sum(mask)
    
def accuracy_hitk(pred,k):
    correct_index = pred.shape[-1]-1
    acc = [int(correct_index in i) for i in tf.nn.top_k(pred,k).indices]
    return acc



time_stamp = str(int(time.time()))
fconfig = open('mv-config-%s.txt'%time_stamp,'w')
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
# fconfig.write('user:%s item:%s rate:%s\n'%(len(user_place_pmulti),len(p5_set),sum([len(i) for i in user_place_pmulti.values()])))
# ui_dic = user_place_pmulti

f = open('../../ml-1m/ratings.dat')
ui_dic = defaultdict(set)
for l in f:
    l_list = l.strip().split('::')
    ui_dic[int(l_list[0])].add(((int(l_list[1])),int(l_list[3])))
f.close()

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

seed=10191129
seq_len = 15
d = 16
num_epoch = 300
batch_size = 128
val_batch_size = 1000
k = 10
drop_rate = 0.5
num_layers = 4
neg_train_num=None
neg_eval_num=None
neg_test_num=100
clipnorm=0.01

fconfig.write(f'seed:{seed}\n seq_len:{seq_len}\n d:{d}\n num_epoch:{num_epoch}\n batch_size:{batch_size}\n k:{k}\n drop_rate:{drop_rate}\n num_layers:{num_layers}\n neg_train_num:{neg_train_num}\n neg_eval_num:{neg_eval_num}\n neg_test_num:{neg_test_num}\n')
fconfig.close()

train_set,train_candidate,train_real,train_pos,\
val_set,val_candidate,val_real,val_pos,\
test_set,test_candidate,test_real,place_idx,\
test_target,val_target,train_target = constract_dataset(new_ui_ic,seq_len,neg_train_num,neg_eval_num,neg_test_num)

train_pad_mask = create_padding_mask(train_set)
val_pad_mask = create_padding_mask(val_set)
test_pad_mask = create_padding_mask(test_set)
look_ahead_mask = create_look_ahead_mask(seq_len)
tf.random.set_seed(seed)
encoder = StackedLayers(num_layers=num_layers, d_model=d, num_heads=1,
                         dff=d, input_vocab_size=len(place_idx)+1,
                         maximum_position_encoding=seq_len,rate=drop_rate)

# optimizer = tf.keras.optimizers.SGD(learning_rate)
# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
optimizer = tf.keras.optimizers.Adam(clipnorm=clipnorm)
# checkpoint_path = "./SASR-mv-ckpt-%s/train"%(time_stamp)
checkpoint_path = './SASR-mv-ckpt-1638287748/train'
f = open(checkpoint_path.split('/')[1]+'.txt','a')

ckpt = tf.train.Checkpoint(encoder=encoder,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')
    
for epoch in range(num_epoch):
    acc_train = []
    num_batch = (train_set.shape[0]//batch_size)+int(train_set.shape[0]%batch_size != 0)
    for batch in range(num_batch):
        with tf.GradientTape() as tape:
            pred = encoder(val_set[batch*batch_size:(batch+1)*batch_size], training=True, pad_mask=val_pad_mask[batch*batch_size:(batch+1)*batch_size],look_ahead_mask=look_ahead_mask,candidates=val_candidate[batch*batch_size:(batch+1)*batch_size])
            l = loss(pred,val_pos[batch*batch_size:(batch+1)*batch_size],val_candidate[batch*batch_size:(batch+1)*batch_size])
        gradients = tape.gradient(l, encoder.trainable_variables)
        optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))
        print('Epoch: %s, Batch: %s/%s, loss: %s'%(epoch+1,batch+1,num_batch,float(l)))
        f.write('Epoch: %s, Batch: %s/%s, loss: %s\n'%(epoch+1,batch+1,num_batch,float(l)))
    pred_test = encoder.pred(test_set, pad_mask=test_pad_mask,look_ahead_mask=look_ahead_mask,candidates=test_candidate,target_pos=test_target)
    acc_test = accuracy_hitk(pred_test,k) 
    print('Epoch: %s, Test top%s acc:%s'%(epoch+1,k,sum(acc_test)/len(acc_test)))
    f.write('Epoch: %s, Test top%s acc:%s\n'%(epoch+1,k,sum(acc_test)/len(acc_test)))
    ckpt_save_path = ckpt_manager.save()
    print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')
    f.write(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}\n')

f.close()
