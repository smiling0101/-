#!/usr/bin/env python
# coding: utf-8
# # 基于Transformer的中英文机器翻译
# 机器翻译是利用计算机将一种自然语言(源语言)转换为另一种自然语言(目标语言)的过程。本项目是机器翻译领域主流模型 Transformer 的 PaddlePaddle 实现，包含模型训练，预测以及使用自定义数据等内容。用户可以基于发布的内容搭建自己的翻译模型。
# 
# 
# # 1.资源


# # 2. Transformer 原理解读
import collections
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.fluid import layers

class MultiHeadAttention(nn.Layer):

    Cache = collections.namedtuple("Cache", ["k", "v"])
    StaticCache = collections.namedtuple("StaticCache", ["k", "v"])
    
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 need_weights=False,
                 weight_attr=None,
                 bias_attr=None):   
        super(MultiHeadAttention, self).__init__()
        # 输入的embedding维度
        self.embed_dim = embed_dim
        # key的维度
        self.kdim = kdim if kdim is not None else embed_dim
        # value的维度
        self.vdim = vdim if vdim is not None else embed_dim
        # head的数目
        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        # query
        self.q_proj = Linear(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)
        # key
        self.k_proj = Linear(
            self.kdim, embed_dim, weight_attr, bias_attr=bias_attr)
        # value
        self.v_proj = Linear(
            self.vdim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.out_proj = Linear(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)

    def _prepare_qkv(self, query, key, value, cache=None):
        q = self.q_proj(query)
        q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
        q = tensor.transpose(x=q, perm=[0, 2, 1, 3])

        if isinstance(cache, self.StaticCache):
            # for encoder-decoder attention in inference and has cached
            k, v = cache.k, cache.v
        else:
            k, v = self.compute_kv(key, value)

        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = tensor.concat([cache.k, k], axis=2)
            v = tensor.concat([cache.v, v], axis=2)

        return (q, k, v) if cache is None else (q, k, v, cache)

    def compute_kv(self, key, value):
        k = self.k_proj(key)
        v = self.v_proj(value)
        k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = tensor.transpose(x=v, perm=[0, 2, 1, 3])
        return k, v

    def forward(self, query, key=None, value=None, attn_mask=None, cache=None):
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
       
        q, k, v = self._prepare_qkv(query, key, value, cache)
       

        # scale dot product attention
        product = layers.matmul(
            x=q, y=k, transpose_y=True, alpha=self.head_dim**-0.5)
        if attn_mask is not None:
            # Support bool or int mask
            attn_mask = _convert_attention_mask(attn_mask, product.dtype)
            product = product + attn_mask
        weights = F.softmax(product)
        if self.dropout:
            weights = F.dropout(
                weights,
                self.dropout,
                training=self.training,
                mode="upscale_in_train")

        out = tensor.matmul(weights, v)

        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)

        return out if len(outs) == 1 else tuple(outs)

# ## 2.2 Transformer Encoder
def _convert_attention_mask(attn_mask, dtype):
    if attn_mask is not None and attn_mask.dtype != dtype:
        attn_mask_dtype = convert_dtype(attn_mask.dtype)
        if attn_mask_dtype == 'bool' or 'int' in attn_mask_dtype:
            attn_mask = (paddle.cast(attn_mask, dtype) - 1.0) * 1e9
        else:
            attn_mask = paddle.cast(attn_mask, dtype)
    return attn_mask
def _convert_param_attr_to_list(param_attr, n):
    if isinstance(param_attr, (list, tuple)):
        assert len(param_attr) == n, (
            "length of param_attr should be %d when it is a list/tuple" % n)
        param_attrs = []
        for attr in param_attr:
            if isinstance(attr, bool):
                if attr:
                    param_attrs.append(ParamAttr._to_attr(None))
                else:
                    param_attrs.append(False)
            else:
                param_attrs.append(ParamAttr._to_attr(attr))
        # param_attrs = [ParamAttr._to_attr(attr) for attr in param_attr]
    elif isinstance(param_attr, bool):
        param_attrs = []
        if param_attr:
            param_attrs = [ParamAttr._to_attr(None) for i in range(n)]
        else:
            param_attrs = [False] * n
    else:
        param_attrs = []
        attr = ParamAttr._to_attr(param_attr)
        for i in range(n):
            attr_i = copy.deepcopy(attr)
            if attr.name:
                attr_i.name = attr_i.name + "_" + str(i)
            param_attrs.append(attr_i)
    return param_attrs
class TransformerEncoderLayer(nn.Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 weight_attr=None,
                 bias_attr=None):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerEncoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 2)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 2)

        # multi head attention
        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0])
        
        # feed forward
        self.linear1 = Linear(
            d_model, dim_feedforward, weight_attrs[1], bias_attr=bias_attrs[1])
        self.dropout = Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = Linear(
            dim_feedforward, d_model, weight_attrs[1], bias_attr=bias_attrs[1])
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)
    
    def forward(self, src, src_mask=None, cache=None):
        src_mask = _convert_attention_mask(src_mask, src.dtype)

        residual = src

        #  multi head attention 
        src = self.self_attn(src, src, src, src_mask)
        # 残差连接
        src = residual + self.dropout1(src)
        # Norm
        src = self.norm1(src)
        residual = src

        # feed forward
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # 残差连接
        src = residual + self.dropout2(src)
        # Norm
        src = self.norm2(src)
        return src 

# 2.3 Transformer Decoder
class TransformerDecoderLayer(nn.Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 weight_attr=None,
                 bias_attr=None):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerDecoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 3)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 3)
        # multi head attention
        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0])
        # encoder decoder attention
        self.cross_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[1],
            bias_attr=bias_attrs[1])
        # feed forward
        self.linear1 = Linear(
            d_model, dim_feedforward, weight_attrs[2], bias_attr=bias_attrs[2])
        self.dropout = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = Linear(
            dim_feedforward, d_model, weight_attrs[2], bias_attr=bias_attrs[2])

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout3 = nn.Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, cache=None):
        tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)
        memory_mask = _convert_attention_mask(memory_mask, memory.dtype)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        if cache is None:
            tgt = self.self_attn(tgt, tgt, tgt, tgt_mask, None)
        else:
            tgt, incremental_cache = self.self_attn(tgt, tgt, tgt, tgt_mask,
                                                    cache[0])
        # 残差连接
        tgt = residual + self.dropout1(tgt)
        # Norm
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        if cache is None:
            tgt = self.cross_attn(tgt, memory, memory, memory_mask, None)
        else:
            tgt, static_cache = self.cross_attn(tgt, memory, memory,
                                                memory_mask, cache[1])
        # 残差连接
        tgt = residual + self.dropout2(tgt) 
        # Norm
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm3(tgt)
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # 残差连接
        tgt = residual + self.dropout3(tgt)
        # Norm
        if not self.normalize_before:
            tgt = self.norm3(tgt)
        return tgt if cache is None else (tgt, (incremental_cache,
                                                static_cache))
# # 3、案例实践

# ## 3.1  环境介绍
# 安装依赖
# !pip install -r requirements.txt
import yaml
from pprint import pprint
from attrdict import AttrDict
import jieba
from tqdm import tqdm
from mecab import MeCab
from paddlenlp.data.sampler import SamplerHelper
from paddlenlp.transformers import TransformerModel, InferTransformerModel, CrossEntropyCriterion, position_encoding_init
en_dir='zh-en/train.tags.zh-en.en'
zn_dir='zh-en/train.tags.zh-en.zh'
def filter_out_html(filename1,filename2):
	f1 = open(filename1,'r')
	f2 = open(filename2,'r')

	data1 = f1.readlines()
	data2 = f2.readlines()
	assert len(data1)==len(data2)#用codecs会导致报错不知道为什么
	fw1 = open(filename1+".txt",'w')
	fw2 = open(filename2+".txt",'w')

	for line1,line2 in tqdm(zip(data1,data2)):
		line1 = line1.strip()
		line2 = line2.strip()
		if line1 and line2:
			if '<' not in line1 and '>' not in line1 and '<' not in line2 and '>' not in line2:
				fw1.write(line1+"\n")
				fw2.write(line2+"\n")
	fw1.close()
	f1.close()
	fw2.close()
	f2.close()

	return filename1+".txt",filename2+".txt"
# In[12]:


# filter_out_html(en_dir,zn_dir)


# In[22]:


# tree_source_dev = ET.parse('zh-en/IWSLT15.TED.dev2010.zh-en.zh.xml')
# tree_source_dev = [seg.text for seg in tree_source_dev.iter('seg')]

# tree_target_dev = ET.parse('zh-en/IWSLT15.TED.dev2010.zh-en.en.xml')
# tree_target_dev = [ts.translate_text(seg.text, translator='baidu', to_language='kor') for seg in tree_target_dev.iter('seg')]


# In[23]:


# print(tree_source_dev[:2])
# print(tree_target_dev[:2])


# In[24]:


# with open('dev_cn.txt','w') as f:
#     for item in tree_source_dev:
#         f.write(item+'\n')

# with open('dev_en.txt','w') as f:
#     for item in tree_target_dev:
#         f.write(item+'\n')


# In[16]:



# tree_source_test = ET.parse('zh-en/IWSLT15.TED.tst2011.zh-en.zh.xml')
# tree_source_test = [seg.text for seg in tree_source_test.iter('seg')]

# tree_target_test = ET.parse('zh-en/IWSLT15.TED.tst2011.zh-en.en.xml')
# tree_target_test = [ts.translate_text(seg.text, translator='baidu', to_language='kor') for seg in tree_target_test.iter('seg')]


# In[17]:


# with open('test_cn.txt','w') as f:
#     for item in tree_source_test:
#         f.write(item+'\n')

# with open('test_en.txt','w') as f:
#     for item in tree_target_test:
#         f.write(item+'\n')
# ## 3.2 数据部分
# ### 3.2.1  数据集介绍
# 中文Jieba分词
def jieba_cut(in_file,out_file):
    out_f = open(out_file,'w',encoding='utf8')
    with open(in_file,'r',encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            cut_line = ' '.join(jieba.cut(line))
            out_f.write(cut_line+'\n')
    out_f.close()
zn_dir='zh-en/train.tags.zh-en.zh.txt'
cut_zn_dir='zh-en/train.tags.zh-en.zh.cut.txt'
jieba_cut(zn_dir,cut_zn_dir)
zn_dir='dev_cn.txt'
cut_zn_dir='dev_cn.cut.txt'
jieba_cut(zn_dir,cut_zn_dir)
# zn_dir='test_cn.txt'
# cut_zn_dir='test_cn.cut.txt'
# jieba_cut(zn_dir,cut_zn_dir)



# 韩文文分词
wakati = MeCab()
def mecab_cut(in_file,out_file):
    out_f = open(out_file,'w',encoding='utf8')
    with open(in_file,'r',encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            cut_line = ' '.join(wakati.morphs(line))
            out_f.write(cut_line+'\n')
    out_f.close()
en_dir='zh-en/train.tags.zh-en.en.txt'
cut_en_dir='zh-en/train.tags.zh-en.en.cut.txt'
mecab_cut(en_dir,cut_en_dir)
en_dir='dev_en.txt'
cut_en_dir='dev_en.cut.txt'
mecab_cut(en_dir,cut_en_dir)
# en_dir='test_en.txt'
# cut_en_dir='test_en.cut.txt'
# mecab_cut(en_dir,cut_en_dir)
# print('generate the training data')
# get_ipython().system('subword-nmt learn-bpe -s 32000 < zh-en/train.tags.zh-en.zh.cut.txt > zh-en/bpe.ch.32000')
# get_ipython().system('subword-nmt learn-bpe -s 32000 < zh-en/train.tags.zh-en.en.txt > zh-en/bpe.en.32000')
# ### 2.3.2 bpe分词
# In[35]:
# get_ipython().system('subword-nmt apply-bpe -c zh-en/bpe.ch.32000 < zh-en/train.tags.zh-en.zh.cut.txt > zh-en/train.ch.bpe')
# get_ipython().system('subword-nmt apply-bpe -c zh-en/bpe.ch.32000 < dev_cn.cut.txt > zh-en/dev.ch.bpe')
# get_ipython().system('subword-nmt apply-bpe -c zh-en/bpe.ch.32000 < test_cn.cut.txt > zh-en/test.ch.bpe')
# get_ipython().system('subword-nmt apply-bpe -c zh-en/bpe.en.32000 < zh-en/train.tags.zh-en.en.txt > zh-en/train.en.bpe')
# get_ipython().system('subword-nmt apply-bpe -c zh-en/bpe.en.32000 < dev_en.txt > zh-en/dev.en.bpe')
# get_ipython().system('subword-nmt apply-bpe -c zh-en/bpe.en.32000 < test_en.txt > zh-en/test.en.bpe')
# In[36]:
# get_ipython().system('subword-nmt  get-vocab -i zh-en/train.ch.bpe -o zh-en/temp')
# In[37]:
special_token=['<s>','<e>','<unk>']
cn_vocab=[]
with open('zh-en/temp') as f:
    for item in f.readlines():
        words=item.strip().split()
        cn_vocab.append(words[0])

with open('zh-en/vocab.ch.src','w') as f:
    for item in special_token:
        f.write(item+'\n')
    for item in cn_vocab:
        f.write(item+'\n')

# get_ipython().system('subword-nmt  get-vocab -i zh-en/train.en.bpe -o zh-en/temp')

eng_vocab=[]
with open('zh-en/temp') as f:
    for item in f.readlines():
        words=item.strip().split()
        eng_vocab.append(words[0])
with open('zh-en/vocab.en.tgt','w') as f:
    for item in special_token:
        f.write(item+'\n')
    for item in eng_vocab:
        f.write(item+'\n')
# ## 3.4 数据集划分

# In[40]:
cn_data=[]
with open('zh-en/train.ch.bpe') as f:
    for item in f.readlines():
        words=item.strip()
        cn_data.append(words)
en_data=[]
with open('zh-en/train.en.bpe') as f:
    for item in f.readlines():
        words=item.strip()
        en_data.append(words)
print(cn_data[:10])
print(en_data[:10])
# ## 3.5 构造dataloader
# 
# 下面的`create_data_loader`函数用于创建训练集、验证集所需要的`DataLoader`对象,  
# `create_infer_loader`函数用于创建预测集所需要的`DataLoader`对象，   
# `DataLoader`对象用于产生一个个batch的数据。下面对函数中调用的`paddlenlp`内置函数作简单说明：
# * `paddlenlp.data.Vocab.load_vocabulary`：Vocab词表类，集合了一系列文本token与ids之间映射的一系列方法，支持从文件、字典、json等一系方式构建词表
# * `paddlenlp.datasets.load_dataset`：从本地文件创建数据集时，推荐根据本地数据集的格式给出读取function并传入 load_dataset() 中创建数据集
# * `paddlenlp.data.sampler.SamplerHelper`：构建用于DataLoader的可迭代采样器，它包含shuffle、sort、batch、shard等一系列方法，方便用户灵活使用
# * `paddlenlp.data.Pad`：padding 操作
# 
# 具体可参考[PaddleNLP的文档](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_self_defined.html)

# In[42]:
def min_max_filer(data, max_len, min_len=0):
    # 1 for special tokens.
    data_min_len = min(len(data[0]), len(data[1])) + 1
    data_max_len = max(len(data[0]), len(data[1])) + 1
    return (data_min_len >= min_len) and (data_max_len <= max_len)
def read(src_path, tgt_path, is_predict=False):
    if is_predict:
        with open(src_path, 'r', encoding='utf8') as src_f:
            for src_line in src_f.readlines():
                src_line = src_line.strip()
                if not src_line:
                    continue
                yield {'src':src_line, 'tgt':''}
    else:
        with open(src_path, 'r', encoding='utf8') as src_f, open(tgt_path, 'r', encoding='utf8') as tgt_f:
            for src_line, tgt_line in zip(src_f.readlines(), tgt_f.readlines()):
                src_line = src_line.strip()
                if not src_line:
                    continue
                tgt_line = tgt_line.strip()
                if not tgt_line:
                    continue
                yield {'src':src_line, 'tgt':tgt_line}
# In[43]:


# 创建训练集、验证集的dataloader
def create_data_loader(args):
    train_dataset = load_dataset(read, src_path=args.training_file.split(',')[0], tgt_path=args.training_file.split(',')[1], lazy=False)
    dev_dataset = load_dataset(read, src_path=args.training_file.split(',')[0], tgt_path=args.training_file.split(',')[1], lazy=False)
    print('load src vocab')
    print( args.src_vocab_fpath)
    src_vocab = Vocab.load_vocabulary(
        args.src_vocab_fpath,
        bos_token=args.special_token[0],
        eos_token=args.special_token[1],
        unk_token=args.special_token[2])
    print('load trg vocab')
    print(args.trg_vocab_fpath)
    trg_vocab = Vocab.load_vocabulary(
        args.trg_vocab_fpath,
        bos_token=args.special_token[0],
        eos_token=args.special_token[1],
        unk_token=args.special_token[2])
    print('padding')
    padding_vocab = (
        lambda x: (x + args.pad_factor - 1) // args.pad_factor * args.pad_factor
    )
    args.src_vocab_size = padding_vocab(len(src_vocab))
    args.trg_vocab_size = padding_vocab(len(trg_vocab))
    print('convert example')
    def convert_samples(sample):
        source = sample['src'].split()
        target = sample['tgt'].split()

        source = src_vocab.to_indices(source)
        target = trg_vocab.to_indices(target)

        return source, target

    data_loaders = [(None)] * 2
    print('dataset loop')
    for i, dataset in enumerate([train_dataset, dev_dataset]):
        dataset = dataset.map(convert_samples, lazy=False).filter(
            partial(
                min_max_filer, max_len=args.max_length))

        sampler = SamplerHelper(dataset)

        if args.sort_type == SortType.GLOBAL:
            src_key = (lambda x, data_source: len(data_source[x][0]) + 1)
            trg_key = (lambda x, data_source: len(data_source[x][1]) + 1)
            # Sort twice
            sampler = sampler.sort(key=trg_key).sort(key=src_key)
        else:
            if args.shuffle:
                sampler = sampler.shuffle(seed=args.shuffle_seed)
            max_key = (lambda x, data_source: max(len(data_source[x][0]), len(data_source[x][1])) + 1)
            if args.sort_type == SortType.POOL:
                sampler = sampler.sort(key=max_key, buffer_size=args.pool_size)

        batch_size_fn = lambda new, count, sofar, data_source: max(sofar, len(data_source[new][0]) + 1,
                                                                   len(data_source[new][1]) + 1)
        batch_sampler = sampler.batch(
            batch_size=args.batch_size,
            drop_last=False,
            batch_size_fn=batch_size_fn,
            key=lambda size_so_far, minibatch_len: size_so_far * minibatch_len)

        if args.shuffle_batch:
            batch_sampler = batch_sampler.shuffle(seed=args.shuffle_seed)

        if i == 0:
            batch_sampler = batch_sampler.shard()

        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=partial(
                prepare_train_input,
                bos_idx=args.bos_idx,
                eos_idx=args.eos_idx,
                pad_idx=args.bos_idx),
            num_workers=2,
            return_list=True)
        data_loaders[i] = (data_loader)
    return data_loaders
class SortType(object):
    GLOBAL = 'global'
    POOL = 'pool'
    NONE = "none"
# ## 3.6  模型训练
# PaddleNLP提供Transformer API供调用：
# * `paddlenlp.transformers.TransformerModel`：Transformer模型的实现
# * `paddlenlp.transformers.InferTransformerModel`：Transformer模型用于生成
# * `paddlenlp.transformers.CrossEntropyCriterion`：计算交叉熵损失
# * `paddlenlp.transformers.position_encoding_init`：Transformer 位置编码的初始化

# 运行`do_train`函数，
# 在`do_train`函数中，配置优化器、损失函数，以及评价指标（困惑度）。

# In[44]:


# !pwd

# 读入参数:这个配置文件需要在这里找：https://github.com/paddlepaddle/awesome-DeepLearning
yaml_file = './transformer.base.yaml'
with open(yaml_file, 'rt') as f:
    args = AttrDict(yaml.safe_load(f))
    pprint(args)
print(args.training_file.split(','))
def prepare_train_input(insts, bos_idx, eos_idx, pad_idx):
    """
    Put all padded data needed by training into a list.
    """
    word_pad = Pad(pad_idx)
    src_word = word_pad([inst[0] + [eos_idx] for inst in insts])
    trg_word = word_pad([[bos_idx] + inst[1] for inst in insts])
    lbl_word = np.expand_dims(
        word_pad([inst[1] + [eos_idx] for inst in insts]), axis=2)

    data_inputs = [src_word, trg_word, lbl_word]

    return data_inputs
# In[47]:


# Define data loader
(train_loader), (eval_loader) = create_data_loader(args)
for input_data in train_loader:
    (src_word, trg_word, lbl_word) = input_data
    print(src_word)
    print(trg_word)
    print(lbl_word)
    break
def do_train(args,train_loader,eval_loader):
    #代码运行环境的选择：
    if args.use_gpu:
        rank = dist.get_rank()
        trainer_count = dist.get_world_size()
    else:
        rank = 0
        trainer_count = 1
        paddle.set_device("cpu")
    #如果trainer_count大于1，则执行分布式训练的初始化操作。
    if trainer_count > 1:
        dist.init_parallel_env()

    # Set seed for CE训练迭代
    random_seed = eval(str(args.random_seed))
    if random_seed is not None:
        paddle.seed(random_seed)

    # Define model
    transformer = TransformerModel(
        src_vocab_size=args.src_vocab_size,
        trg_vocab_size=args.trg_vocab_size,
        max_length=args.max_length + 1,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        d_inner_hid=args.d_inner_hid,
        dropout=args.dropout,
        weight_sharing=args.weight_sharing,
        bos_id=args.bos_idx,
        eos_id=args.eos_idx)

    # Define loss
    criterion = CrossEntropyCriterion(args.label_smooth_eps, args.bos_idx)

    scheduler = paddle.optimizer.lr.NoamDecay(
        args.d_model, args.warmup_steps, args.learning_rate, last_epoch=0)

    # Define optimizer
    optimizer = paddle.optimizer.Adam(
        learning_rate=scheduler,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=float(args.eps),
        parameters=transformer.parameters())

    # Init from some checkpoint, to resume the previous training
    if args.init_from_checkpoint:
        model_dict = paddle.load(
            os.path.join(args.init_from_checkpoint, "transformer.pdparams"))
        opt_dict = paddle.load(
            os.path.join(args.init_from_checkpoint, "transformer.pdopt"))
        transformer.set_state_dict(model_dict)
        optimizer.set_state_dict(opt_dict)
        print("loaded from checkpoint.")
    # Init from some pretrain models, to better solve the current task
    if args.init_from_pretrain_model:
        model_dict = paddle.load(
            os.path.join(args.init_from_pretrain_model, "transformer.pdparams"))
        transformer.set_state_dict(model_dict)
        print("loaded from pre-trained model.")

    if trainer_count > 1:
        transformer = paddle.DataParallel(transformer)

    # The best cross-entropy value with label smoothing
    loss_normalizer = -(
        (1. - args.label_smooth_eps) * np.log(
            (1. - args.label_smooth_eps)) + args.label_smooth_eps *
        np.log(args.label_smooth_eps / (args.trg_vocab_size - 1) + 1e-20))

    ce_time = []
    ce_ppl = []
    step_idx = 0

    # Train loop
    for pass_id in range(args.epoch):
        epoch_start = time.time()

        batch_id = 0
        batch_start = time.time()
        for input_data in train_loader:
            (src_word, trg_word, lbl_word) = input_data

            logits = transformer(src_word=src_word, trg_word=trg_word)

            sum_cost, avg_cost, token_num = criterion(logits, lbl_word)

            avg_cost.backward()

            optimizer.step()
            optimizer.clear_grad()

            if step_idx % args.print_step == 0 and rank == 0:
                total_avg_cost = avg_cost.numpy()

                if step_idx == 0:
                    logger.info(
                        "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                        "normalized loss: %f, ppl: %f " %
                        (step_idx, pass_id, batch_id, total_avg_cost,
                         total_avg_cost - loss_normalizer,
                         np.exp([min(total_avg_cost, 100)])))
                else:
                    train_avg_batch_cost = args.print_step / (
                        time.time() - batch_start)
                    logger.info(
                        "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                        "normalized loss: %f, ppl: %f, avg_speed: %.2f step/sec"
                        % (
                            step_idx,
                            pass_id,
                            batch_id,
                            total_avg_cost,
                            total_avg_cost - loss_normalizer,
                            np.exp([min(total_avg_cost, 100)]),
                            train_avg_batch_cost, ))
                batch_start = time.time()

            if step_idx % args.save_step == 0 and step_idx != 0:
                # Validation
                transformer.eval()
                total_sum_cost = 0
                total_token_num = 0
                with paddle.no_grad():
                    for input_data in eval_loader:
                        (src_word, trg_word, lbl_word) = input_data
                        logits = transformer(
                            src_word=src_word, trg_word=trg_word)
                        sum_cost, avg_cost, token_num = criterion(logits,
                                                                  lbl_word)
                        total_sum_cost += sum_cost.numpy()
                        total_token_num += token_num.numpy()
                        total_avg_cost = total_sum_cost / total_token_num
                    logger.info("validation, step_idx: %d, avg loss: %f, "
                                "normalized loss: %f, ppl: %f" %
                                (step_idx, total_avg_cost,
                                 total_avg_cost - loss_normalizer,
                                 np.exp([min(total_avg_cost, 100)])))
                transformer.train()

                if args.save_model and rank == 0:
                    model_dir = os.path.join(args.save_model,
                                             "step_" + str(step_idx))
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    paddle.save(transformer.state_dict(),
                                os.path.join(model_dir, "transformer.pdparams"))
                    paddle.save(optimizer.state_dict(),
                                os.path.join(model_dir, "transformer.pdopt"))
                batch_start = time.time()
            batch_id += 1
            step_idx += 1
            scheduler.step()

        train_epoch_cost = time.time() - epoch_start
        ce_time.append(train_epoch_cost)
        logger.info("train epoch: %d, epoch_cost: %.5f s" %
                    (pass_id, train_epoch_cost))

    if args.save_model and rank == 0:
        model_dir = os.path.join(args.save_model, "step_final")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        paddle.save(transformer.state_dict(),
                    os.path.join(model_dir, "transformer.pdparams"))
        paddle.save(optimizer.state_dict(),
                    os.path.join(model_dir, "transformer.pdopt"))
print('training the model')
do_train(args,train_loader,eval_loader)
