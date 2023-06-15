# ## 3.7 模型预测
import logging
import os
import re
from functools import partial

import jieba
import paddle
import yaml
from attrdict import AttrDict
from paddle.fluid.reader import DataLoader
from paddlenlp.data import Vocab, Pad, SamplerHelper
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import InferTransformerModel, position_encoding_init
from typing import List

# 载入配置文件
yaml_file = './config/transformer.base.yaml'
with open(yaml_file, 'rt') as f:
    args = AttrDict(yaml.safe_load(f))
# 指定使用cpu还是GPU
device = paddle.get_device()
paddle.set_device(device)
# 设置jieba的输出信息的等级,即屏蔽jieba输出的警告
jieba.setLogLevel(logging.INFO)

# 中文的词汇表
src_vocab = Vocab.load_vocabulary(
    args.src_vocab_fpath,
    bos_token=args.special_token[0],
    eos_token=args.special_token[1],
    unk_token=args.special_token[2])
# 韩语的词汇表
trg_vocab = Vocab.load_vocabulary(
    args.trg_vocab_fpath,
    bos_token=args.special_token[0],
    eos_token=args.special_token[1],
    unk_token=args.special_token[2])

padding_vocab = (
    lambda x: (x + args.pad_factor - 1) // args.pad_factor * args.pad_factor
)
# 计算词汇表尺寸
args.src_vocab_size = padding_vocab(len(src_vocab))
args.trg_vocab_size = padding_vocab(len(trg_vocab))
# 定义模型结构
transformer = InferTransformerModel(
    src_vocab_size=args.src_vocab_size,
    trg_vocab_size=args.trg_vocab_size,
    num_encoder_layers=args.n_layer,
    num_decoder_layers=args.n_layer,
    max_length=args.max_length + 1,
    n_head=args.n_head,
    d_model=args.d_model,
    d_inner_hid=args.d_inner_hid,
    dropout=args.dropout,
    weight_sharing=args.weight_sharing,
    bos_id=args.bos_idx,
    eos_id=args.eos_idx,
    beam_size=args.beam_size,
    max_out_len=args.max_out_len)
# 打印模型的结构
# paddle.summary(transformer, (4, args.max_length), dtypes='int32')
# 导入已经训练过的模型
init_from_params = 'trained_models'
model_dict = paddle.load(
    os.path.join(init_from_params, "transformer.pdparams"))
# To avoid a longer length than training, reset the size of position
# encoding to max_length
model_dict["encoder.pos_encoder.weight"] = position_encoding_init(
    args.max_length + 1, args.d_model)
model_dict["decoder.pos_encoder.weight"] = position_encoding_init(
    args.max_length + 1, args.d_model)
# 将已训练的模型参数填充进当前的模型对象
transformer.set_state_dict(model_dict)
# 切换为评估模式,将模型用于预测
transformer.eval()


# 对输入的中文进行处理
def read(src_texts):
    for src_text in src_texts:
        # 去除字符串的空格
        src_text = src_text.replace(' ', '')
        # 对中文进行分词
        src_text = " ".join(jieba.cut(src_text))
        if not src_text:
            continue
        yield {'src': src_text, 'tgt': ''}


# 创建用于预测的dataloader
def create_infer_loader():
    dataset = load_dataset(read, src_texts=args.predict_texts, lazy=False)

    def convert_samples(sample):
        source = sample['src'].split()
        target = sample['tgt'].split()

        source = src_vocab.to_indices(source)
        target = trg_vocab.to_indices(target)

        return source, target

    dataset = dataset.map(convert_samples, lazy=False)

    batch_sampler = SamplerHelper(dataset).batch(
        batch_size=args.infer_batch_size, drop_last=False)

    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=partial(
            prepare_infer_input,
            bos_idx=args.bos_idx,
            eos_idx=args.eos_idx,
            pad_idx=args.bos_idx),
        num_workers=2,
        return_list=True)
    return data_loader, trg_vocab.to_tokens


def prepare_infer_input(insts, bos_idx, eos_idx, pad_idx):
    """
    Put all padded data needed by beam search decoder into a list.
    """
    word_pad = Pad(pad_idx)
    src_word = word_pad([inst[0] + [eos_idx] for inst in insts])

    return [src_word, ]


def post_process_seq(seq, bos_idx, eos_idx, output_bos=False, output_eos=False):
    """
    Post-process the decoded sequence.
    """
    eos_pos = len(seq) - 1
    for i, idx in enumerate(seq):
        if idx == eos_idx:
            eos_pos = i
            break
    seq = [
        idx for idx in seq[:eos_pos + 1]
        if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)
    ]
    return seq


def do_predict(src_texts: List[str]):
    """
    执行预测,将中文翻译为韩文
    :param src_texts: 待翻译的句子列表
    :return: 对应的翻译结果列表
    """
    args.predict_texts = src_texts
    # 定义数据加载器
    test_loader, to_tokens = create_infer_loader()
    result = []  # 记录每一句的翻译结果
    with paddle.no_grad():
        for (src_word,) in test_loader:
            finished_seq = transformer(src_word=src_word)
            finished_seq = finished_seq.numpy().transpose([0, 2, 1])
            for ins in finished_seq:
                for beam_idx, beam in enumerate(ins):
                    if beam_idx >= args.n_best:
                        break
                    id_list = post_process_seq(beam, args.bos_idx, args.eos_idx)
                    word_list = to_tokens(id_list)
                    sequence = "".join(word_list) + "\n"
                    sequence = re.sub('(@@ )|(@@ \\?\\$)', '', sequence)
                    result.append(sequence)
    return result
