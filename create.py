import os.path
import translators as ts
from tqdm import tqdm
zh_dir = 'train.tags.zh-en.zh.txt'
with open(zh_dir, 'r', encoding='utf8') as f:
    sentences = 0
    if os.path.exists('zh-en/1train.tags.zh-en.en.txt'):
        with open('zh-en/1train.tags.zh-en.en.txt', 'r', encoding='utf8') as out:
            sentences = len(out.readlines())
    out_f = open('zh-en/1train.tags.zh-en.en.txt', 'a', encoding='utf8')
    # [50000:100000][100000:150000]
    for line in tqdm(f.readlines()[sentences:],desc='翻译进度',unit='句'):
        line = line.strip()
        out_f.write(ts.translate_text(line, translator='baidu', to_language='kor') + '\n')
    out_f.close()
