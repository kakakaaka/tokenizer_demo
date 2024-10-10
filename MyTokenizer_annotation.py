# 导入json模块，用于处理JSON格式的数据
import json
# 导入os模块，用于进行文件和路径操作
import os

# 定义一个自定义的分词器类
class MyTokenizer:
    # 初始化方法，接收词汇表、合并规则、添加的标记和特殊标记映射
    def __init__(self, vocab, merges, added_tokens, special_tokens_map):
        # 保存词汇表
        self.vocab = vocab
        # 创建ID到标记的映射字典
        self.id_to_token = {v: k for k, v in vocab.items()}
        # 保存合并规则
        self.merges = merges
        # 创建添加的标记字典，键为内容，值为ID
        self.added_tokens = {token['content']: token['id'] for token in added_tokens}
        # 保存特殊标记映射
        self.special_tokens_map = special_tokens_map

        # 获取特殊标记：序列开始标记
        self.bos_token = special_tokens_map.get('bos_token', '[BOS]')
        # 获取特殊标记：序列结束标记
        self.eos_token = special_tokens_map.get('eos_token', '[EOS]')
        # 获取特殊标记：填充标记
        self.pad_token = special_tokens_map.get('pad_token', '[PAD]')
        # 获取额外的特殊标记列表
        self.additional_special_tokens = special_tokens_map.get('additional_special_tokens', [])
        # 将所有特殊标记组合成列表
        self.special_tokens = [self.bos_token, self.eos_token, self.pad_token] + self.additional_special_tokens

        # 更新词汇表和ID映射，确保所有特殊标记都在词汇表中
        for token in self.special_tokens:
            if token not in self.vocab:
                index = len(self.vocab)
                self.vocab[token] = index
                self.id_to_token[index] = token

    # 类方法，从预训练的分词器文件中加载
    @classmethod
    def from_pretrained(cls, path):
        # 打开并读取 tokenizer.json 文件
        with open(os.path.join(path, 'tokenizer.json'), 'r', encoding='utf-8') as f:
            tokenizer_json = json.load(f)
        # 获取词汇表
        vocab = tokenizer_json['model']['vocab']
        # 获取合并规则
        merges = tokenizer_json['model']['merges']
        # 获取添加的标记
        added_tokens = tokenizer_json['added_tokens']

        # 打开并读取 special_tokens_map.json 文件
        with open(os.path.join(path, 'special_tokens_map.json'), 'r', encoding='utf-8') as f:
            special_tokens_map = json.load(f)

        # 返回类的实例
        return cls(vocab, merges, added_tokens, special_tokens_map)

    # 定义对象的可调用方法，用于编码文本
    def __call__(self, texts, max_length=20, padding='max_length', truncation=True,
                 return_offsets_mapping=True, return_special_tokens_mask=True):
        # 如果输入是字符串，转换为包含一个元素的列表
        if isinstance(texts, str):
            texts = [texts]

        # 初始化编码结果的字典
        encoded_batch = {
            'input_ids': [],
            'attention_mask': [],
            'offset_mapping': [],
            'special_tokens_mask': []
        }

        # 遍历每个文本
        for text in texts:
            # 对文本进行分词
            tokens = self.tokenize(text)
            # 将标记转换为ID
            token_ids = self.convert_tokens_to_ids(tokens)

            # 如果需要截断，并且长度超过最大长度，进行截断
            if truncation and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]

            # 生成注意力掩码，标记有效的token位置
            attention_mask = [1] * len(token_ids)

            # 如果需要填充到最大长度
            if padding == 'max_length':
                # 计算需要填充的长度
                padding_length = max_length - len(token_ids)
                # 在token_ids后添加填充标记的ID
                token_ids += [self.vocab.get(self.pad_token)] * padding_length
                # 在注意力掩码后添加0，表示填充位置
                attention_mask += [0] * padding_length

            # 如果需要返回偏移映射
            if return_offsets_mapping:
                offsets = []
                current_position = 0
                # 遍历每个标记，计算其在原文本中的起止位置
                for token in tokens:
                    token_length = len(token)
                    offsets.append((current_position, current_position + token_length))
                    current_position += token_length
                # 如果需要填充，填充偏移映射
                if padding == 'max_length':
                    offsets += [(0, 0)] * (max_length - len(offsets))
            else:
                offsets = None

            # 如果需要返回特殊标记掩码
            if return_special_tokens_mask:
                special_tokens_mask = [1 if self.id_to_token.get(id_) in self.special_tokens else 0 for id_ in
                                       token_ids]
            else:
                special_tokens_mask = None

            # 将结果添加到编码批次中
            encoded_batch['input_ids'].append(token_ids)
            encoded_batch['attention_mask'].append(attention_mask)
            if offsets is not None:
                encoded_batch['offset_mapping'].append(offsets)
            if special_tokens_mask is not None:
                encoded_batch['special_tokens_mask'].append(special_tokens_mask)

        # 返回编码后的批次
        return encoded_batch

    # 定义分词方法
    def tokenize(self, text):
        # 简化的分词实现，将文本按字符拆分
        tokens = list(text)
        return tokens

    # 将标记转换为ID列表
    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.vocab.get('[UNK]', 0)) for token in tokens]

    # 将ID列表转换为标记列表
    def convert_ids_to_tokens(self, ids):
        return [self.id_to_token.get(id_, '[UNK]') for id_ in ids]

    # 解码ID列表为原始文本
    def decode(self, ids):
        # 如果第一个元素是列表，取出第一项
        if isinstance(ids[0], list):
            ids = ids[0]
        # 将ID转换为标记
        tokens = self.convert_ids_to_tokens(ids)
        # 将标记列表拼接成字符串
        text = ''.join(tokens)
        # 移除特殊标记
        for token in self.special_tokens:
            text = text.replace(token, '')
        # 去除首尾空格并返回
        return text.strip()

    # 添加新的特殊标记
    def add_special_tokens(self, special_tokens_dict):
        # 遍历特殊标记字典的键和值
        for key, value in special_tokens_dict.items():
            # 如果值是列表，说明有多个标记
            if isinstance(value, list):
                for token in value:
                    # 如果标记不在词汇表中，添加进去
                    if token not in self.vocab:
                        index = len(self.vocab)
                        self.vocab[token] = index
                        self.id_to_token[index] = token
                        self.special_tokens.append(token)
            else:
                # 如果值不是列表，直接处理单个标记
                if value not in self.vocab:
                    index = len(self.vocab)
                    self.vocab[value] = index
                    self.id_to_token[index] = value
                    self.special_tokens.append(value)
            # 更新特殊标记映射
            self.special_tokens_map[key] = value
