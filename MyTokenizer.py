import json
import os

class MyTokenizer:
    def __init__(self, vocab, merges, added_tokens, special_tokens_map):
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.added_tokens = {token['content']: token['id'] for token in added_tokens}
        self.special_tokens_map = special_tokens_map

        # 特殊标记
        self.bos_token = special_tokens_map.get('bos_token', '[BOS]')
        self.eos_token = special_tokens_map.get('eos_token', '[EOS]')
        self.pad_token = special_tokens_map.get('pad_token', '[PAD]')
        self.additional_special_tokens = special_tokens_map.get('additional_special_tokens', [])
        self.special_tokens = [self.bos_token, self.eos_token, self.pad_token] + self.additional_special_tokens

        # 更新词汇表和ID映射
        for token in self.special_tokens:
            if token not in self.vocab:
                index = len(self.vocab)
                self.vocab[token] = index
                self.id_to_token[index] = token

    @classmethod
    def from_pretrained(cls, path):
        # 加载 tokenizer.json
        with open(os.path.join(path, 'tokenizer.json'), 'r', encoding='utf-8') as f:
            tokenizer_json = json.load(f)
        vocab = tokenizer_json['model']['vocab']
        merges = tokenizer_json['model']['merges']
        added_tokens = tokenizer_json['added_tokens']

        # 加载 special_tokens_map.json
        with open(os.path.join(path, 'special_tokens_map.json'), 'r', encoding='utf-8') as f:
            special_tokens_map = json.load(f)

        return cls(vocab, merges, added_tokens, special_tokens_map)

    def __call__(self, texts, max_length=20, padding='max_length', truncation=True,
                 return_offsets_mapping=True, return_special_tokens_mask=True):
        if isinstance(texts, str):
            texts = [texts]

        encoded_batch = {
            'input_ids': [],
            'attention_mask': [],
            'offset_mapping': [],
            'special_tokens_mask': []
        }

        for text in texts:
            tokens = self.tokenize(text)
            token_ids = self.convert_tokens_to_ids(tokens)

            # 截断
            if truncation and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]

            # 计算注意力掩码
            attention_mask = [1] * len(token_ids)

            # 填充
            if padding == 'max_length':
                padding_length = max_length - len(token_ids)
                token_ids += [self.vocab.get(self.pad_token)] * padding_length
                attention_mask += [0] * padding_length

            # 偏移映射
            if return_offsets_mapping:
                offsets = []
                current_position = 0
                for token in tokens:
                    token_length = len(token)
                    offsets.append((current_position, current_position + token_length))
                    current_position += token_length
                if padding == 'max_length':
                    offsets += [(0, 0)] * (max_length - len(offsets))
            else:
                offsets = None

            # 特殊标记掩码
            if return_special_tokens_mask:
                special_tokens_mask = [1 if self.id_to_token.get(id_) in self.special_tokens else 0 for id_ in
                                       token_ids]
            else:
                special_tokens_mask = None

            encoded_batch['input_ids'].append(token_ids)
            encoded_batch['attention_mask'].append(attention_mask)
            if offsets is not None:
                encoded_batch['offset_mapping'].append(offsets)
            if special_tokens_mask is not None:
                encoded_batch['special_tokens_mask'].append(special_tokens_mask)

        return encoded_batch

    def tokenize(self, text):
        # 简化的分词实现（实际应根据BPE算法和 merges 进行）
        tokens = list(text)
        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.vocab.get('[UNK]', 0)) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.id_to_token.get(id_, '[UNK]') for id_ in ids]

    def decode(self, ids):
        if isinstance(ids[0], list):
            ids = ids[0]
        tokens = self.convert_ids_to_tokens(ids)
        text = ''.join(tokens)
        for token in self.special_tokens:
            text = text.replace(token, '')
        return text.strip()


    def add_special_tokens(self, special_tokens_dict):
        for key, value in special_tokens_dict.items():
            if isinstance(value, list):
                for token in value:
                    if token not in self.vocab:
                        index = len(self.vocab)
                        self.vocab[token] = index
                        self.id_to_token[index] = token
                        self.special_tokens.append(token)
            else:
                if value not in self.vocab:
                    index = len(self.vocab)
                    self.vocab[value] = index
                    self.id_to_token[index] = value
                    self.special_tokens.append(value)
            self.special_tokens_map[key] = value
