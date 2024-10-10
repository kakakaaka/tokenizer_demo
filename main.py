
from MyTokenizer import MyTokenizer

if __name__ == '__main__':

    # 假设您的文件位于当前目录下的 'Mini-Chinese-Phi3' 文件夹中

    # 加载分词器
    tokenizer = MyTokenizer.from_pretrained('Mini-Chinese-Phi3')

    # 编码单个句子
    encoded = tokenizer('你好，世界！', max_length=20, padding='max_length', truncation=True,
                        return_offsets_mapping=True, return_special_tokens_mask=True)
    print('编码结果：')
    print(encoded)

    # 解码
    decoded_text = tokenizer.decode(encoded['input_ids'])
    print('\n解码结果：')
    print(decoded_text)

    # 编码批量句子
    texts = ['今天天气怎么样？', '我想去公园散步。']
    encoded_batch = tokenizer(texts, max_length=20, padding='max_length', truncation=True,
                              return_offsets_mapping=True, return_special_tokens_mask=True)
    print('\n批量编码结果：')
    print(encoded_batch)

    # 添加特殊标记
    tokenizer.add_special_tokens({'additional_special_tokens': ['[NEW_TOKEN]']})
    print('\n更新后的特殊标记列表：')
    print(tokenizer.special_tokens)
