# MyTokenizer 方法详解

以下将详细介绍 `MyTokenizer` 类中的关键方法，包括 `__call__`、`decode`、`add_special_tokens` 和 `from_pretrained`。将结合数学语言和直观的操作步骤，详细描述每个方法的算法和数据流程。

---

## **1. `__call__` 方法**

### **方法概述**

`__call__` 方法用于将输入的文本或文本列表编码为模型可接受的格式。主要完成以下功能：

- **分词**：将文本拆分为词元（tokens）。
- **映射**：将词元转换为对应的词汇表ID（token IDs）。
- **截断**：根据指定的最大长度截断序列。
- **填充**：对序列进行填充以达到统一的长度。
- **生成注意力掩码**：标识实际词元和填充词元的位置。
- **生成偏移映射**：记录每个词元在原始文本中的位置。
- **生成特殊标记掩码**：标识特殊标记在序列中的位置。

### **详细步骤**

1. **输入处理**

   - **单个文本处理**：如果输入是字符串 `texts`，则将其转换为包含一个元素的列表。

     ```python
     if isinstance(texts, str):
         texts = [texts]
     ```

2. **初始化结果字典**

   - 创建一个字典 `encoded_batch`，用于存储编码结果。

     ```python
     encoded_batch = {
         'input_ids': [],
         'attention_mask': [],
         'offset_mapping': [],
         'special_tokens_mask': []
     }
     ```

3. **遍历输入文本列表**

   - 对于每个文本 `text`，执行以下步骤：

     ```python
     for text in texts:
         # 后续步骤
     ```

4. **分词**

   - 调用 `tokenize` 方法，将文本拆分为词元列表 `tokens`。

     ```python
     tokens = self.tokenize(text)
     ```

   - **示例**：对于文本 `"你好，世界！"`，分词结果为：

     ```
     tokens = ['你', '好', '，', '世', '界', '！']
     ```

5. **词元转换为ID**

   - 调用 `convert_tokens_to_ids` 方法，将词元列表转换为对应的ID列表 `token_ids`。

     ```python
     token_ids = self.convert_tokens_to_ids(tokens)
     ```

   - **映射关系**：使用词汇表 `self.vocab`。

     - 数学表示：

       $$
       \text{token\_ids}[i] = \text{self.vocab.get}(\text{tokens}[i], \text{self.vocab.get}('[UNK]', 0))
       $$

   - **示例**：假设词汇表中 `'你'` 的ID为6，`'好'` 的ID为7，则：

     ```
     token_ids = [6, 7, 8, 9, 10, 11]
     ```

6. **截断**

   - 如果 `truncation=True` 且 `token_ids` 长度超过 `max_length`，则截断。

     ```python
     if truncation and len(token_ids) > max_length:
         token_ids = token_ids[:max_length]
     ```

   - **数学表示**：

     $$
     \text{token\_ids} = \text{token\_ids}[: \text{max\_length}]
     $$

7. **生成注意力掩码**

   - 创建一个与 `token_ids` 长度相同的列表 `attention_mask`，其中有效词元位置为1。

     ```python
     attention_mask = [1] * len(token_ids)
     ```

   - **数学表示**：

     $$
     \text{attention\_mask}[i] = 1, \quad \forall i \in [0, \text{len}(\text{token\_ids}) - 1]
     $$

8. **填充**

   - 如果 `padding='max_length'`，则对 `token_ids` 和 `attention_mask` 进行填充。

     ```python
     if padding == 'max_length':
         padding_length = max_length - len(token_ids)
         token_ids += [self.vocab.get(self.pad_token)] * padding_length
         attention_mask += [0] * padding_length
     ```

   - **计算填充长度**：

     $$
     \text{padding\_length} = \text{max\_length} - \text{len}(\text{token\_ids})
     $$

   - **填充 `token_ids` 和 `attention_mask`**：

     $$
     \text{token\_ids} = \text{token\_ids} + [\text{PAD\_ID}] \times \text{padding\_length}
     $$

     $$
     \text{attention\_mask} = \text{attention\_mask} + [0] \times \text{padding\_length}
     $$

   - **示例**：如果 `max_length=10`，原始 `token_ids` 长度为6，则需要填充4个 `[PAD]`。

9. **生成偏移映射**

   - 如果 `return_offsets_mapping=True`，则计算每个词元在原始文本中的起止位置。

     ```python
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
     ```

   - **计算过程**：

     - 初始化 `current_position = 0`。

     - 对于每个词元 `token`：

       - 计算 `token_length = len(token)`。

       - 偏移量为 `(current_position, current_position + token_length)`。

       - 更新 `current_position += token_length`。

   - **数学表示**：

     $$
     \text{offsets}[i] = (\sum_{k=0}^{i-1} \text{len}(\text{tokens}[k]), \sum_{k=0}^{i} \text{len}(\text{tokens}[k]))
     $$

   - **示例**：对于 `tokens = ['你', '好', '，', '世', '界', '！']`，偏移映射为：

     ```
     offsets = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6)]
     ```

   - **填充偏移映射**：如果需要填充，则在 `offsets` 后添加 `(0, 0)`。

10. **生成特殊标记掩码**

    - 如果 `return_special_tokens_mask=True`，则生成特殊标记掩码 `special_tokens_mask`。

      ```python
      if return_special_tokens_mask:
          special_tokens_mask = [1 if self.id_to_token.get(id_) in self.special_tokens else 0 for id_ in token_ids]
      else:
          special_tokens_mask = None
      ```

    - **计算过程**：

      - 对于每个 `id_`：

        - 如果 `id_` 对应的词元在 `self.special_tokens` 中，则标记为1，否则为0。

    - **数学表示**：

      $$
      \text{special\_tokens\_mask}[i] = \begin{cases}
      1, & \text{if } \text{id\_to\_token}(\text{token\_ids}[i]) \in \text{special\_tokens} \\
      0, & \text{otherwise}
      \end{cases}
      $$

    - **示例**：如果 `token_ids` 包含 `[PAD]` 的ID，则对应位置为1。

11. **结果收集**

    - 将计算得到的各项添加到 `encoded_batch` 中。

      ```python
      encoded_batch['input_ids'].append(token_ids)
      encoded_batch['attention_mask'].append(attention_mask)
      if offsets is not None:
          encoded_batch['offset_mapping'].append(offsets)
      if special_tokens_mask is not None:
          encoded_batch['special_tokens_mask'].append(special_tokens_mask)
      ```

12. **返回结果**

    - 返回包含所有编码信息的 `encoded_batch`。

      ```python
      return encoded_batch
      ```

### **数据流程示例**

- **输入**：`texts = ["你好，世界！"]`，`max_length = 10`，`padding = 'max_length'`。

- **步骤概览**：

  1. **分词**：`tokens = ['你', '好', '，', '世', '界', '！']`
  2. **映射为ID**：`token_ids = [6, 7, 8, 9, 10, 11]`
  3. **截断**：长度未超过 `max_length`，无需截断。
  4. **注意力掩码**：`attention_mask = [1, 1, 1, 1, 1, 1]`
  5. **填充**：填充 `[PAD]` 的ID，假设 `[PAD]` 的ID为0。

     ```
     padding_length = 10 - 6 = 4
     token_ids += [0, 0, 0, 0]
     attention_mask += [0, 0, 0, 0]
     ```

     最终 `token_ids = [6, 7, 8, 9, 10, 11, 0, 0, 0, 0]`

  6. **偏移映射**：

     ```
     offsets = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (0,0), (0,0), (0,0), (0,0)]
     ```

  7. **特殊标记掩码**：

     ```
     special_tokens_mask = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
     ```

- **输出**：

  ```python
  {
      'input_ids': [[6, 7, 8, 9, 10, 11, 0, 0, 0, 0]],
      'attention_mask': [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]],
      'offset_mapping': [[(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (0,0), (0,0), (0,0), (0,0)]],
      'special_tokens_mask': [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]]
  }
  ```

---

## **2. `decode` 方法**

### **方法概述**

`decode` 方法用于将ID序列转换回原始文本。主要完成以下功能：

- **ID到词元的映射**：将ID列表转换为词元列表。
- **拼接词元**：将词元列表拼接成字符串。
- **移除特殊标记**：从生成的文本中移除特殊标记。
- **返回结果**：返回去除特殊标记后的文本。

### **详细步骤**

1. **输入处理**

   - 如果 `ids` 的第一个元素是列表，则取第一个列表。

     ```python
     if isinstance(ids[0], list):
         ids = ids[0]
     ```

2. **ID转换为词元**

   - 调用 `convert_ids_to_tokens` 方法，将ID列表转换为词元列表 `tokens`。

     ```python
     tokens = self.convert_ids_to_tokens(ids)
     ```

   - **映射关系**：使用 `self.id_to_token`。

     - 数学表示：

       $$
       \text{tokens}[i] = \text{self.id\_to\_token.get}(\text{ids}[i], '[UNK]')
       $$

3. **拼接词元**

   - 将词元列表拼接成字符串 `text`。

     ```python
     text = ''.join(tokens)
     ```

4. **移除特殊标记**

   - 遍历 `self.special_tokens`，将文本中的特殊标记替换为空字符串。

     ```python
     for token in self.special_tokens:
         text = text.replace(token, '')
     ```

5. **去除首尾空格并返回**

   - 使用 `strip()` 方法去除首尾空格。

     ```python
     return text.strip()
     ```

### **数据流程示例**

- **输入**：`ids = [6, 7, 8, 9, 10, 11, 0, 0, 0, 0]`（对应之前编码的 `input_ids`）。

- **步骤概览**：

  1. **ID转换为词元**：

     ```
     tokens = ['你', '好', '，', '世', '界', '！', '[PAD]', '[PAD]', '[PAD]', '[PAD]']
     ```

  2. **拼接词元**：

     ```
     text = '你好，世界！[PAD][PAD][PAD][PAD]'
     ```

  3. **移除特殊标记**：

     - 移除 `[PAD]`：

       ```
       text = '你好，世界！'
       ```

  4. **去除首尾空格**：

     ```
     text = '你好，世界！'
     ```

- **输出**：返回字符串 `'你好，世界！'`

---

## **3. `add_special_tokens` 方法**

### **方法概述**

`add_special_tokens` 方法用于向分词器中添加新的特殊标记。主要完成以下功能：

- **更新词汇表**：将新的特殊标记添加到词汇表中。
- **更新ID映射**：在 `id_to_token` 中添加新的ID到标记的映射。
- **更新特殊标记列表**：在 `special_tokens` 中添加新的特殊标记。
- **更新特殊标记映射**：在 `special_tokens_map` 中更新特殊标记的键值对。

### **详细步骤**

1. **遍历特殊标记字典**

   - 对于 `special_tokens_dict` 中的每个键值对 `key`, `value`，执行以下操作：

     ```python
     for key, value in special_tokens_dict.items():
         # 后续步骤
     ```

2. **处理值为列表的情况**

   - 如果 `value` 是列表，表示有多个特殊标记需要添加。

     ```python
     if isinstance(value, list):
         for token in value:
             # 添加单个标记
     ```

3. **添加单个特殊标记**

   - 检查标记是否不在词汇表中，如果不在则添加。

     ```python
     if token not in self.vocab:
         index = len(self.vocab)
         self.vocab[token] = index
         self.id_to_token[index] = token
         self.special_tokens.append(token)
     ```

   - **更新词汇表和ID映射**：

     - 将新的标记添加到 `self.vocab`，ID为当前词汇表长度。

     - 在 `self.id_to_token` 中添加新的ID到标记的映射。

   - **更新特殊标记列表**：

     - 将新的标记添加到 `self.special_tokens`。

4. **处理值为单个标记的情况**

   - 如果 `value` 不是列表，直接处理单个标记。

     ```python
     else:
         if value not in self.vocab:
             index = len(self.vocab)
             self.vocab[value] = index
             self.id_to_token[index] = value
             self.special_tokens.append(value)
     ```

5. **更新特殊标记映射**

   - 在 `self.special_tokens_map` 中更新对应的键值对。

     ```python
     self.special_tokens_map[key] = value
     ```

### **数据流程示例**

- **输入**：`special_tokens_dict = {'additional_special_tokens': ['[NEW_TOKEN]']}`

- **步骤概览**：

  1. **遍历字典**：

     - `key = 'additional_special_tokens'`
     - `value = ['[NEW_TOKEN]']`

  2. **值为列表，遍历列表**：

     - `token = '[NEW_TOKEN]'`

  3. **检查并添加标记**：

     - 如果 `'[NEW_TOKEN]'` 不在 `self.vocab`，则：

       - `index = len(self.vocab)`

       - 添加到 `self.vocab`：

         ```
         self.vocab['[NEW_TOKEN]'] = index
         ```

       - 添加到 `self.id_to_token`：

         ```
         self.id_to_token[index] = '[NEW_TOKEN]'
         ```

       - 添加到 `self.special_tokens`：

         ```
         self.special_tokens.append('[NEW_TOKEN]')
         ```

  4. **更新特殊标记映射**：

     ```
     self.special_tokens_map['additional_special_tokens'] = ['[NEW_TOKEN]']
     ```

- **结果**：

  - 词汇表 `self.vocab` 新增 `'[NEW_TOKEN]'`。

  - ID映射 `self.id_to_token` 新增对应关系。

  - 特殊标记列表 `self.special_tokens` 新增 `'[NEW_TOKEN]'`。

---

## **4. `from_pretrained` 方法**

### **方法概述**

`from_pretrained` 是一个类方法，用于从预训练的分词器文件中加载分词器。主要完成以下功能：

- **加载 `tokenizer.json`**：获取词汇表、合并规则和添加的标记。
- **加载 `special_tokens_map.json`**：获取特殊标记的映射。
- **实例化 `MyTokenizer` 对象**：使用加载的数据创建分词器实例。

### **详细步骤**

1. **加载 `tokenizer.json` 文件**

   - 构建文件路径，打开并读取文件内容。

     ```python
     with open(os.path.join(path, 'tokenizer.json'), 'r', encoding='utf-8') as f:
         tokenizer_json = json.load(f)
     ```

   - **获取数据**：

     - 词汇表：

       ```python
       vocab = tokenizer_json['model']['vocab']
       ```

     - 合并规则：

       ```python
       merges = tokenizer_json['model']['merges']
       ```

     - 添加的标记：

       ```python
       added_tokens = tokenizer_json['added_tokens']
       ```

2. **加载 `special_tokens_map.json` 文件**

   - 构建文件路径，打开并读取文件内容。

     ```python
     with open(os.path.join(path, 'special_tokens_map.json'), 'r', encoding='utf-8') as f:
         special_tokens_map = json.load(f)
     ```

3. **实例化 `MyTokenizer` 对象**

   - 使用加载的数据，调用类的构造方法创建实例。

     ```python
     return cls(vocab, merges, added_tokens, special_tokens_map)
     ```

### **数据流程**

- **输入**：分词器文件所在的路径 `path`。

- **步骤概览**：

  1. **读取 `tokenizer.json`**：

     - 获取词汇表 `vocab`。

     - 获取合并规则 `merges`。

     - 获取添加的标记 `added_tokens`。

  2. **读取 `special_tokens_map.json`**：

     - 获取特殊标记映射 `special_tokens_map`。

  3. **实例化对象**：

     - 调用 `__init__` 方法，创建 `MyTokenizer` 实例。

- **输出**：返回一个初始化完毕的 `MyTokenizer` 对象。。
