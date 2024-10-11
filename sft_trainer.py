# sft_trainer.py

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from utils import set_seed
from typing import Optional, Dict, Any
from tqdm.auto import tqdm

class SFTTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        args,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        callbacks=None,
        max_seq_length: int = 512,
        dataset_text_field: str = "text",
        packing: bool = False,
    ):
        """
        SFTTrainer的初始化函数。

        参数：
            model (torch.nn.Module): 要训练的模型。
            args: 训练参数，类型为TrainingArguments。
            data_collator: 数据整理函数，将样本整理成批次。
            train_dataset: 训练数据集。
            eval_dataset: 验证数据集。
            tokenizer: 分词器，用于处理文本数据。
            callbacks: 回调函数列表，用于在训练过程中执行自定义操作。
            max_seq_length (int): 输入序列的最大长度。
            dataset_text_field (str): 数据集中文本字段的名称。
            packing (bool): 是否启用数据打包。
        """
        self.model = model
        self.args = args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.callbacks = callbacks if callbacks is not None else []
        self.max_seq_length = max_seq_length
        self.dataset_text_field = dataset_text_field
        self.packing = packing
        self.optimizer = None
        self.lr_scheduler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.should_stop = False  # 用于提前停止训练的标志

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        创建优化器和学习率调度器。

        参数：
            num_training_steps (int): 训练的总步数。
        """
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.args.learning_rate
        )
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

    def preprocess_dataset(self, dataset):
        """
        对数据集进行预处理，包括分词和数据打包。

        参数：
            dataset: 原始数据集。

        返回：
            预处理后的数据集。
        """
        def tokenize_function(examples):
            # 将文本进行分词，并添加 labels
            inputs = self.tokenizer(
                examples[self.dataset_text_field],
                truncation=True,
                max_length=self.max_seq_length,
                padding='max_length',  # 为了对齐序列
            )
            inputs['labels'] = inputs['input_ids'].copy()
            return inputs

        dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
        dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        return dataset

    def pack_dataset(self, dataset):
        """
        对数据集进行打包，以充分利用序列长度。

        参数：
            dataset: 分词后的数据集。

        返回：
            打包后的数据集。
        """
        # 这里实现数据打包的逻辑
        # 由于实现可能较为复杂，此处仅给出示例
        return dataset  # 返回打包后的数据集

    def get_train_dataloader(self):
        """
        创建训练数据的DataLoader。
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
        )

    def get_eval_dataloader(self):
        """
        创建验证数据的DataLoader。
        """
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
        )

    def train(self):
        """
        开始训练流程。
        """
        # 设置随机种子
        set_seed(self.args.seed)

        # 预处理数据集
        if self.train_dataset is not None:
            self.train_dataset = self.preprocess_dataset(self.train_dataset)
        if self.eval_dataset is not None:
            self.eval_dataset = self.preprocess_dataset(self.eval_dataset)

        # 准备数据加载器
        train_dataloader = self.get_train_dataloader()
        total_steps = len(train_dataloader) * self.args.num_train_epochs

        # 创建优化器和学习率调度器
        self.create_optimizer_and_scheduler(num_training_steps=total_steps)

        global_step = 0  # 全局步数计数器

        # 开始训练循环
        for epoch in range(self.args.num_train_epochs):
            if self.should_stop:
                print("训练提前停止。")
                break

            self.model.train()
            epoch_iterator = tqdm(
                train_dataloader, desc=f"Epoch {epoch + 1}/{self.args.num_train_epochs}"
            )

            for step, inputs in enumerate(epoch_iterator):
                if self.should_stop:
                    print("训练提前停止。")
                    break

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                loss = outputs.loss

                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                global_step += 1

                # 日志记录
                if global_step % self.args.logging_steps == 0:
                    print(f"Step {global_step}: loss = {loss.item()}")

                # 执行回调函数
                for callback in self.callbacks:
                    callback.on_step_end(self, loss, global_step)

                # 保存模型
                if global_step % self.args.save_steps == 0:
                    self.save_model()

                # 评估模型
                if (
                    self.args.evaluation_strategy == "steps"
                    and global_step % self.args.eval_steps == 0
                ):
                    self.evaluate()

            # 每个epoch结束后的操作
            if self.args.evaluation_strategy == "epoch":
                self.evaluate()

        # 训练结束后的操作
        self.save_model()

    def evaluate(self):
        """
        评估模型性能。
        """
        self.model.eval()
        eval_dataloader = self.get_eval_dataloader()
        total_eval_loss = 0

        for inputs in tqdm(eval_dataloader, desc="Evaluating"):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                loss = outputs.loss
                total_eval_loss += loss.item()

        avg_eval_loss = total_eval_loss / len(eval_dataloader)
        print(f"Evaluation loss: {avg_eval_loss}")

        # 执行回调函数
        for callback in self.callbacks:
            if hasattr(callback, "on_evaluate"):
                callback.on_evaluate(self, avg_eval_loss)

    def save_model(self):
        """
        保存当前模型到指定目录。
        """
        output_dir = self.args.output_dir
        self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
