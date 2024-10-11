# main.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from training_args import TrainingArguments
from sft_trainer import SFTTrainer
from data_collator import DataCollator
from callbacks import LoggingCallback, EarlyStoppingCallback

def main():
    # 加载预训练的分词器和模型
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('gpt2')

    # **添加 pad_token 并调整模型嵌入层**
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # 加载 Alpaca 数据集
    dataset = load_dataset('tatsu-lab/alpaca')
    train_dataset = dataset['train']
    eval_dataset = dataset['train'].train_test_split(test_size=0.1)['test']

    # 定义训练参数
    args = TrainingArguments(
        output_dir='./sft_results',
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=50,
        evaluation_strategy='steps',
        eval_steps=20,
        seed=42
    )

    # 定义回调函数列表
    callbacks = [
        LoggingCallback(),
        EarlyStoppingCallback(patience=3)
    ]

    # 定义数据整理器
    data_collator = DataCollator(tokenizer=tokenizer, padding=True)

    # 创建SFTTrainer实例
    trainer = SFTTrainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks,
        max_seq_length=512,
        dataset_text_field='text',
        packing=False
    )

    # 开始训练
    trainer.train()

if __name__ == '__main__':
    main()
