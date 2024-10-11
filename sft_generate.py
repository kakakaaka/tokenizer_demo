# generate.py

from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    tokenizer = AutoTokenizer.from_pretrained('./sft_results')
    model = AutoModelForCausalLM.from_pretrained('./sft_results')

    # 添加 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = "写一首关于友谊的诗。"
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    output = model.generate(
        **inputs,
        max_length=100,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.pad_token_id  # 确保指定 pad_token_id
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print(generated_text)

if __name__ == '__main__':
    main()
