import csv, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

# accelerate launch --multi_gpu --num_processes=2 --mixed_precision=bf16 train.py

model_name = 'Qwen/Qwen3-0.6B'
dataset_path = 'sentence_dataset.csv'

tokenizer = AutoTokenizer.from_pretrained(model_name)

data = []

with open(dataset_path, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)

    for row in reader:
        data.append({
            'prompt': [
                {
                    'role': 'user',
                    'content': row['incorrect_sentence'],
                }
            ],
            'completion': [
                {
                    'role': 'assistant',
                    'content': row['correct_sentence'],
                }
            ],
        })

dataset = Dataset.from_list(data)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    attn_implementation='flash_attention_2',
)

sft_config = SFTConfig(
    output_dir='./qwen_spelling_model',
    bf16=True,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    num_train_epochs=1,
    completion_only_loss=True,
    max_length=128,
    packing=True,
    gradient_checkpointing=False,
    ddp_find_unused_parameters=False,
    logging_steps=50,
    save_strategy='steps',
    save_steps=200,
    save_total_limit=3,
    dataloader_num_workers=8,
    dataloader_persistent_workers=True,
    dataloader_prefetch_factor=4,
    dataset_num_proc=24,
    eos_token=tokenizer.eos_token,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    processing_class=tokenizer,
)

trainer.train()

trainer.save_model('./qwen_spelling_model/final')
tokenizer.save_pretrained('./qwen_spelling_model/final')
