import csv
import transformers, torch


class SentenceGenerator:
    def __init__(self):
        self.pipe = transformers.pipeline(
            'text-generation',
            model='Qwen/Qwen3-4B-Instruct-2507',
            device=0,
            dtype=torch.bfloat16,
        )

        self.system_prompt = 'Return one sentence containing the word: '

    def generate(self, words):
        sample_words = words * 3

        messages = [
            [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': word}
            ]
            for word in sample_words
        ]

        outputs = self.pipe(
            messages,
            batch_size=len(messages),
            max_new_tokens=64,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
        )

        sentences = [
            output[0]['generated_text'][-1]['content'].strip()
            for output in outputs
        ]

        return zip(sample_words, sentences)

generator = SentenceGenerator()

with open('common_english_words.txt', encoding='utf-8') as file:
    words = file.read().splitlines()

with open('sentence_dataset.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['word', 'sentence'])

    for index in range(0, len(words), 50):
        word_slice = words[index:index + 50]
        writer.writerows(generator.generate(word_slice))
