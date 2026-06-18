import csv
from concurrent.futures import ThreadPoolExecutor
import transformers, torch
from tqdm import tqdm

batch_size = 64
samples_per_word = 3

class SentenceGenerator:
    def __init__(self, batch_size, samples_per_word):
        self.batch_size = batch_size
        self.samples_per_word = samples_per_word

        self.pipes = [
            transformers.pipeline(
                'text-generation',
                model='Qwen/Qwen3-4B-Instruct-2507',
                device=device,
                dtype=torch.bfloat16,
            )
            for device in range(2)
        ]

        self.executor = ThreadPoolExecutor(max_workers=2)
        self.system_prompt = 'Return one sentence containing the word: '

    def generate_batch(self, pipe, words):
        sample_words = words * self.samples_per_word

        messages = [
            [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': word}
            ]
            for word in sample_words
        ]

        outputs = pipe(
            messages,
            batch_size=self.batch_size,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
        )

        sentences = [
            output[0]['generated_text'][-1]['content'].strip()
            for output in outputs
        ]

        return list(zip(sample_words, sentences))

    def generate(self, words):
        middle = len(words) // 2

        results = list(self.executor.map(
            self.generate_batch,
            self.pipes,
            [words[:middle], words[middle:]],
        ))

        return results[0] + results[1]

generator = SentenceGenerator(batch_size, samples_per_word)

with open('common_english_words.txt', encoding='utf-8') as file:
    words = file.read().splitlines()

with open('sentence_dataset.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['word', 'sentence'])

    words_per_batch = batch_size // samples_per_word * 2

    for index in tqdm(range(0, len(words), words_per_batch), desc='Batches'):
        word_batch = words[index:index + words_per_batch]
        writer.writerows(generator.generate(word_batch))
