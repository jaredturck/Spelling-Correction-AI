import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

import transformers, torch
from tqdm import tqdm


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

        for pipe in self.pipes:
            pipe.generation_config.max_length = None
            pipe.generation_config.max_new_tokens = 32
            pipe.generation_config.do_sample = True
            pipe.generation_config.temperature = 0.9
            pipe.generation_config.top_p = 0.95
            pipe.tokenizer.clean_up_tokenization_spaces = False

        self.executors = [
            ThreadPoolExecutor(max_workers=1)
            for _ in range(2)
        ]

        self.system_prompt = 'Return one sentence containing the word, sentence should be no more than 10 words: '

    def generate_batch(self, pipe, words):
        sample_words = words * self.samples_per_word

        messages = (
            [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': word}
            ]
            for word in sample_words
        )

        outputs = list(pipe(
            messages,
            batch_size=self.batch_size,
        ))

        sentences = [
            output[0]['generated_text'][-1]['content'].strip()
            for output in outputs
        ]

        return list(zip(sample_words, sentences))

    def generate(self, words):
        words_per_batch = self.batch_size // self.samples_per_word
        futures = []

        for index in range(0, len(words), words_per_batch):
            gpu = len(futures) % 2
            word_batch = words[index:index + words_per_batch]

            futures.append(self.executors[gpu].submit(
                self.generate_batch,
                self.pipes[gpu],
                word_batch,
            ))

        return futures


batch_size = 500
samples_per_word = 3

generator = SentenceGenerator(batch_size, samples_per_word)

with open('common_english_words.txt', encoding='utf-8') as file:
    words = file.read().splitlines()

with open('sentence_dataset.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['word', 'sentence'])

    futures = generator.generate(words)

    for future in tqdm(
        as_completed(futures),
        total=len(futures),
        desc='Batches',
    ):
        writer.writerows(future.result())
