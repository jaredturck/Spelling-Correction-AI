import csv
import re
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import transformers, torch
from tqdm import tqdm


class SentenceGenerator:
    def __init__(self, batch_size, samples_per_word):
        self.batch_size = batch_size
        self.samples_per_word = samples_per_word
        self.words_per_batch = batch_size // (samples_per_word * 2)

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

        self.executor = ThreadPoolExecutor(max_workers=2)

        self.sentence_prompt = 'Return one sentence containing the word, sentence should be no more than 10 words: '
        self.misspelling_prompt = 'Return one realistic misspelling of the word. Return only the misspelled word: '

    def replace_word(self, sentence, word, misspelling):
        match = re.search(rf'\b{re.escape(word)}\b', sentence, re.IGNORECASE)

        if not match:
            return sentence

        matched_word = match.group()

        if matched_word.isupper():
            misspelling = misspelling.upper()
        elif matched_word[0].isupper():
            misspelling = misspelling.capitalize()

        return sentence[:match.start()] + misspelling + sentence[match.end():]

    def generate_batch(self, pipe, words):
        sample_words = words * self.samples_per_word

        sentence_messages = [
            [
                {'role': 'system', 'content': self.sentence_prompt},
                {'role': 'user', 'content': word}
            ]
            for word in sample_words
        ]

        misspelling_messages = [
            [
                {'role': 'system', 'content': self.misspelling_prompt},
                {'role': 'user', 'content': word}
            ]
            for word in sample_words
        ]

        outputs = pipe(
            sentence_messages + misspelling_messages,
            batch_size=self.batch_size,
        )

        sample_count = len(sample_words)

        sentences = [
            output[0]['generated_text'][-1]['content'].strip()
            for output in outputs[:sample_count]
        ]

        misspellings = [
            output[0]['generated_text'][-1]['content'].strip().strip('.,!?;:"\'')
            for output in outputs[sample_count:]
        ]

        incorrect_sentences = [
            self.replace_word(sentence, word, misspelling)
            for word, sentence, misspelling in zip(
                sample_words,
                sentences,
                misspellings,
            )
        ]

        return list(zip(
            sample_words,
            sentences,
            incorrect_sentences,
        ))

    def generate(self, words):
        batches = (
            words[index:index + self.words_per_batch]
            for index in range(0, len(words), self.words_per_batch)
        )

        futures = {}

        for gpu in range(2):
            word_batch = next(batches, None)

            if word_batch:
                future = self.executor.submit(
                    self.generate_batch,
                    self.pipes[gpu],
                    word_batch,
                )

                futures[future] = gpu

        while futures:
            completed, _ = wait(
                futures,
                return_when=FIRST_COMPLETED,
            )

            for future in completed:
                gpu = futures.pop(future)

                yield future.result()

                word_batch = next(batches, None)

                if word_batch:
                    future = self.executor.submit(
                        self.generate_batch,
                        self.pipes[gpu],
                        word_batch,
                    )

                    futures[future] = gpu


batch_size = 500
samples_per_word = 3
word_limit = 1000

generator = SentenceGenerator(batch_size, samples_per_word)

with open('common_english_words.txt', encoding='utf-8') as file:
    words = file.read().splitlines()

if word_limit:
    words = words[:word_limit]

total_batches = (
    len(words) + generator.words_per_batch - 1
) // generator.words_per_batch

with open('sentence_dataset.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow([
        'word',
        'correct_sentence',
        'incorrect_sentence',
    ])

    for rows in tqdm(
        generator.generate(words),
        total=total_batches,
        desc='Batches',
    ):
        writer.writerows(rows)
