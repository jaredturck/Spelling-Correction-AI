import csv
import random
import re
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import transformers, torch
from tqdm import tqdm


class SentenceGenerator:
    def __init__(self, batch_size, samples_per_word):
        self.batch_size = batch_size
        self.samples_per_word = samples_per_word
        self.words_per_batch = batch_size // (samples_per_word * 2)
        self.seen_misspellings = set()

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

        self.executor = ThreadPoolExecutor(max_workers=2)

        self.sentence_prompt = 'Return one sentence containing the word, sentence should be no more than 10 words: '
        self.misspelling_prompt = 'Return one realistic misspelling of the word. Return only the misspelled word: '
        self.retry_prompt = 'Return a different realistic misspelling of the word. Return only one word with no spaces or explanation: '

    def is_valid_misspelling(self, word, misspelling):
        return (
            misspelling
            and not any(character.isspace() for character in misspelling)
            and misspelling.lower() != word.lower()
        )

    def replace_word(self, sentence, word, misspelling):
        match = re.search(rf'\b{re.escape(word)}\b', sentence, re.IGNORECASE)

        if not match:
            return None

        matched_word = match.group()

        if matched_word.isupper():
            misspelling = misspelling.upper()
        elif matched_word[0].isupper():
            misspelling = misspelling.capitalize()

        return sentence[:match.start()] + misspelling + sentence[match.end():]

    def filter_rows(self, rows):
        accepted_rows = []
        retry_samples = []

        for word, sentence, incorrect_sentence, misspelling in rows:
            misspelling_key = (
                word.lower(),
                misspelling.lower(),
            )

            if misspelling_key in self.seen_misspellings:
                retry_samples.append((word, sentence))
                continue

            self.seen_misspellings.add(misspelling_key)

            accepted_rows.append((
                word,
                sentence,
                incorrect_sentence,
            ))

        return accepted_rows, retry_samples

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
            clean_up_tokenization_spaces=False,
        )

        sample_count = len(sample_words)

        sentences = [
            output[0]['generated_text'][-1]['content'].strip()
            for output in outputs[:sample_count]
        ]

        misspellings = [
            output[0]['generated_text'][-1]['content'].strip()
            for output in outputs[sample_count:]
        ]

        rows = []
        retry_samples = []

        for word, sentence, misspelling in zip(
            sample_words,
            sentences,
            misspellings,
        ):
            if not re.search(
                rf'\b{re.escape(word)}\b',
                sentence,
                re.IGNORECASE,
            ):
                retry_samples.append((word, None))
                continue

            if not self.is_valid_misspelling(word, misspelling):
                retry_samples.append((word, sentence))
                continue

            incorrect_sentence = self.replace_word(
                sentence,
                word,
                misspelling,
            )

            rows.append((
                word,
                sentence,
                incorrect_sentence,
                misspelling,
            ))

        return rows, retry_samples

    def retry_batch(self, pipe, samples):
        sentence_messages = [
            [
                {'role': 'system', 'content': self.sentence_prompt},
                {'role': 'user', 'content': word}
            ]
            for word, sentence in samples
            if sentence is None
        ]

        misspelling_messages = [
            [
                {'role': 'system', 'content': self.retry_prompt},
                {'role': 'user', 'content': word}
            ]
            for word, sentence in samples
        ]

        outputs = pipe(
            sentence_messages + misspelling_messages,
            batch_size=self.batch_size,
            clean_up_tokenization_spaces=False,
        )

        sentence_count = len(sentence_messages)
        sentence_outputs = outputs[:sentence_count]
        misspelling_outputs = outputs[sentence_count:]

        rows = []
        discarded = 0
        sentence_index = 0

        for sample, output in zip(samples, misspelling_outputs):
            word, sentence = sample

            if sentence is None:
                sentence = (
                    sentence_outputs[sentence_index][0]
                    ['generated_text'][-1]['content']
                    .strip()
                )

                sentence_index += 1

            misspelling = (
                output[0]['generated_text'][-1]['content'].strip()
            )

            if not re.search(
                rf'\b{re.escape(word)}\b',
                sentence,
                re.IGNORECASE,
            ):
                discarded += 1
                continue

            if not self.is_valid_misspelling(word, misspelling):
                discarded += 1
                continue

            incorrect_sentence = self.replace_word(
                sentence,
                word,
                misspelling,
            )

            rows.append((
                word,
                sentence,
                incorrect_sentence,
                misspelling,
            ))

        return rows, discarded

    def generate_clean_batch(self, pipe, words):
        messages = [
            [
                {'role': 'system', 'content': self.sentence_prompt},
                {'role': 'user', 'content': word}
            ]
            for word in words
        ]

        outputs = pipe(
            messages,
            batch_size=self.batch_size,
            clean_up_tokenization_spaces=False,
        )

        rows = []
        retry_words = []

        for word, output in zip(words, outputs):
            sentence = output[0]['generated_text'][-1]['content'].strip()

            if not re.search(
                rf'\b{re.escape(word)}\b',
                sentence,
                re.IGNORECASE,
            ):
                retry_words.append(word)
                continue

            rows.append((
                word,
                sentence,
                sentence,
            ))

        return rows, retry_words

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

    def generate_clean(self, words):
        batches = (
            words[index:index + self.batch_size]
            for index in range(0, len(words), self.batch_size)
        )

        futures = {}

        for gpu in range(2):
            word_batch = next(batches, None)

            if word_batch:
                future = self.executor.submit(
                    self.generate_clean_batch,
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
                        self.generate_clean_batch,
                        self.pipes[gpu],
                        word_batch,
                    )

                    futures[future] = gpu

    def retry(self, samples):
        for index in range(0, len(samples), self.batch_size):
            sample_batch = samples[index:index + self.batch_size]

            yield self.retry_batch(
                self.pipes[0],
                sample_batch,
            )


batch_size = 512
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

rows_skipped = 0
retry_samples = []
accepted_words = []

with open('sentence_dataset.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    writer.writerow([
        'word',
        'correct_sentence',
        'incorrect_sentence',
    ])

    for rows, failed_samples in tqdm(
        generator.generate(words),
        total=total_batches,
        desc='Batches',
    ):
        accepted_rows, duplicate_samples = generator.filter_rows(rows)

        writer.writerows(accepted_rows)
        accepted_words.extend(row[0] for row in accepted_rows)

        retry_samples.extend(failed_samples)
        retry_samples.extend(duplicate_samples)

    if retry_samples:
        retry_batches = (
            len(retry_samples) + batch_size - 1
        ) // batch_size

        for rows, discarded in tqdm(
            generator.retry(retry_samples),
            total=retry_batches,
            desc='Retries',
        ):
            accepted_rows, duplicate_samples = generator.filter_rows(rows)

            writer.writerows(accepted_rows)
            accepted_words.extend(row[0] for row in accepted_rows)

            rows_skipped += discarded
            rows_skipped += len(duplicate_samples)

    clean_words = random.sample(
        accepted_words,
        len(accepted_words) // 2,
    )

    pending_clean_words = clean_words

    with tqdm(
        total=len(clean_words),
        desc='Clean rows',
    ) as progress:
        while pending_clean_words:
            retry_clean_words = []

            for rows, failed_words in generator.generate_clean(
                pending_clean_words,
            ):
                writer.writerows(rows)
                progress.update(len(rows))
                retry_clean_words.extend(failed_words)

            pending_clean_words = retry_clean_words

print(f'Rows skipped: {rows_skipped}')
