import random, string, torch, os, time, datetime, math, re

charset = {i : n+1 for n,i in enumerate(string.ascii_lowercase)}
UNK_ID = len(charset) + 1
charset['<UNK>'] = UNK_ID

MAX_SAMPLES = 50_000_000

class DictionaryDataset():
    def __init__(self):
        self.operations = [1,1,1,0,0,0]
        self.aug_count = 5

        # Create euclidean distance lookup
        rows = ('qwertyuiop', 'asdfghjkl', 'zxcvbnm')
        offsets = (0, 0.5, 1)
        self.distance = {}
        for r,row in enumerate(rows):
            off = offsets[r]
            for c,ch in enumerate(row):
                self.distance[ch] = (off + c, r)
        
        self.phonetic_rules = [
            (r"tion", [("shun", 0.7), ("chun", 0.2), ("shon", 0.1)]),
            (r"ph",   [("f", 1.0)]),
            (r"ough", [("off", 0.4), ("aw", 0.25), ("oh", 0.2), ("uff", 0.15)]),
            (r"\bkn", [("n", 1.0)]),
            (r"\bwr", [("r", 1.0)]),
            (r"ee",   [("ea", 0.5), ("i", 0.3), ("e", 0.2)]),
            (r"oo",   [("u", 0.5), ("ou", 0.3), ("ew", 0.2)]),
            (r"c(?=[eiy])", [("s", 0.8), ("c", 0.2)]),
            (r"c",    [("k", 0.7), ("q", 0.1), ("c", 0.2)]),
            (r"x",    [("ks", 0.8), ("gz", 0.2)])
        ]

    def get_letter(self, letter):
        ''' Choose random letter weighted by euclidean distance '''
        weights = []
        letters = []
        for l in string.ascii_lowercase:
            (x1, y1), (x2, y2) = self.distance[letter], self.distance[l]
            prob = math.hypot(x1 - x2, y1 - y2)
            prob = max(math.exp(-1.2 * prob), 1e-3)
            weights.append(prob)
            letters.append(l)
        
        return random.choices(letters, weights=weights, k=1)[0]
    
    def phonetic_spelling(self, text):
        for p, alts in self.phonetic_rules:
            if random.random() < 0.25:
                choices, w = zip(*alts)
                text = re.sub(p, lambda _: random.choices(choices, weights=w, k=1)[0], text)

        text = re.sub(r'([a-z])\1', r'\1', text) # Replaces double letters with single
        text = re.sub(r'e\b', '', text) # Removes silent e at end of words
        return text

    def augment_text(self, text):

        if len(text) <= 1:
            return text
        
        if random.random() < 0.25:
            text = self.phonetic_spelling(text)
        
        src = list(text)
        tgt = list(text)

        self.operations = [random.choices([1,2,3,4], weights=[0.10, 0.25, 0.35, 0.35])[0] for i in range(random.randint(1, math.ceil(len(src) / 2)))]
        self.operations = self.operations + [0 for i in range(len(src) - len(self.operations))]
        random.shuffle(self.operations)

        if self.operations[0] != 0:
            random.shuffle(self.operations)

        for pos in range(len(src)):
            if pos + 1 >= len(src):
                break

            match self.operations[pos]:
                case 1:
                    src[pos], src[pos+1] = src[pos+1], src[pos]
                case 2:
                    src[pos] = self.get_letter(src[pos])
                case 3:
                    src.pop(pos)
                case 4:
                    src.insert(pos, self.get_letter(src[pos]))
        
        if src == tgt:
            pos = random.choice(list(range(len(src))))
            src[pos] = self.get_letter(src[pos])
            
        return src
    
    def read_data(self):

        with open('words_alpha.txt', 'r', encoding='utf-8') as file, \
            open('sample_data.txt', 'w', encoding='utf-8') as output_file:
            file_content = file.read().split('\n')
            random.shuffle(file_content)

            no_words = min(len(file_content), MAX_SAMPLES // self.aug_count)
            start = time.time()

            for word_id,row in enumerate(file_content[:no_words]):
                for _ in range(self.aug_count):
                    output_file.write(f'{row},{''.join(self.augment_text(row))}\n')
                
                if time.time() - start > 10:
                    start = time.time()
                    print(f'[+] Processed {word_id:,} words')
                    break

if __name__ == '__main__':
    dataset = DictionaryDataset()
    dataset.read_data()
