import random, string, torch, os, time, datetime, math

charset = {i : n+1 for n,i in enumerate(string.ascii_lowercase)}
UNK_ID = len(charset) + 1
charset['<UNK>'] = UNK_ID

MAX_SAMPLES = 50_000_000

class DictionaryDataset():
    def __init__(self):
        self.operations = [1,1,1,0,0,0]
        self.aug_count = 5

        # Setup query euclidean distance lookup
        rows = ('qwertyuiop', 'asdfghjkl', 'zxcvbnm')
        offsets = (0, 0.5, 1)
        self.distance = {}
        for r,row in enumerate(rows):
            off = offsets[r]
            for c,ch in enumerate(row):
                self.distance[ch] = (off + c, r)
    
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

    def augment_text(self, text):
        
        src = list(text)
        tgt = list(text)

        self.operations = [random.choice([1,2,3,4]) for i in range(random.randint(1, math.ceil(len(src) / 2)))]
        self.operations = self.operations + [0 for i in range(len(src) - len(self.operations))]
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
