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
            weights.append(prob)
            letters.append(l)
        
        return random.choices(letters, weights=weights, k=1)[0]

    def augment_text(self, text):
        
        src = list(text)
        tgt = list(text)
        counter = 0

        while src == tgt:
            tgt = list(text)
            swaps = random.randint(0, len(src) // 2) # ab --> ba
            replaces = random.randint(0, len(src) // 2) # a --> b
            deletes = random.randint(0, len(src) // 2) # ab --> a
            inserts = random.randint(0, len(src) // 2) # ab --> acb
            random.shuffle(self.operations)

            # swaps
            if self.operations[0]:
                for i in range(swaps):
                    pos = random.randint(0, len(src)-2)
                    src[pos], src[pos+1] = src[pos+1], src[pos]
            
            # replaces
            if self.operations[1]:
                for i in range(replaces):
                    pos = random.randint(0, len(src)-1)
                    new_char = self.get_letter(src[pos])
                    src[pos] = new_char

            # deletes
            if self.operations[2]:
                for i in range(deletes):
                    pos = random.randint(0, len(src)-1)
                    src.pop(pos)
            
            # inserts
            if self.operations[3]:
                for i in range(inserts):
                    pos = random.randint(0, len(src)-1)
                    new_char = self.get_letter(src[pos])
                    src.insert(pos, new_char)

            counter += 1
            if counter >= 10:
                return src
            
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
