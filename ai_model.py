import torch, random
from torch.utils.data import Dataset

charset = {chr(i) : n for n,i in enumerate(range(32, 126))}

class WikiDataset(Dataset):
    def __init__(self):
        self.training_data = []
        self.buffer_size = 1024 * 1024 * 10  # 10 MB buffer size
    
    def __len__(self):
        return len(self.training_data)
    
    def __getitem__(self, idx):
        return self.training_data[idx]
    
    def augment_text(self, text):
        
        src = list(text)
        tgt = list(text)
        counter = 0

        while src == tgt:
            tgt = list(text)
            swaps = random.randint(0, len(src) // 3) # ab --> ba
            replaces = random.randint(0, len(src) // 3) # a --> b
            deletes = random.randint(0, len(src) // 3) # ab --> a
            inserts = random.randint(0, len(src) // 3) # ab --> acb
            operations = [random.randint(0,1) for i in range(4)] # which operations to apply

            # swaps
            if operations[0]:
                for i in range(swaps):
                    pos = random.randint(0, len(src)-2)
                    src[pos], src[pos+1] = src[pos+1], src[pos]
            
            # replaces
            if operations[1]:
                for i in range(replaces):
                    pos = random.randint(0, len(src)-1)
                    new_char = random.choice(list(charset.keys()))
                    src[pos] = new_char

            # deletes
            if operations[2]:
                for i in range(deletes):
                    pos = random.randint(0, len(src)-1)
                    src.pop(pos)
            
            # inserts
            if operations[3]:
                for i in range(inserts):
                    pos = random.randint(0, len(src))
                    new_char = random.choice(list(charset.keys()))
                    src.insert(pos, new_char)

            counter += 1
            if counter >= 10:
                return src
            
        return src

    def read_data(self):
        with open('datasets/wiki_dump_1.txt', 'r', encoding='utf-8') as file:
            file_content = '\n'
            while file_content:
                file_content = file.read(self.buffer_size)
                words = file_content.split(' ')
                seek = 0

                for word in words:
                    seek += len(word) + 1
                    context_window = [charset[i] for i in file_content[max(seek - 32,0) : min(seek + 32, len(file_content))]]
                    src_word = [charset[i] for i in self.augment_text(word)]
                    tgt_word = [charset[i] for i in word]

                    print([src_word, context_window], tgt_word)

                    self.training_data.append(([src_word, context_window],tgt_word))
                    input('STOP')

if __name__ == "__main__":
    dataset = WikiDataset()
    dataset.read_data()
    