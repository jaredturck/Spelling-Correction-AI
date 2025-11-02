import torch, random, time, sys, os, datetime, platform, math, re
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from bk_tree import BKTree
import string

charset = {i : n+1 for n,i in enumerate(string.ascii_lowercase)}
UNK_ID = len(charset) + 1
charset['<UNK>'] = UNK_ID

WEIGHTS_PATH = 'weights/'
DATASETS_PATH = 'datasets/'

MAX_SAMPLES = 50_000_000
CONTEXT_LEN = 64
WORD_LEN = 16
VOCAB_SIZE = max(charset.values()) + 1
OUTPUT_EMB_SIZE = 370_107
DEVICE = 'cuda'
TARGET_LOSS = 0.1
DROPOUT = 0

if platform.uname().node == 'Jared-PC':
    BATCH_SIZE = 512
else:
    BATCH_SIZE = 3200

class DictionaryDataset(Dataset):
    def __init__(self):
        self.operations = [1,1,1,0,0,0]
        self.aug_count = 100

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
    
    def __len__(self):
        return self.x.size(0)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def phonetic_spelling(self, text):
        for p, alts in self.phonetic_rules:
            if random.random() < 0.25:
                choices, w = zip(*alts)
                text = re.sub(p, lambda _: random.choices(choices, weights=w, k=1)[0], text)

        text = re.sub(r'([a-z])\1', r'\1', text) # Replaces double letters with single
        text = re.sub(r'e\b', '', text) # Removes silent e at end of words
        return text
    
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

        if len(text) <= 1:
            return text
        
        if random.random() < 0.25:
            original_text = text
            text = self.phonetic_spelling(text)
            if len(text) == 0:
                text = original_text
        
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

        dataset_files = [os.path.join(DATASETS_PATH, file) for file in os.listdir(DATASETS_PATH) if file.endswith('.pt')]
        max_file = max(dataset_files, key=os.path.getctime) if dataset_files else None

        if max_file:
            file = torch.load(max_file, map_location=DEVICE)
            self.x = file['x']
            self.y = file['y']
            print(f'[+] Loaded {self.x.size(0):,} samples')

        else:
            with open(os.path.join(DATASETS_PATH, 'words_alpha.txt'), 'r', encoding='utf-8') as file:
                file_content = file.read().split('\n')

                no_words = min(len(file_content), MAX_SAMPLES // self.aug_count)
                start = time.time()
                self.x = torch.zeros((no_words * self.aug_count, WORD_LEN), dtype=torch.long)
                self.y = torch.empty((no_words * self.aug_count,), dtype=torch.long)
                counter = 0

                for word_id,row in enumerate(file_content[:no_words]):
                    for _ in range(self.aug_count):
                        src_list = [charset.get(i,UNK_ID) for i in self.augment_text(row)][:WORD_LEN]
                        src_word = torch.tensor(src_list, dtype=torch.long)
                        if len(src_list) == 0:
                            print('Empty line')
                            continue

                        self.x[counter, :src_word.size(0)] = src_word
                        self.y[counter] = word_id
                        counter += 1
                        
                    
                    if time.time() - start > 10:
                        start = time.time()
                        print(f'[+] Processed {counter:,} samples')
            
            print(f'[+] Loaded {counter:,} samples')
            torch.save(
                {'x' : self.x, 'y' : self.y}, 
                os.path.join(DATASETS_PATH, f'tensors_{datetime.datetime.now().strftime("%d-%b-%Y_%H-%M")}_{counter}_.pt')
            )
    
    def generate_lookup(self):
        self.word2int_lookup = {}
        self.int2word_lookup = {}
        with open(os.path.join(DATASETS_PATH, 'words_alpha.txt'), 'r', encoding='utf-8') as file:
            file_content = file.read().split('\n')
            for word_id,row in enumerate(file_content):
                self.word2int_lookup[row] = word_id
                self.int2word_lookup[word_id] = row

class SpellingModel(Module):
    def __init__(self):
        super().__init__()
        Module.train(self, True)
        self.dataset = DictionaryDataset()
        self.bk_tree = BKTree()
        self.dropout = DROPOUT
        self.optimizer = None
        self.d_model = 256

        self.src_embedding = torch.nn.Embedding(VOCAB_SIZE+1, self.d_model, padding_idx=0)

        self.main_lstm = torch.nn.LSTM(input_size=self.d_model, hidden_size=self.d_model, num_layers=8, batch_first=True, bidirectional=False)
        self.out_proj = torch.nn.Linear(self.d_model, 1)
        self.dropout = torch.nn.Dropout(self.dropout)

        self.adaptive_softmax = torch.nn.AdaptiveLogSoftmaxWithLoss(
            in_features=self.d_model, 
            n_classes=OUTPUT_EMB_SIZE, 
            cutoffs=[50_000, 200_000], 
            div_value=4.0
        )
    
    def forward(self, x):
        logits, (h_n, _) = self.main_lstm(
            self.dropout(
                self.src_embedding(x)
            )
        )
        mask = (x != 0)
        scores = self.out_proj(logits).squeeze(-1).masked_fill(~mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        features = torch.bmm(weights.unsqueeze(1), logits).squeeze(1)
        return features
    
    def train_model(self):
        self.dataset.read_data()
        self.dataloader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        self.load_weights()
        self.train()
        start = time.time()
        save_start = time.time()

        print('[+] Starting training')
        for epoch in range(10000):
            total_loss = 0.0
            for n,(src,tgt) in enumerate(self.dataloader):

                src = src.to(DEVICE, non_blocking=True)
                tgt = tgt.to(DEVICE, non_blocking=True)

                self.optimizer.zero_grad() 
                feat = self.forward(src) 
                out = self.adaptive_softmax(feat, tgt)

                loss = out.loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                if time.time() - start > 10:
                    print(f'[+] Batch {n+1:,} of {len(self.dataloader):,}, loss: {loss.item():.4f}')
                    start = time.time()

                    if time.time() - save_start > 600:
                        self.save_weights()
                        save_start = time.time()
                        print('[+] Saved weights')

            avg_loss = total_loss / len(self.dataloader)
            print(f'[+] Epoch {epoch+1}, Loss: {avg_loss:.4f}')

            if avg_loss <= TARGET_LOSS:
                print('[+] Finished training')
                return
    
    def save_weights(self):
        fname = f'weights_{datetime.datetime.now().strftime('%d-%b-%Y_%H-%M')}.pt'
        torch.save({
            'weights': self.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, os.path.join(WEIGHTS_PATH, fname))
        print(f'[+] Saved weights {fname}')
    
    def load_weights(self):
        files = [os.path.join(WEIGHTS_PATH, file) for file in os.listdir(WEIGHTS_PATH) if file.endswith('.pt')]
        if files:
            max_file = max(files, key=os.path.getctime)
            weights_data = torch.load(max_file, map_location=DEVICE)
            if 'weights' in weights_data:
                self.load_state_dict(weights_data['weights'])
                print(f'[+] Loaded weights from {max_file}')

            if self.optimizer and 'optimizer' in weights_data:
                self.optimizer.load_state_dict(weights_data['optimizer'])
                print(f'[+] Loaded optimizer state from {max_file}')
    
    def predict(self, src):
        self.eval()
        with torch.no_grad():
            src_word = torch.tensor([[charset.get(i, UNK_ID) for i in src]]).to(DEVICE)
            logits = self.forward(src_word)
            prob = self.adaptive_softmax.log_prob(logits).squeeze(0)

            similar_words = self.bk_tree.get_similar_words(src)

            word_ids = [self.dataset.word2int_lookup[i] for i in similar_words if i in self.dataset.word2int_lookup]
            if word_ids:
                mask = torch.full_like(prob, float('-inf'))
                idx = torch.tensor(word_ids, device=DEVICE)
                mask[idx] = 0
                prob = prob + mask
            
            
            _, ids = torch.topk(prob, k=5, dim=-1)
            suggestions = [self.dataset.int2word_lookup[int(i.item())] for i in ids]
            return suggestions

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        try:
            model = SpellingModel().to(DEVICE)
            model.train_model()
        except KeyboardInterrupt:
            model.save_weights()
    else:
        model = SpellingModel().to(DEVICE)
        model.load_weights()
        model.dataset.generate_lookup()
        while True:
            txt = input('> ')
            model.predict(txt)
