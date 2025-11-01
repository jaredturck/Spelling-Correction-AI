import torch, random, time, sys, os, datetime
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
import string

charset = {i : n+1 for n,i in enumerate(string.ascii_lowercase)}
UNK_ID = len(charset) + 1
charset['<UNK>'] = UNK_ID

WEIGHTS_PATH = 'weights/'
DATASETS_PATH = 'datasets/'

MAX_SAMPLES = 1_000_000
CONTEXT_LEN = 64
WORD_LEN = 16
VOCAB_SIZE = max(charset.values()) + 1
OUTPUT_EMB_SIZE = 370_105
DEVICE = 'cuda'
BATCH_SIZE = 256
TARGET_LOSS = 0.1
DROPOUT = 0

class DictionaryDataset(Dataset):
    def __init__(self):
        self.operations = [1,1,1,0,0,0]
        self.aug_count = 100
    
    def __len__(self):
        return self.x.size(0)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
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
                    new_char = random.choice(list(charset.keys()))
                    src[pos] = new_char

            # deletes
            if self.operations[2]:
                for i in range(deletes):
                    pos = random.randint(0, len(src)-1)
                    src.pop(pos)
            
            # inserts
            if self.operations[3]:
                for i in range(inserts):
                    pos = random.randint(0, len(src))
                    new_char = random.choice(list(charset.keys()))
                    src.insert(pos, new_char)

            counter += 1
            if counter >= 10:
                return src
            
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
                        self.x[counter, :src_word.size(0)] = src_word
                        self.y[counter] = word_id + 1
                        counter += 1
                    
                    if time.time() - start > 10:
                        start = time.time()
                        print(f'[+] Processed {self.x.size(0):,} samples')
            
            print(f'[+] Loaded {self.x.size(0):,} samples')
            torch.save({'x' : self.x, 'y' : self.y}, os.path.join(DATASETS_PATH, f'tensors_{datetime.datetime.now().strftime("%d-%b-%Y_%H-%M")}.pt'))

class SpellingModel(Module):
    def __init__(self):
        super().__init__()
        Module.train(self, True)
        self.dataset = DictionaryDataset()
        self.dropout = DROPOUT
        self.optimizer = None
        self.d_model = 256

        self.src_embedding = torch.nn.Embedding(VOCAB_SIZE+1, self.d_model, padding_idx=0)
        self.tgt_embedding = torch.nn.Embedding(OUTPUT_EMB_SIZE+1, WORD_LEN+1, padding_idx=0)

        self.main_lstm = torch.nn.LSTM(input_size=self.d_model, hidden_size=VOCAB_SIZE+1, num_layers=8, batch_first=True, bidirectional=False)
        self.out_proj = torch.nn.Linear(VOCAB_SIZE+1, OUTPUT_EMB_SIZE+1)
        self.dropout = torch.nn.Dropout(self.dropout)
    
    def forward(self, x):

        logits, _ = self.main_lstm(
            self.dropout(
                self.src_embedding(x)
            )
        )
        return self.out_proj(logits[:, -1, :])
    
    def train_model(self):
        self.dataset.read_data()
        self.dataloader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        loss_func = torch.nn.CrossEntropyLoss()
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
                logits = self.forward(src)

                loss = loss_func(logits, tgt)
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
            context_window = torch.tensor([[charset.get(i, UNK_ID) for i in src]]).to(DEVICE)
            logits = self.forward((src_word, context_window))
            predicted_ids = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
            predicted_chars = [list(charset.keys())[list(charset.values()).index(i)] if i in charset.values() else '' for i in predicted_ids]
        
        txt = ''.join(predicted_chars).strip()
        print(txt)

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
        while True:
            txt = input('> ')
            model.predict(txt)
