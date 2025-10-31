import torch, random, time, sys, os, datetime
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from unidecode import unidecode

charset = {chr(i) : n+1 for n,i in enumerate(range(32, 127))}
UNK_ID = len(charset) + 1
charset['<UNK>'] = UNK_ID

WEIGHTS_PATH = 'weights/'

MAX_SAMPLES = 1_000_000
CONTEXT_LEN = 64
WORD_LEN = 16
VOCAB_SIZE = max(charset.values()) + 1
DEVICE = 'cuda'
BATCH_SIZE = 16384
TARGET_LOSS = 0.1

class WikiDataset(Dataset):
    def __init__(self):
        self.training_data = []
        self.buffer_size = 1024 * 1024 * 16  # 16 MB buffer size
    
    def __len__(self):
        return len(self.training_data)
    
    def __getitem__(self, idx):
        return self.training_data[idx]
    
    def collate_fn(self, batch):
        x_word, x_context, y = zip(*batch)
        x_word = pad_sequence(x_word, batch_first=True, padding_value=0)
        x_context = pad_sequence(x_context, batch_first=True, padding_value=0)
        y = pad_sequence(y, batch_first=True, padding_value=0)

        T = x_word.size(1)
        if y.size(1) < T:
            pad = torch.zeros(y.size(0), T - y.size(1))
            y = torch.cat([y, pad], dim=1)
        elif y.size(1) > T:
            y = y[:, :T]

        return x_word.long(), x_context.long(), y.long()
    
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
            start = time.time()
            while file_content:
                file_content = unidecode(file.read(self.buffer_size))
                words = file_content.split(' ')
                seek = 0
                
                for word in words:
                    seek += len(word) + 1
                    context_window = [charset.get(i,UNK_ID) for i in file_content[max(seek-CONTEXT_LEN//2,0) : min(seek+CONTEXT_LEN//2, len(file_content))]]
                    src_word = [charset.get(i,UNK_ID) for i in self.augment_text(word)][:WORD_LEN]
                    tgt_word = [charset.get(i,UNK_ID) for i in word][:WORD_LEN]
                    self.training_data.append((torch.tensor(src_word), torch.tensor(context_window), torch.tensor(tgt_word)))

                    if time.time() - start > 10:
                        print(f'[+] Processed {len(self.training_data):,} samples')
                        start = time.time()
                    
                    if len(self.training_data) >= MAX_SAMPLES:
                        print(f'[+] Finished processing {len(self.training_data):,} samples')
                        return

class SpellingModel(Module):
    def __init__(self):
        super().__init__()
        Module.train(self, True)
        self.dataset = WikiDataset()
        self.dropout = 0.1

        self.embedding = torch.nn.Embedding(VOCAB_SIZE+1, WORD_LEN, padding_idx=0)
        self.embedding_context = torch.nn.Embedding(VOCAB_SIZE+1, CONTEXT_LEN, padding_idx=0)
        self.context_lstm = torch.nn.LSTM(input_size=CONTEXT_LEN, hidden_size=WORD_LEN, num_layers=2, batch_first=True, bidirectional=False)
        self.main_lstm = torch.nn.LSTM(input_size=WORD_LEN, hidden_size=WORD_LEN, num_layers=6, batch_first=True, bidirectional=False)
        self.dropout = torch.nn.Dropout(self.dropout)
        self.out_proj = torch.nn.Linear(WORD_LEN, VOCAB_SIZE+1)
    
    def forward(self, x):
        src_word, context_window = x

        _, (context_out, _) = self.context_lstm(self.embedding_context(context_window))
        w_emb = self.embedding(src_word)

        logits, _ = self.main_lstm(
            self.dropout(
                w_emb + context_out[-1].unsqueeze(1).expand(-1, w_emb.size(1), -1)
            )
        )

        return self.out_proj(logits)
    
    def train_model(self):
        self.dataset.read_data()
        self.dataloader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=self.dataset.collate_fn, 
            num_workers=os.cpu_count()-1, pin_memory=True, prefetch_factor=8, persistent_workers=True, drop_last=True)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.load_weights()
        start = time.time()
        save_start = time.time()

        for epoch in range(100):
            total_loss = 0.0
            for n,batch in enumerate(self.dataloader):

                src_word, src_context, target = batch
                src_word = src_word.to(DEVICE, non_blocking=True)
                src_context = src_context.to(DEVICE, non_blocking=True)
                target = target.to(DEVICE, non_blocking=True)

                self.optimizer.zero_grad()
                logits = self.forward((src_word, src_context))
                B, T, V = logits.shape

                loss = loss_func(logits.view(B*T, V), target.view(B*T))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                if time.time() - start > 10:
                    print(f'[+] Batch {n+1} of {len(self.dataloader)}, loss: {loss.item():.4f}')
                    start = time.time()

                    if time.time() - save_start > 600:
                        self.save_weights()
                        save_start = time.time()
                        print('[+] Saved weights')

            avg_loss = total_loss / len(self.dataloader)
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

            if avg_loss <= TARGET_LOSS:
                print('[+] Finished training')
                return
    
    def save_weights(self):
        fname = f'weights_{datetime.datetime.now().strftime('%d-%b-%Y_%H-%M')}.pt'
        torch.save({
            'weights': self.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, os.path.join(WEIGHTS_PATH, fname))
    
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

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        try:
            model = SpellingModel().to(DEVICE)
            model.train_model()
        except KeyboardInterrupt:
            model.save_weights()
    else:
        model = SpellingModel().to(DEVICE)
