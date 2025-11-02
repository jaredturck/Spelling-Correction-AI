import os, time, pickle

DATASETS_PATH = 'datasets/'

class Node:
    def __init__(self, word):
        self.word = word
        self.children = {}

class BKTree:
    def __init__(self):
        self.root = None
        self.tolerance = 2

        fname = os.path.join(DATASETS_PATH, 'bk_tree.pkl')
        if os.path.isfile(fname):
            with open(fname, 'rb') as file: 
                self.root = pickle.load(file).root
        else:
            self.generate_tree()

    def add(self, word):
        ''' Insert word into tree '''
        if self.root is None:
            self.root = Node(word)
            return

        node = self.root
        while True:
            d = self.edit_distance(word, node.word)
            child = node.children.get(d)
            if child is None:
                node.children[d] = Node(word)
                return
            node = child
    
    def generate_tree(self):
        ''' Add words from dictionary to BK-tree '''
        with open(os.path.join(DATASETS_PATH, 'words_alpha.txt'), 'r', encoding='utf-8') as file:
            start = time.time()
            for n,word in enumerate(file):
                self.add(word.strip())

                if time.time() - start > 10:
                    start = time.time()
                    print(f'[+] Added {n:,} words')
        
        # Save tree
        pickle.dump(self, open(os.path.join(DATASETS_PATH, 'bk_tree.pkl'), 'wb'))

    def get_similar_words(self, s):
        if self.root is None:
            return []
        out = []
        self.search(self.root, s, out)
        return out

    def search(self, node, s, out):
        d = self.edit_distance(node.word, s)
        if d <= self.tolerance:
            out.append(node.word)

        low = max(1, d - self.tolerance)
        high = d + self.tolerance
        for dist, child in node.children.items():
            if low <= dist <= high:
                self.search(child, s, out)

    def edit_distance(self, a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            ai = a[i - 1]
            for j in range(1, n + 1):
                cost = 0 if ai == b[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return dp[m][n]

if __name__ == "__main__":
    bk = BKTree()

    while True:
        q = input('> ')
        print([q, bk.get_similar_words(q)])
