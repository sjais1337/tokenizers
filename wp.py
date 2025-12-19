from collections import Counter, defaultdict
import heapq
import math
import argparse
import re 

class _Node:
  def __init__(self, value):
    self.value = value
    self.prev = None
    self.next = None

class _LL:
  def __init__(self):
    self.head = None
    self.tail = None
  
  def append(self, value):
    new = _Node(value)
    if not self.head:
      self.head = self.tail = new
    else:
      self.tail.next = new
      new.prev = self.tail
      self.tail = new

  def to_list(self):
    temp = []
    node = self.head
    while node:
      temp.append(node)
      node = node.next
    return temp

class WP:
  def __init__(self, size=5000):
    self.target = size
    self.vocab = {}
    self.freqs = Counter()
    self.splits = {}

    self.t_counts = defaultdict(int)  
    self.p_counts = defaultdict(int)
    self.pqueue = []
    self.tot = 0
  
  def _init_courpus(self, corpus):
    words = corpus.split()
    self.freqs = Counter(words)

    init = set("".join(words))

    special_tokens = ['<pad>', '<unk>', '<s>', '</s>']
    self.vocab = {token: i for i, token in enumerate(special_tokens)}

    for ch in sorted(list(init)):
      self.vocab[ch] = len(self.vocab)
      self.vocab[f"##{ch}"] = len(self.vocab)

  def _merge_tokens(self, t1, t2):
    """Correctly merge two tokens while preserving continuation markers."""
    if t1.startswith("##"):
      return "##" + t1[2:] + (t2[2:] if t2.startswith("##") else t2)
    else:
      return t1 + (t2[2:] if t2.startswith("##") else t2)

  def _populate(self):
    for word, freq in self.freqs.items():
      tokens = [word[0]] + [f"##{c}" for c in word[1:]]
      ll = _LL()
      self.splits[word] = ll

      self.tot += len(tokens) * freq

      for token in tokens:
        self.t_counts[token] += freq
        ll.append(token)
      
      for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i+1])
        self.p_counts[pair] += freq
        
    for pair, freq in self.p_counts.items():
      score = self._calculate_score(pair)
      if score is not None:
        merged = self._merge_tokens(pair[0], pair[1])
        heapq.heappush(self.pqueue, (-score, merged, pair))

  def _calculate_score(self, pair):
    # Uses the formula delta L = -f_new * log(f_new / N)
    f_new = self.p_counts.get(pair, 0)
    N = self.tot
    if f_new == 0 or N == 0:
      return None
    score = -f_new * math.log(f_new/N)
    return score

  def _merge(self, pair):
    t1, t2 = pair
    new = self._merge_tokens(t1, t2)

    if new not in self.vocab:
      self.vocab[new] = len(self.vocab)
    
    n = self.p_counts.get(pair, 0)
    if n > 0:
      self.t_counts[t1] -= n
      self.t_counts[t2] -= n
      self.t_counts[new] += n
      self.p_counts[pair] = 0
      self.tot -= n

    for word, freq in self.freqs.items():
      ll = self.splits[word]
      node = ll.head

      while node and node.next:
        if node.value == t1 and node.next.value == t2:
          if node.prev:
            left_p = (node.prev.value, node.value)
            self.p_counts[left_p] -= freq
          if node.next.next:
            right_p = (node.next.value, node.next.next.value)
            self.p_counts[right_p] -= freq
          
          merged = node
          second = node.next
          merged.value = new
          merged.next = second.next
          if second.next:
            second.next.prev = merged
          if ll.tail == second:
            ll.tail = merged

          if merged.prev:
            n_left_p = (merged.prev.value, merged.value)
            self.p_counts[n_left_p] += freq
            score = self._calculate_score(n_left_p)
            if score is not None:
              merged_str = self._merge_tokens(n_left_p[0], n_left_p[1])
              heapq.heappush(self.pqueue, (-score, merged_str, n_left_p))
          
          if merged.next:
            n_right_p = (merged.value, merged.next.value)
            self.p_counts[n_right_p] += freq
            score = self._calculate_score(n_right_p)
            if score is not None:
              merged_str = self._merge_tokens(n_right_p[0], n_right_p[1])
              heapq.heappush(self.pqueue, (-score, merged_str, n_right_p))
          
          node = merged

        if node:
          node = node.next

  def train(self, corpus):
    self._init_courpus(corpus)
    self._populate()

    while len(self.vocab) < self.target:
      best = None
      while self.pqueue:
        neg_score, merged, pair = heapq.heappop(self.pqueue)
        if self.p_counts.get(pair, 0) == 0:
          continue
        best = pair
        break
      if best is None:
        print("No more valid pairs to merge.")
        break

      self._merge(best)

      if len(self.vocab) % 500 == 0:
        print(f"  Vocabulary size: {len(self.vocab)}")
  
  def encode(self, text: str):
    def tokenize_word(word):
        subwords, start = [], 0
        while start < len(word):
            match = None
            for end in range(len(word), start, -1):  
                sub = word[start:end]
                tok = sub if start == 0 else f"##{sub}"
                if tok in self.vocab:
                    match = tok
                    break
            subwords.append(match if match else "<unk>")
            start += len(match.replace("##", "")) if match else len(word)
        return subwords

    tokens = []
    parts = re.split(r'(\s+)', text)
    for part in parts:
      if not part: continue
      if part.isspace():
        tokens.append(part)
      else:
        tokens.extend(tokenize_word(part))
    return tokens

  def decode(self, tokens):
    text_parts = []
    for i, token in enumerate(tokens):
        if token.startswith("##"):
            text_parts.append(token[2:])  # continuation of previous word
        else:
          text_parts.append(token)
    return "".join(text_parts)

  def save_vocab(self, path):
    sorted_v = sorted(self.vocab.items(), key=lambda item: item[1])
    with open(path, 'w', encoding='utf-8') as f:
      for token, _ in sorted_v:
        f.write(f'{token}\n')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--train', required=True)
  parser.add_argument('--input', required=True)
  parser.add_argument('--vocab_size', type=int, required=True)
  args = parser.parse_args()
  
  vocab_file = f'240981_assignment2_wp_vocab_{args.vocab_size}.txt'
  tokens_file = '240981_assignment2_wp_tokens.txt'
  detokenized_file = '240981_assignment2_wp_detokenized.txt'
  
  with open(args.train, 'r', encoding='utf-8') as f:
    corpus = f.read()
  
  tokenizer = WP(args.vocab_size)
  tokenizer.train(corpus)

  tokenizer.save_vocab(vocab_file)

  with open(args.input, 'r', encoding='utf-8') as f:
    sample_text = f.read()

  token_ids = tokenizer.encode(sample_text)

  with open(tokens_file, 'w', encoding='utf-8') as f:
    for i in token_ids:
      f.write(f'{i}\n')
  
  with open(detokenized_file, 'w', encoding='utf-8') as f:
    detokenized_text = tokenizer.decode(token_ids)
    f.write(detokenized_text)