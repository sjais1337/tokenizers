import unicodedata
from collections import defaultdict, Counter
import argparse

class SP_BPE:

  class _Node:
    def __init__(self, value, prev=None, next=None):
      self.value = value
      self.prev = prev
      self.next = next

  def __init__(self):
    self.merges = {}
    self.vocabulary = {}
    self.ws = ' '

  def _normalize(self, text):
    normalized_text = unicodedata.normalize('NFKC', text).lower()
    
    return normalized_text.replace(' ', self.ws)
  
  def train(self, text, size):
    self.vocab = {
      **{
        0: b'<pad>',
        1: b'<unk>',
        2: b'<s>',
        3: b'</s>',
      },
      **{i+4: bytes([i]) for i in range(256)}
    }

    text_bytes = self._normalize(text).encode('utf-8')

    # Creating a double linked list 
    d_head = self._Node(None)
    prev_node = d_head
    for x in text_bytes:
      new = self._Node(x+4, prev=prev_node)
      prev_node.next = new
      prev_node = new
    head = d_head.next
    if head:
      head.prev = None


    # Creating an index of where each pair occurs, and their frequencies.
    freqs = Counter()
    p_nodes = defaultdict(set)
    node = head

    while node and node.next:
      pair = (node.value, node.next.value)
      freqs[pair] += 1
      p_nodes[pair].add(node)
      node = node.next
    
    num_merges = size - len(self.vocab)
    next_id = len(self.vocab)

    for i in range(num_merges):
      if not freqs:
        break
      
      best = max(freqs, key=freqs.get)

      self.merges[best] = next_id
      self.vocab[next_id] = self.vocab[best[0]] + self.vocab[best[1]]

      process_nodes = list(p_nodes.pop(best))
      del freqs[best]

      processed_nodes = set()

      for x in process_nodes:
        if x in processed_nodes or (x.next and x.next in processed_nodes):
          continue 

        if not x.next or (x.value, x.next.value) != best:
          continue
          
        prev_node = x.prev
        next_node = x.next
        next_next_node = next_node.next

        def update_stats(pair, modifyng_node, delta):
          if not pair: 
            return

          freqs[pair] += delta

          if delta > 0 :
            p_nodes[pair].add(modifyng_node)
          else:
            p_nodes[pair].discard(modifyng_node)
          if freqs[pair] <= 0:
            del freqs[pair]
            if pair in p_nodes:
              del p_nodes[pair]
  
        update_stats((prev_node.value, x.value) if prev_node else None, prev_node, - 1)
        update_stats((next_node.value, next_next_node.value) if next_next_node else None, next_node, - 1)

        x.value = next_id
        x.next = next_next_node
        if next_next_node:
          next_next_node.prev = x
        
        processed_nodes.add(x)
        processed_nodes.add(next_node)
        
        update_stats((prev_node.value, x.value) if prev_node else None, prev_node, 1)
        update_stats((x.value, next_next_node.value) if next_next_node else None, x, 1)

      next_id += 1

      if (i + 1) % 500 == 0:
        print(f"Merge {i+1}/{num_merges} complete.")
    
    print("Training complete.")
    self._structurize()
  
  def _structurize(self):
    self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
    self._inverse_vocab = {v: k for k, v in self.vocab.items()}
  
  def encode(self, text):
    if not self.merges:
      raise RuntimeError("Tokenizer has not been trained.")

    text = self._normalize(text)
    text_bytes = text.encode('utf-8')

    tokens = [b+4 for b in text_bytes]

    while len(tokens) > 1:
      pairs = {(tokens[i], tokens[i+1]): i for i in range(len(tokens) - 1)}
      best = min(pairs, key=lambda p: self.merge_ranks.get(p, float('inf')))

      if best not in self.merge_ranks:
        break

      idx = pairs[best]
      new_id = self.merges[best]
      tokens = tokens[:idx] + [new_id] + tokens[idx+2:]
    
    return tokens

  def decode(self, ids):
    bytes_chunks = [self.vocab.get(id, self.vocab[1]) for id in ids]
    text = b"".join(bytes_chunks).decode('utf-8', 'replace').replace(self.ws, ' ')
    return text

  def save(self, filename):
    with open(filename, 'w', encoding='utf-8') as f:
      for i in self.vocab.values():
        f.write(f'{i.decode('utf-8', 'replace')}\n')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--train", required=True)
  parser.add_argument("--input", required=True)
  parser.add_argument("--vocab_size", type=int, default=5000)
  args = parser.parse_args()

  vocab_size = args.vocab_size
  vocab_file = f'240981_assignment2_sp_vocab_{vocab_size}.txt'
  tokens_file = '240981_assignment2_sp_tokens.txt'
  detokenized_file = '240981_assignment2_sp_detokenized.txt'

  with open(args.train, 'r', encoding='utf-8') as f:
    corpus = f.read()
  
  tokenizer = SP_BPE()
  tokenizer.train(corpus, vocab_size)
  tokenizer.save(vocab_file)

  with open(args.input, 'r', encoding='utf-8') as f:
    sample_text = f.read()

  ids = tokenizer.encode(sample_text)

  with open(tokens_file, 'w', encoding='utf-8') as f:
    for i in ids:
      f.write(f'{tokenizer.vocab[i].decode('utf-8', 'replace')}\n')
  
  with open(detokenized_file, 'w', encoding='utf-8') as f:
    text = tokenizer.decode(ids)
    f.write(text)