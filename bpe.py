import re
import collections
import argparse
import heapq

# TODO: Implement Priority Queue.

class BPE:
  def __init__(self):
    self.merges = {}
    self.vocabulary = {}

  # The rank of the merges is found by just the order in which the merges are found
  # the first merge has the highest rank, second as the second highest and so on.

  def train(self, text, size):
    # BPE finds the most frequent subword byte pair in the corpus, and then merges
    # it. Thus, it'll have to pass through the entire corpus in each iteration to 
    # find the next merge. It then has to iterate once again through the entire corpus
    # to replace all instances of the particular pair with the new token id minted.
    # Thus if the sizes of the corpus is m, and the number of merges
    # required is n, then it'll take n*m operations to do this job. 

    # Our method is different, it goes through the entire corpus initially, maintains 
    # the frequency of each pair, and the words that contain them.

    # Now in each iteration, it finds the most common pair, handles the ties, and then
    # finds the words in which the particular pair occurs, using a mapping of pair
    # to words. Now, for each of the words in the corresponding pair, it goes through the 
    # pairs in the old word token representation, subtracts its counts, creates the new 
    # token by merging the best token, and then adds the corresponding counts for the 
    # new pairs. 

    # Initialize the vocabulary as mentioned, 
    self.vocabulary = {
      **{
        0: b'<pad>',
        1: b'<unk>',
        2: b'<s>',
        3: b'</s>',
      },
      **{i+4: bytes([i]) for i in range(256)}
    }

    # Now, we wish to not remove anything, but also we wish to do pre tokenisation,
    # by splitting the text into words. We use the following regex pattern,

    word_pattern = re.compile(r'\S+|\s+', flags=re.UNICODE)
    words = word_pattern.findall(text)
    
    # Maintain the frequency of each word in the corpus.
    str_freqs = collections.Counter(words)
    byte_freqs = {}
    
    # The present token representation of each word
    word_to_tokens = {}

    # Frequency of each pair
    stats = collections.Counter()

    # Mapping of each pair to the set of words that contain it.
    pair_to_words = collections.defaultdict(set) 

    for word, freq in str_freqs.items():
      if word.isspace():
        w_bytes = word.encode('utf-8')
      else:
        w_bytes = (word + '</w>').encode('utf-8')
      
      byte_freqs[w_bytes] = freq

      tokens = [b+4 for b in w_bytes]
      word_to_tokens[w_bytes] = tokens

      for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i+1])
        stats[pair] += freq
        pair_to_words[pair].add(w_bytes)
    
    pq = [(-freq,pair) for pair,freq in stats.items()]
    heapq.heapify(pq)

    num_merges = size - len(self.vocabulary)
    next_id = len(self.vocabulary)

    # Start the training loop
    for _ in range(num_merges):
      best_pair = None
      while pq:
        neg, pair = heapq.heappop(pq)
        if stats.get(pair, 0) == -neg:
          best_pair = pair
          break
      
      if best_pair is None:
        print("No more pairs to merge.")
        break

      # We do not need to break the ties since the heap does it itself.

      # Get the words which use the best_pair 
      in_words = list(pair_to_words.pop(best_pair, []))

      for w_bytes in in_words:
        freq = byte_freqs[w_bytes]
        old = word_to_tokens[w_bytes]

        # Subtract the count
        for j in range(len(old) - 1):
          pair = (old[j], old[j+1])
          stats[pair] -= freq
          pair_to_words[pair].discard(w_bytes)
        
        # Get the new token
        new = []
        j = 0
        while j < len(old):
          if j < len(old) - 1 and old[j] == best_pair[0] and old[j+1] == best_pair[1]:
            new.append(next_id)
            j+= 2
          else:
            new.append(old[j])
            j+=1
        
        # Add the new counts
        for j in range(len(new) - 1):
          pair = (new[j], new[j+1])
          stats[pair] += freq
          pair_to_words[pair].add(w_bytes)

          heapq.heappush(pq, (-stats[pair], pair))

        word_to_tokens[w_bytes] = new

      self.merges[best_pair] = next_id 
      self.vocabulary[next_id] = self.vocabulary[best_pair[0]] + self.vocabulary[best_pair[1]]
      next_id += 1

      # Periodically refresh the stats, removing those pairs whose count has reached 0.
      if(_ + 1) % 500 == 0:
        stats = collections.Counter({k: v for k, v in stats.items() if v > 0})
        print(f"Merge {_+1}/{num_merges} complete.")
    
    print("Tokenizer Trained")

    self.inverse = {v:k for k,v in self.vocabulary.items()}
    self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

  def _encode_word(self, word):
   
    # To tokenize a given word, you are given a list of bytes for that word.
    # Now, you wish to keep on merging till we have no further merges to do.
    # Create each of the pairs, then find the best pair, with the lowest rank in the
    # merges dictionary, and merge the pair. 

    # Keep doing this till the next best pair that can be found is not found in the 
    # merges dictionary. 

    w_bytes = (word + '</w>').encode('utf-8')
    tokens = [b + 4 for b in w_bytes]

    while len(tokens) > 1:
      pairs = {(tokens[i], tokens[i+1]): i for i in range(len(tokens)-1)}
      best_pair = min(pairs, key = lambda p: self.merge_ranks.get(p, float('inf')))

      if best_pair not in self.merge_ranks:
        break

      idx = pairs[best_pair]
      new_id = self.merges[best_pair]
      tokens = tokens[:idx] + [new_id] + tokens[idx+2:]
    
    return tokens
  
  def encode(self, text):
    if not self.merges:
      raise RuntimeError("You must train the tokenizer before encoding.")
    word_pattern = re.compile(r'\S+|\s+', flags=re.UNICODE)
    words = word_pattern.findall(text)
    text_tokens = []
    for word in words:
      if word.isspace():
        word_bytes = word.encode('utf-8')
        token_ids = [b + 4 for b in word_bytes]
        text_tokens.extend(token_ids)
      else:
        text_tokens.extend(self._encode_word(word))
    return text_tokens
  
  def decode(self, tokens):
    if not self.merges:
        raise RuntimeError("You must train the tokenizer before decoding.")
    
    bytes_list = [self.vocabulary.get(t, b'') for t in tokens]
    text_bytes = b''.join(bytes_list)
    
    words = text_bytes.split(b'</w>')
    decoded_words = []
    for w in words:
      if not w:
        continue
      decoded_words.append(w.decode('utf-8', 'replace'))
    
    return " ".join(decoded_words)
  
  def save(self, filename):
    with open(filename,'w', encoding='utf-8') as f:
      sorted_vocab = sorted(self.vocabulary.items())
      for i, j in sorted_vocab:
        f.write(f'{j.decode("utf-8", "replace")}\n')

    print(f"Vocabulary saved to {filename}")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--train", required=True)
  parser.add_argument("--input", required=True)
  parser.add_argument("--vocab_size", type=int, default=5000)
  args = parser.parse_args()

  vocab_size = args.vocab_size
  vocab_file = f'240981_assignment2_bpe_vocab_{vocab_size}.txt'
  tokens_file = '240981_assignment2_bpe_tokens.txt'
  detokenized_file = '240981_assignment2_bpe_detokenized.txt'

  with open(args.train, 'r', encoding='utf-8') as f:
    corpus = f.read()
  
  tokenizer = BPE()
  tokenizer.train(corpus, vocab_size)

  tokenizer.save(vocab_file)

  with open(args.input, 'r', encoding='utf-8') as f:
    sample_text = f.read()

  token_ids = tokenizer.encode(sample_text)

  with open(tokens_file, 'w', encoding='utf-8') as f:
    for i in token_ids:
      f.write(f'{tokenizer.vocabulary[i].decode("utf-8", "replace")}\n')
  

  with open(detokenized_file, 'w', encoding='utf-8') as f:
    detokenized_text = tokenizer.decode(token_ids)
    f.write(detokenized_text)