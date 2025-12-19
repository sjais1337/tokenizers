import argparse
from collections import defaultdict, Counter
import math
import numpy as np
import re

def _log_sum_exp(a,b):
  if a == -np.inf: return b
  if b == -np.inf: return a

  if a > b:
    return a + np.log(1+np.exp(b-a))
  else:
    return b + np.log(1+np.exp(a-b))

# Based on the assumption that each subword unit exists independently. Thus the segmentation
# of a particular word has the probability P(x) = PI_i^M p(x_i) thus, we choose the segmentation
# which maximises this, i.e. x^* = argmax P(x)  

# In a normal Unigram tokenizer, we start from a seed vocabulary. Which can be generated in 
# multiple different ways. We use the Enhance Suffix Algorithm to do the same here.
# 
# Next, we repeat the following steps till we reach the desired vocabulary size,
#   1. Compute the ll_i for each subword x_i, by calculating the drop in
#      likelihood by removing x_i.
#   2. Sort the tokens by ll_i and remove the bottom 20%.
#   3. Renormalize the probabilities so that the sum to 1.


class Unigram:
  def __init__(self, vocab_size=5000):
    self.target = vocab_size
    self.vocab = {}
    self.special_tokens = ['<unk>', '<pad>', '<s>', '</s>']
    self.unk_token = '<unk>'
    self.cache = {}
    self.id_to_token = []
    self.token_to_id = {}

  def train(self, corpus, seed_size=20000):
    self._init_seed(corpus, seed_size)

    counts = Counter(corpus)
    iter = 0 

    while len(self.vocab) > self.target:
      iter+=1

      print(f'Training Iteration {iter}\n Current Vocab Size = {len(self.vocab)}')

      self.cache.clear()
      likelihood = self._em(counts)

      if len(self.vocab) > self.target:
        self._prune()

  def _renormalize(self):
    tot = - np.inf

    for log_prob in self.vocab.values():
      tot = _log_sum_exp(tot, log_prob)
    if tot > -np.inf:
      for token in self.vocab:
        self.vocab[token] -= tot

  def _prune(self):
    if len(self.vocab) <= self.target:
      return

    # Only consider the tokens that are eligible for pruning
    prunable = [
      (prob, token) for token, prob in self.vocab.items()
      if len(token) > 1 and token not in self.special_tokens
    ]

    prunable.sort()

    num_to_prune = max(1, int(len(self.vocab)*0.2))
    num_to_prune = min(num_to_prune, len(self.vocab) - self.target)

    tokens = {token for _k, token in prunable[:num_to_prune]}

    print(f"   Pruning {len(tokens)} tokens.")

    self.vocab = {token: prob for token, prob in self.vocab.items() if token not in tokens}

    self._renormalize()

  def _em(self, counts):
    expected_counts = defaultdict(float)
    tot_ll = 0.0

    for word, count in counts.items():
      fow_scores = self._fow(word)
      w_ll = fow_scores[-1] # P(w) = score(n)


      if w_ll == -np.inf: continue

      back_scores = self._back(word)
      tot_ll += w_ll * count

      # P(token w[j:i] | word) = P(w[:i])*p(w[j:i])*P(w[j:])/P(w)

      for i in range(len(word)):
        for j in range(i+1, len(word) + 1):
          subword = word[i:j]
          if subword in self.vocab:
            log_prob = fow_scores[i] + self.vocab[subword] + back_scores[j] - w_ll

            # Multiple the probability of the token given the word with the number of times the 
            # word has occurred in the corups, to get the expected counts. 
            expected_counts[subword] += np.exp(log_prob)*count
    
    total_expected_counts = sum(expected_counts.values())
    if total_expected_counts == 0: return tot_ll

    for token in self.vocab:
      log_prob = math.log(expected_counts.get(token, 0) + 1) - math.log(total_expected_counts + len(self.vocab))
      self.vocab[token] = log_prob
    
    return tot_ll

  def _fow(self, word):
    # Gives the prefix likelihoods 
    key = word
    
    if key in self.cache:
      return self.cache[key]
    
    # The scores are the log probability of reaching that particular prefix.
    # P(w) = SUM_{segmentation S} PI_{t \in S} p(t) by the unigram assumption
    # here p(t) is the probability of the token t, simply by the frequency
    # 
    # The score is P(w[:i]) now to end up at i, the last token must have
    # been for some j, w[j:i] then by definition, P(w[:i]) = P(w[:j])*p(w[j:i])
    # thus we get a recurrence, score(i) = SUM_{j=0}^{i-1} score(j) p(w[j:i])
    # and that's it. 

    scores = np.full(len(word) + 1, -np.inf)
    scores[0] = 0.0

    for i in range(1, len(word) + 1):
      for j in range(i):
        subword = word[j:i]
        if subword in self.vocab:
          # in the first run, the self.vocab[subword] all are equal to the same value
          scores[i] = _log_sum_exp(scores[i], scores[j] + self.vocab[subword])
    
    self.cache[key] = scores
    return scores
  
  def _back(self, word):
    # The same thing as in the previous thing just a bit different, and does
    # the same for the suffixes and not the prefixes.
    scores = np.full(len(word) + 1, -np.inf)
    scores[len(word)] = 0.0

    for i in range(len(word) - 1, -1, -1):
      for j in range(i+1, len(word)+1):
        subword = word[i:j]
        if subword in self.vocab:
          scores[i] = _log_sum_exp(scores[i], self.vocab[subword] + scores[j] )

    return scores
    
  def _init_seed(self, corpus, seed_size):
    counts = defaultdict(int)

    seed_vocab = []
    seed_vocab.extend(self.special_tokens)
    
    chars = set()
    for chunk in corpus:
      chars.update(chunk)
    
    seed_vocab.extend(sorted(chars))

    for word in corpus:
      for i in range(len(word)):
        for j in range(i+1, min(i+11, len(word) + 1)):
          counts[word[i:j]] += 1

    frequent = sorted(
      [(sub,count) for sub, count in counts.items() if count >= 2 and len(sub) > 1],
      key = lambda x: x[1], reverse=True
    )

    rem = seed_size - len(seed_vocab)

    if rem > 0:
      for str, _ in frequent[:rem]:
        seed_vocab.append(str)

    self.id_to_token = seed_vocab
    self.token_to_id = {token: i for i, token in enumerate(self.id_to_token)}

    log_prob = -math.log(len(seed_vocab))
    self.vocab = {token: log_prob for token in seed_vocab}
    for token in self.special_tokens: self.vocab[token] = 0.0

    print(f"Initialized seed vocabulary.")

  def _viterbi_segment(self, word):
    # Segments the word into the subwords using the viterbi algorithm, and chooses 
    # the single most probable segmentation path under the unigram assumption

    # Best probability of segmenting word[:i]
    best_scores = np.full(len(word)+1,-np.inf)
    
    # backpointers[i] has the index j where the split before i happened
    backpointers = np.zeros(len(word)+1, dtype=int)
 
    best_scores[0] = 0.0    

    for i in range(len(word)+1):
      for j in range(i):
        subword = word[j:i]

        if subword in self.vocab:
          log_prob = self.vocab[subword]

          score = best_scores[j] + log_prob
          
          if score > best_scores[i]:
            best_scores[i] = score
            backpointers[i] = j
    
    if best_scores[-1] == -np.inf:
      return [self.unk_token]
    
    tokens = []
    i = len(word)

    while i > 0:
      j = backpointers[i]
      tokens.append(word[j:i])
      i = j
    
    return list(reversed(tokens))

  def encode(self, text):
    if not self.vocab:
      raise RuntimeError("Tokenizer has not been trained.")

    strings = self._segment(text)
    ids = [self.token_to_id.get(t, self.token_to_id[self.unk_token]) for t in strings]
    for i, k in enumerate(ids):
      if k == self.token_to_id[self.unk_token]:
        print(strings[i])
    return ids

  def _segment(self, text):
    chunks = re.split(r'(\s)', text)
    all_tokens = []
    for chunk in chunks:
      if not chunk: continue
      if chunk.isspace():
        all_tokens.append(chunk)
      else:
        all_tokens.extend(self._viterbi_segment(chunk))
    
    return all_tokens

  def decode(self, ids):
    tokens = [self.id_to_token[id] for id in ids]
    return "".join(tokens)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--train", required=True)
  parser.add_argument("--input", required=True)
  parser.add_argument("--vocab_size", type=int, default=5000)
  args = parser.parse_args()

  vocab_size = args.vocab_size
  vocab_file = f'240981_assignment2_unigram_vocab_{vocab_size}.txt'
  tokens_file = '240981_assignment2_unigram_tokens.txt'
  detokenized_file = '240981_assignment2_unigram_detokenized.txt'

  with open(args.train, 'r', encoding='utf-8') as f:
    corpus = f.read()
  
  corpus_chunks = [chunk for chunk in re.split(r'(\s)', corpus)]
  tokenizer = Unigram(vocab_size=args.vocab_size)
  tokenizer.train(corpus_chunks)

  sorted_vocab = sorted(tokenizer.vocab.keys(), key= lambda x: tokenizer.vocab[x], reverse=True)

  with open(vocab_file, "w", encoding="utf-8") as f:
    for token in sorted_vocab:
      f.write(f'{token}\n')
  
  print("Vocabulary saved.")

  with open(args.input, 'r', encoding='utf-8') as f:
    input_text = f.read()

  ids = tokenizer.encode(input_text)

  with open(tokens_file, 'w', encoding='utf-8') as f:
    for id in ids:
      f.write(f'{tokenizer.id_to_token[id]}\n')
  print('Tokens saved.')

  detokenized = tokenizer.decode(ids)
  with open(detokenized_file, 'w', encoding='utf-8') as f:
    f.write(detokenized)
  print('Detokenized text saved.')