import nltk
import pickle
import argparse
from collections import Counter
import json
import numpy as np

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json_file, threshold, wordvec_file):
    """Build a simple vocabulary wrapper."""
    
    counter = Counter()
    with open(json_file, 'r') as data_file:    
        data = json.load(data_file)	 

    for i, key in enumerate(data['positive'].keys()):
        caption = data['positive'][key]
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

    for i, key in enumerate(data['negative'].keys()):
        caption = data['negative'][key]
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized the captions." %(i, len(data)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    #vocab.add_word('<pad>')
    #vocab.add_word('<start>')
    #vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)

    embedding_vec = None
    # load data from the pre-trained word embedding  	
    lines = open(wordvec_file, "r").readlines()	
    total = len(lines)
    hit = 0
    count = len(vocab)
 
    for line in lines:
        line = line.strip().split()
	word = line[0]
	vec = np.asarray([float(i) for i in line[1:]])
	vec_dim = len(vec)
	if embedding_vec is None:
	   embedding_vec = np.zeros((len(vocab), vec_dim), dtype=np.float32) 
	if word in vocab.word2idx.keys():
	   embedding_vec[vocab.word2idx[word], :] = vec 
	   hit += 1
	
	# once we get enough data, then we break
	if hit/float(count) >= 0.98:
	   break

    return vocab, embedding_vec

def main(args):

    vocab, embedding_vec = build_vocab(json_file=args.description_path,
                        threshold=args.threshold, wordvec_file = args.glove_path)

    vocab_path = args.vocab_path
    embedding_path = args.embedding_path

    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

    with open(embedding_path, 'wb') as f:
        pickle.dump(embedding_vec, f, pickle.HIGHEST_PROTOCOL)

    print("Total vocabulary size: %d" %len(vocab))
    print("Saved the vocabulary wrapper to '%s'" %vocab_path)
    print("Saved the mebedding file to '%s'" %embedding_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--description_path', type=str, 
                        default='../datasets/all_data3_pos_neg.json', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='../datasets/vocab.pkl', 
                        help='path for saving vocabulary wrapper')

    parser.add_argument('--glove_path', type=str, default='../datasets/glove/glove.6B.100d.txt', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--embedding_path', type=str, default='../datasets/vocab_vec_100d.pkl', 
                        help='path for saving vocabulary wrapper')	
    parser.add_argument('--threshold', type=int, default=1, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
