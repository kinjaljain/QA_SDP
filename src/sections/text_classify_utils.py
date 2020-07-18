import itertools
from collections import Counter

import numpy as np
import torch


def build_vocab(train_set, f):
    all_words = itertools.chain.from_iterable(
        f(e)
        for e
        in train_set
    )

    word_freq = Counter(all_words)
    valid_words = sorted(
        filter(lambda x: word_freq[x] > 0, word_freq),
        key=lambda x: word_freq[x],
        reverse=True
    )

    word_to_id = {
        '<pad>': 0,
        '<unk>': 1,
    }

    word_to_id.update({
        word: idx
        for idx, word
        in enumerate(valid_words, start=2)
    })

    return word_to_id


def to_input_tensor(
    token_sequence,
    token_to_id_map,
    pad_index: int = 0,
    unk_index: int = 1
) -> torch.Tensor:


    max_sequence_len = max(len(seq) for seq in token_sequence)
    batch_size = len(token_sequence)
    sequence_array = np.zeros((batch_size, max_sequence_len), dtype=np.int64)
    sequence_array.fill(pad_index)
    for e_id in range(batch_size):
        sequence_i = token_sequence[e_id]

        id_sequence = [
            token_to_id_map.get(token, unk_index)
            for token
            in sequence_i
        ]

        sequence_array[e_id, :len(id_sequence)] = id_sequence

    sequence_array = torch.from_numpy(sequence_array)

    return sequence_array

def batch_iter(data, batch_size, shuffle = False):

    batch_num = int(np.ceil(len(data) / batch_size))
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        batch_examples = [data[idx] for idx in indices]

        yield batch_examples
