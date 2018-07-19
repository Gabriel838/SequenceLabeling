from itertools import zip_longest


class PadSequence(object):
    #TODO: check if given shape is smaller than max
    def __init__(self, seq, shape=None, fillvalue=0):
        if shape is None:
            shape = self.get_shape(seq)

        self.embedding = self.init_embedding(shape, fillvalue)
        self.lengths = self.init_embedding(shape[:-1])

        # update embedding and lengths from seq
        self.embed(seq, self.embedding, self.lengths)

    def get_shape(self, seq):
        """Find padding shape of sequence."""
        sublen = (self.get_shape(subseq) for subseq in seq if isinstance(subseq, list))
        return (len(seq), *map(max, zip_longest(*sublen)))

    def init_embedding(self, shape, fillvalue=0):
        """Initialize embedding to a specified shape."""
        d, *sub_dims = shape
        return [self.init_embedding(sub_dims) if sub_dims else fillvalue for _ in range(d)]

    def embed(self, seq, embedding, lengths):
        """Copy seq to embedding and output sub-sequence lengths."""
        for i in range(len(embedding)):
            if isinstance(embedding[i][0], list):
                self.embed(seq[i], embedding[i], lengths[i])
            else:
                try:
                    embedding[i][:len(seq[i])] = seq[i]
                    lengths[i] = len(seq[i])
                except IndexError:
                    lengths[i] = 0


if __name__ == "__main__":
    # seq = [[1,4,5,7],
    #        [2,8],
    #        [6,3,2]]

    seq = [[[1, 2, 3], [4, 5, 6, 7]],
           [[1], ],
           [[7, 8, 9], [10, 11]]]

    pad_obj = PadSequence(seq, [3, 3, 5])
    print("lengths:", pad_obj.lengths)
    print("embedding:", pad_obj.embedding)