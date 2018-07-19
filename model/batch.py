import random
import math

class Batch(object):
    def __init__(self, *args, batch_size):
        self.args = args
        self.batch_size = batch_size
        self.iter = 0
        self.num_batchs = math.ceil(len(args[0]) / batch_size)
        self.indices = list(range(len(args[0])))
        random.shuffle(self.indices)

    def __iter__(self):
        return self

    def __next__(self):
        batch = self.indices[self.iter * self.batch_size : (self.iter + 1) * self.batch_size]
        result = [[item[i] for i in batch] for item in self.args]

        if self.iter >= self.num_batchs:
            raise StopIteration
        self.iter += 1

        return result

    def __len__(self):
        return self.num_batchs


if __name__ == "__main__":
    inputs = [1,2,3,4,5]
    batch = Batch(inputs, batch_size=2)
    print(len(batch))