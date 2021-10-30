from pathlib import Path

def count_lines(filepath):
    n_lines = 0
    with Path(filepath).open() as f:
        for l in f:
            n_lines += 1
    return n_lines

def yield_lines(filepath, n_lines=float('inf'), prop=1):
    if prop < 1:
        assert n_lines == float('inf')
        n_lines = int(prop * count_lines(filepath))
    with open(filepath, 'r') as f:
        for i, l in enumerate(f):
            if i >= n_lines:
                break
            yield l.rstrip('\n')

def read_lines(filepath, n_lines=float('inf'), prop=1):
    return list(yield_lines(filepath, n_lines, prop))