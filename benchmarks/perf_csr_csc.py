""" Test the performance of csr vs csc slicing """
import sys
import time
import numpy as np
from functools import wraps
import colorlog, logging

logger = colorlog.getLogger(__name__)
logger.setLevel(colorlog.DEBUG)
logger.addHandler(logging.StreamHandler())


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__.rjust(25)} Took {total_time:.4f} seconds. {args}")
        return result

    return timeit_wrapper


@timeit
def convert(matrix):
    if matrix.getformat() == 'csc':
        return matrix.tocsr()
    elif matrix.getformat() == 'csr':
        return matrix.tocsc()
    else:
        print("Neither csc or csr.")
        sys.exit(1)


@timeit
def time_column_access(matrix, columns):
    return matrix[:, ]


if __name__ == '__main__':
    import pandas as pd
    from juxtorpus.corpus import Corpus

    corpus = Corpus.from_dataframe(
        pd.read_csv("./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv"),
        col_doc='processed_text'
    )
    csr = corpus.dtm.matrix
    csc = convert(csr)
    csr = convert(csc)
    assert csr.shape == csc.shape, "They're the same matrix, they should have the same shape."

    columns = np.random.choice(csr.shape[1], size=25)
    time_column_access(csr, columns)
    time_column_access(csc, columns)
