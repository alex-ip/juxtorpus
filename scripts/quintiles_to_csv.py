""" Ad-hoc script to convert quintiles to csvs.
"""

import os
from pathlib import Path
from collections import namedtuple
from itertools import chain
import re

import pandas as pd

File = namedtuple("File", ['quintile', 'date', 'path'])
Row = namedtuple("Row", ['text', 'quintile', 'date'])
row_pattern = re.compile(r'<TWEET>(.+)</TWEET>')


def create_file_obj(path: Path):
    chunks = path.name.rstrip(path.suffix).split('_')
    quintile = chunks[1]
    date = '/'.join(chunks[4:7])
    return File(quintile=quintile, date=date, path=path)


def read_rows(file: File):
    with open(file.path, 'r') as h:
        matched = row_pattern.finditer(' '.join(h.readlines()))
    return (Row(text=match.group(0), quintile=file.quintile, date=file.date) for match in matched)


if __name__ == '__main__':
    quintile_dirs = [Path("/Users/hcha9747/Downloads/Quintile_0"),
                     Path("/Users/hcha9747/Downloads/Quintile_4")]

    for quintile_dir in quintile_dirs:
        assert quintile_dir.exists() and quintile_dir.is_dir(), f"Must be an existing directory. {quintile_dir=}"
        print(f"Parsing files in {quintile_dir} to dataframe...")
        files = [create_file_obj(f) for f in quintile_dir.glob("*.txt")]
        print(f"Number of files: {len(files)}")

        rows = chain(*(read_rows(f) for f in files))
        df = pd.DataFrame(rows)
        save_path = quintile_dir.joinpath(Path(f"{quintile_dir.name}.csv"))
        df.to_csv(save_path)
        print(f"Written to {save_path}. Size: {save_path.stat().st_size/1e6}Mb.")
