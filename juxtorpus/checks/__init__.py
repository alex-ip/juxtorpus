from abc import ABCMeta, abstractmethod
import pathlib
from typing import Iterable, Callable, Generator
import pandas as pd
import os


class FlaggedPath(object):
    def __init__(self, path: pathlib.Path, reason: str):
        self.path = path
        self.reason = reason

    def __repr__(self):
        return f"Path: {self.path}\t [REASON: {self.reason}]"


class Check(object):
    def __call__(self, path):
        raise NotImplementedError()


class FileSizeCheck(Check):
    REASON: str = "Exceeded the maximum size of {} bytes."

    def __init__(self, max_bytes: int):
        self.max_size = max_bytes

    def __call__(self, path: pathlib.Path) -> bool:
        # print(f"File: {path} Size: {path.stat().st_size} bytes")
        return path.stat().st_size < self.max_size

    def reason(self):
        return self.REASON.format(self.max_size)


def check_file_lang(file: pathlib.Path):
    # print(f"file_lang: checking file - {file}")
    return True


class FileCheckers(object):
    def __init__(self, checks: list[Callable]):
        self._checks = checks
        self._flagged = dict()
        self._passed = list()

    def run(self, paths: Iterable[pathlib.Path]):
        if isinstance(paths, Generator):
            paths = list(paths)
        paths = self._to_paths(paths)
        flagged = dict()
        passed = list()
        path: pathlib.Path
        for path in paths:
            for check in self.checks:
                passed = check(path)
                if not passed:
                    if isinstance(check, Check):
                        reason = check.reason()
                    else:
                        reason = str(check)

                    key: str = str(path)
                    reasons = flagged.get(key, list())
                    reasons.append(reason)
                    flagged[key] = reasons
        self._passed = [str(p) for p in paths if str(p) not in flagged.keys()]
        self._flagged = flagged
        return flagged

    @property
    def checks(self):
        return self._checks

    def flagged(self):
        return self._flagged

    def passed(self):
        return self._passed

    def summary(self):
        num_flagged = len(self._flagged.keys())
        num_passed = len(self._passed)
        return pd.Series([num_flagged, num_passed, num_flagged + num_passed],
                         index=['Flagged', 'Passed', 'Total'],
                         name='File Check Summary')

    def _to_paths(self, paths):
        for i, p in enumerate(paths):
            if not isinstance(p, pathlib.Path):
                paths[i] = pathlib.Path(p)
        return paths


if __name__ == '__main__':
    checks = [
        check_file_lang,
        FileSizeCheck(max_bytes=1_000_000)  # 1Mb
    ]
    file_checks = FileCheckers(checks)

    HOME = os.getenv("HOME")
    paths = pathlib.Path(f"{HOME}/Downloads/Data/Top100_Text_1/").rglob("*.txt")

    flagged = file_checks.run(paths)

    print("+++ Results +++")
    print("FLAGGED ", file_checks.flagged())
    print("PASSED ", file_checks.passed())
    print(file_checks.summary())
