from typing import Union
import pathlib, os, shutil, tempfile
import hashlib
import filecmp


class DeduplicatedDirectory(object):
    def __init__(self, dir_path: Union[str, pathlib.Path] = None):
        if dir_path is None: dir_path = tempfile.mkdtemp()
        if isinstance(dir_path, str): dir_path = pathlib.Path(dir_path)
        self._dir_path = dir_path
        self._index = dict()
        self._hash_alg = hashlib.md5

    @property
    def path(self):
        return self._dir_path

    def files(self):
        return list(pathlib.Path(self._dir_path).glob('**/*'))

    def list(self):
        return [f.name for f in self.files()]

    def add(self, file: pathlib.Path):
        if not file.is_file():
            raise ValueError(f"{file.name} is not a file.")
        if self.exists(file):
            raise ValueError(f"{file.name} already exists.")
        shutil.copy(file, self._dir_path.joinpath(file.name))
        self._build_index()

    def add_content(self, content: bytes, fname: str):
        # check if content already exists.
        if self._filename_exists(fname) and self.content_exists(content):
            raise ValueError(f"File name: {fname} and its content are duplicated.")
        with open(self._dir_path.joinpath(fname), 'wb') as fh: fh.write(content)
        self._build_index()

    def remove(self, fname: str):
        for existing in self.files():
            if existing.name == fname:
                os.remove(existing)
                for digest, path in self._index.items():
                    if path == existing:
                        del self._index[digest]
                        return
        raise ValueError(f"{fname} does not exist.")

    def upsert(self, file: pathlib.Path):
        raise NotImplementedError()  # first compares file on stats(), if found, update, else insert. Content of file not compared.

    def exists(self, file: pathlib.Path, shallow: bool = True):
        for existing in self.files():
            if filecmp.cmp(file, existing, shallow=shallow):
                return True
        return False

    def content_exists(self, content: bytes, shallow: bool = True):
        size = len(content)
        digest = ''
        if not shallow:
            digest = self._hash_alg(content).hexdigest()
        for existing in self.files():
            if shallow:
                if existing.stat().st_size == size:
                    return True
            else:
                if self._index.get(digest, None) is not None:
                    return True
        return False

    def _filename_exists(self, fname: str):
        return fname in self.list()

    def _build_index(self):  # todo: this should be async
        for existing in self.files():
            if existing in self._index.values():
                continue
            with open(existing, 'rb') as fh:
                digest = self._hash_alg(fh.read()).hexdigest()
            self._index[digest] = existing

    def _add_to_index(self, digest: str, path: pathlib.Path):
        self._index[digest] = path
