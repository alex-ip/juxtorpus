import zipfile
from unittest import TestCase
import pathlib
from juxtorpus.utils import DeduplicatedDirectory

"""
TODO: 22.09.22 - For deduplication, it seems the file directory should be associated with the widget object?

With the widget, what we get is the metadata. We can compare that against the directory first. If it fails, 
we can then compare the data using hash.

Keep a cache of message digest in the object itself when a data comparison is required.


note to self: 
there should be a deduplicated 'directory' object. --- upsert() method?
this object can pretty much index the directory with message digest (of its content + stats e.g. file name) async.

also, make upload directory using tempdir. Then wrap that around with a deduplicatedDir object. This is the staging
area. Separate uploaded? with from staged? maybe not.
"""


class TestDeduplicatedDirectory(TestCase):
    def setUp(self) -> None:
        print()
        self.dd = DeduplicatedDirectory()
        print(f"[TEST] DeduplicatedDirectory created at {self.dd.path}.")
        self.dir_ = pathlib.Path('./assets')
        self.file = pathlib.Path('./assets/content.txt')
        self.zip_ = pathlib.Path('./assets/nested.zip')
        self.content = b'some content'
        self.fname = 'filename'

    def tearDown(self) -> None:
        del self.dd

    def test_add(self):
        print()
        file = pathlib.Path('./assets/content.txt')
        self.dd.add(file)
        assert file.name in self.dd.list(), "File not added to directory."

    def test_exists(self):
        print()
        self.dd.add(self.file)
        assert self.dd.exists(self.file), "File should already exist."

    def test_not_exists(self):
        print()
        assert not self.dd.exists(self.file), "File should already exist."

    def test_add_duplicate(self):
        print()
        self.dd.add(self.file)
        with self.assertRaises(ValueError):
            self.dd.add(self.file)

    def test_add_content(self):
        print()
        self.dd.add_content(self.content, self.fname)
        assert self.fname in self.dd.list(), "File not added to directory."

    def test_content_exists(self):
        print()
        self.dd.add_content(self.content, self.fname)
        assert self.dd.content_exists(self.content), "Content should already exist."

    def test_content_exists_not_shallow(self):
        print()
        self.dd.add(self.file)
        with open(self.file, 'rb') as fh:
            content = fh.read()
            print(content)
        assert self.dd.content_exists(content, shallow=False), "Content should already exist."

    def test_content_not_exists(self):
        print()
        content = b'some content'
        assert not self.dd.content_exists(content), "Content should not already exist."

    def test_add_content_duplicate(self):
        print()
        content = b'some content'
        fname = 'filename'
        self.dd.add_content(content, fname)
        with self.assertRaises(ValueError):
            self.dd.add_content(content, fname)

    def test_remove(self):
        print()
        content = b'some content'
        fname = 'filename'
        self.dd.add_content(content, fname)
        self.dd.remove(fname)
        assert fname not in self.dd.list(), "File was not removed from directory."

    def test_add_directory(self):
        print()
        self.dd.add_directory(self.dir_)
        files = [f.name for f in self.dir_.glob("**/*") if f.is_file()]
        for f in files:
            assert f in self.dd.list(), f"{f} should exist in the directory."

    def test_add_zip(self):
        print()
        self.dd.add_zip(self.zip_)
        with zipfile.ZipFile(self.zip_) as z:
            for zipinfo in z.infolist():
                if zipinfo.filename.endswith('/'): continue
                assert pathlib.Path(zipinfo.filename).name in self.dd.list(), \
                    f"{zipinfo.filename} should exist in directory."
