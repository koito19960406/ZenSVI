import unittest
import os
from zensvi.download import GSDownloader


class TestGS(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.local_dir = 'test_download/'
        pass

    def test_download_manual_labels(self):
        gs_download = GSDownloader()
        gs_download.download_manual_labels(self.local_dir)
        # assert True if there are files in the output directory
        self.assertTrue(len(os.listdir(self.local_dir)) > 0)

    def test_download_train(self):
        gs_download = GSDownloader()
        gs_download.download_train(self.local_dir)
        # assert True if there are files in the output directory
        self.assertTrue(len(os.listdir(self.local_dir)) > 0)

    def test_download_test(self):
        gs_download = GSDownloader()
        gs_download.download_test(self.local_dir)
        # assert True if there are files in the output directory
        self.assertTrue(len(os.listdir(self.local_dir)) > 0)

    def test_download_img_tar(self):
        gs_download = GSDownloader()
        gs_download.download_img_tar(self.local_dir)
        # assert True if there are files in the output directory
        self.assertTrue(len(os.listdir(self.local_dir)) > 0)


if __name__ == "__main__":
    unittest.main()
