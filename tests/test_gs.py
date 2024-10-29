import unittest
from zensvi.download import GSDownloader
from test_base import TestBase


class TestGS(TestBase):
    def setUp(self):
        super().setUp()
        self.local_dir = self.output_dir / "gs_download"
        self.ensure_dir(self.local_dir)
        self.gs_download = GSDownloader()

    def test_download_all_data(self):
        self.gs_download.download_all_data(self.local_dir)
        # assert True if there are files in the output directory
        self.assertTrue(len(list(self.local_dir.iterdir())) > 0)

    def test_download_manual_labels(self):
        self.gs_download.download_manual_labels(self.local_dir)
        # assert True if there are files in the output directory
        self.assertTrue(len(list(self.local_dir.iterdir())) > 0)

    def test_download_train(self):
        self.gs_download.download_train(self.local_dir)
        # assert True if there are files in the output directory
        self.assertTrue(len(list(self.local_dir.iterdir())) > 0)

    def test_download_test(self):
        self.gs_download.download_test(self.local_dir)
        # assert True if there are files in the output directory
        self.assertTrue(len(list(self.local_dir.iterdir())) > 0)

    def test_download_img_tar(self):
        self.gs_download.download_img_tar(self.local_dir)
        # assert True if there are files in the output directory
        self.assertTrue(len(list(self.local_dir.iterdir())) > 0)


if __name__ == "__main__":
    unittest.main()
