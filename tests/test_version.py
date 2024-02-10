import unittest

import chequeparser


class VersionTestCase(unittest.TestCase):
    """Version tests"""

    def test_version(self):
        """check chequeparser exposes a version attribute"""
        self.assertTrue(hasattr(chequeparser, "__version__"))
        self.assertIsInstance(chequeparser.__version__, str)


if __name__ == "__main__":
    unittest.main()
