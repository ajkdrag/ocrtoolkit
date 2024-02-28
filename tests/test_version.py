import unittest

import ocrtoolkit


class VersionTestCase(unittest.TestCase):
    """Version tests"""

    def test_version(self):
        """check ocrtoolkit exposes a version attribute"""
        self.assertTrue(hasattr(ocrtoolkit, "__version__"))
        self.assertIsInstance(ocrtoolkit.__version__, str)


if __name__ == "__main__":
    unittest.main()
