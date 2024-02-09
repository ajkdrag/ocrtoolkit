import unittest

import cheqeueparser


class VersionTestCase(unittest.TestCase):
    """ Version tests """

    def test_version(self):
        """ check cheqeueparser exposes a version attribute """
        self.assertTrue(hasattr(cheqeueparser, "__version__"))
        self.assertIsInstance(cheqeueparser.__version__, str)


if __name__ == "__main__":
    unittest.main()
