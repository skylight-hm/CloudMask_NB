import unittest


class AddTest(unittest.TestCase):
    def setUp(self):
        self.a = 1
        self.b = 2

    def test_add(self):
        c = self.a + self.b
        self.assertEqual(c, 3)
