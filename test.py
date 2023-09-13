import torch
import unittest
from rt2.model import RT2

class TestRT2(unittest.TestCase):
    def setUp(self):
        self.rt2 = RT2()
        self.video = torch.rand((1, 3, 10, 224, 224))
        self.texts = ["This is a test"]

    def test_forward(self):
        output = self.rt2(self.video, self.texts)
        self.assertEqual(output.shape, (1, 10, 11, 256))

    def test_forward_no_texts(self):
        output = self.rt2(self.video)
        self.assertEqual(output.shape, (1, 10, 11, 256))

    def test_forward_different_video_shape(self):
        video = torch.rand((2, 3, 5, 224, 224))
        output = self.rt2(video, self.texts)
        self.assertEqual(output.shape, (2, 5, 11, 256))

    def test_forward_different_num_actions(self):
        self.rt2.num_actions = 5
        output = self.rt2(self.video, self.texts)
        self.assertEqual(output.shape, (1, 10, 5, 256))

    def test_forward_different_action_bins(self):
        self.rt2.action_bins = 128
        output = self.rt2(self.video, self.texts)
        self.assertEqual(output.shape, (1, 10, 11, 128))

if __name__ == '__main__':
    unittest.main()