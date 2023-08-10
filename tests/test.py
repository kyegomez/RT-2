import torch
import unittest
from rt2.experimental.rt2_palme import PALME, RT2

class TestRT2(unittest.TestCase):
    def setUp(self):
        self.rt2 = RT2(
            palme=PALME(),
            num_actions=11,
            action_bins=256,
            depth=6,
            heads=8,
            dim_head=64,
            token_learner_ff_mult=2,
            token_learner_num_layers=2,
            token_learner_num_output_tokens=8,
            cond_drop_prob=0.2,
            use_attn_conditioner=False,
            conditioner_kwargs=dict()
        )
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