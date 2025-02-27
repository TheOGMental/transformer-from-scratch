from T1000 import *

import torch
import unittest

test_config = GPTConfig()
class TestT1000(unittest.TestCase):
    n_context = 10
    input_shape = (n_context,)
    embeded_shape = (n_context, test_config.d_model)
    transformer_shape = (n_context, test_config.d_vocab)
    def setUp(self):
        self.x_vector = torch.randint(0, test_config.d_vocab, self.input_shape)
        self.x_embeded = torch.rand(self.embeded_shape)
        return super().setUp()
    
    def test_attentionhead(self):
        '''Test that the attention head works and does not produce NaN or Inf'''
        attention_head = AttentionHead(test_config)
        output = attention_head(self.x_embeded)
        self.assertEqual(output.shape, self.embeded_shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_multiheadattention(self):
        '''Test that the multihead attention works and does not produce NaN or Inf'''
        multihead_attention = MultiHeadedAttention(test_config)
        output = multihead_attention(self.x_embeded)
        self.assertEqual(output.shape, self.embeded_shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_MLP(self):
        '''Test that the MLP works and does not produce NaN or Inf'''
        mlp = MLP(test_config)
        output = mlp(self.x_embeded)
        self.assertEqual(output.shape, self.embeded_shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_transformer(self):
        '''Test that the transformer works and does not produce NaN or Inf'''
        transformer = Transformer(test_config)
        output = transformer(self.x_vector)
        self.assertEqual(output.shape, self.transformer_shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

