from T1000 import *
from utility import create_tokenizer

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

    def test_generate(self):
        '''Test that model generates a total of test_max_tokens when given prompt'''
        model_file = "model.pt"
        model = torch.load(model_file, weights_only=False)
        model.eval()
        test_input = "How much wood could a woodchuck chuck if a woodchuck could chuck wood?"
        test_max_tokens = 50

        # May sometimes randomly break in the first run or two?
        self.assertTrue(len(model.generate(test_input, test_max_tokens).split()) == 50)