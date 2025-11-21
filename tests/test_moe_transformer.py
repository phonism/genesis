"""Tests for MoE Transformer models.

This module tests the Mixture of Experts transformer implementation,
including configuration, model components, and full model forward passes.
"""

import unittest
import genesis
from genesis import Tensor
from genesis.models.transformers import (
    MoEConfig,
    MoEModel,
    MoEForCausalLM,
    MoEAttention,
    MoEDecoderLayer,
    DenseFFN,
    get_moe_config,
)


class TestMoEConfig(unittest.TestCase):
    """
    Test cases for MoEConfig class.
    """

    def test_default_config(self):
        """
        Test that default configuration is valid.
        """
        config = MoEConfig()
        self.assertEqual(config.vocab_size, 32000)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_local_experts, 8)
        self.assertEqual(config.num_experts_per_tok, 2)

    def test_custom_config(self):
        """
        Test custom configuration creation.
        """
        config = MoEConfig(
            vocab_size=1000,
            hidden_size=256,
            num_local_experts=4,
            num_experts_per_tok=2,
        )
        self.assertEqual(config.vocab_size, 1000)
        self.assertEqual(config.hidden_size, 256)
        self.assertEqual(config.num_local_experts, 4)

    def test_head_dim_computation(self):
        """
        Test automatic computation of head_dim.
        """
        config = MoEConfig(hidden_size=768, num_attention_heads=12)
        self.assertEqual(config.head_dim, 64)

    def test_config_validation(self):
        """
        Test configuration validation.
        """
        # Invalid: hidden_size not divisible by num_attention_heads
        with self.assertRaises(ValueError):
            MoEConfig(hidden_size=777, num_attention_heads=12)

        # Invalid: num_experts_per_tok > num_local_experts
        with self.assertRaises(ValueError):
            MoEConfig(num_experts_per_tok=10, num_local_experts=4)

    def test_config_to_dict(self):
        """
        Test configuration serialization to dictionary.
        """
        config = MoEConfig(vocab_size=1000, hidden_size=256)
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["vocab_size"], 1000)
        self.assertEqual(config_dict["hidden_size"], 256)

    def test_config_from_dict(self):
        """
        Test configuration creation from dictionary.
        """
        config_dict = {"vocab_size": 1000, "hidden_size": 256}
        config = MoEConfig.from_dict(config_dict)
        self.assertEqual(config.vocab_size, 1000)
        self.assertEqual(config.hidden_size, 256)

    def test_pretrained_configs(self):
        """
        Test loading predefined configurations.
        """
        # Test moe-small config
        config = get_moe_config("moe-small")
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.num_local_experts, 4)

        # Test invalid config name
        with self.assertRaises(ValueError):
            get_moe_config("nonexistent-config")


class TestMoEComponents(unittest.TestCase):
    """
    Test cases for MoE model components.
    """

    def setUp(self):
        """
        Set up test configuration.
        """
        self.config = MoEConfig(
            vocab_size=100,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_local_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=64,
        )

    def test_dense_ffn(self):
        """
        Test DenseFFN module.
        """
        ffn = DenseFFN(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            hidden_act="silu",
        )

        batch_size, seq_len = 2, 10
        x = genesis.randn(batch_size, seq_len, self.config.hidden_size)
        output = ffn(x)

        self.assertEqual(output.shape, x.shape)

    def test_moe_attention(self):
        """
        Test MoEAttention module.
        """
        attention = MoEAttention(self.config, layer_idx=0)

        batch_size, seq_len = 2, 10
        hidden_states = genesis.randn(batch_size, seq_len, self.config.hidden_size)

        output, past_kv = attention(hidden_states, use_cache=True)

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertIsNotNone(past_kv)
        self.assertEqual(len(past_kv), 2)  # (key, value)

    def test_moe_decoder_layer(self):
        """
        Test MoEDecoderLayer module.
        """
        layer = MoEDecoderLayer(self.config, layer_idx=0)

        batch_size, seq_len = 2, 10
        hidden_states = genesis.randn(batch_size, seq_len, self.config.hidden_size)

        output, past_kv = layer(hidden_states, use_cache=True)

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertIsNotNone(past_kv)

    def test_moe_vs_dense_layer(self):
        """
        Test that layers correctly alternate between MoE and dense when configured.
        """
        config = MoEConfig(
            hidden_size=128,
            num_hidden_layers=4,
            use_moe_in_all_layers=False,
            moe_layer_interval=2,
            first_moe_layer=1,
        )

        # Layer 0: Dense (before first_moe_layer)
        layer0 = MoEDecoderLayer(config, layer_idx=0)
        self.assertFalse(layer0.is_moe_layer)

        # Layer 1: MoE (first_moe_layer)
        layer1 = MoEDecoderLayer(config, layer_idx=1)
        self.assertTrue(layer1.is_moe_layer)

        # Layer 2: Dense
        layer2 = MoEDecoderLayer(config, layer_idx=2)
        self.assertFalse(layer2.is_moe_layer)

        # Layer 3: MoE (interval of 2)
        layer3 = MoEDecoderLayer(config, layer_idx=3)
        self.assertTrue(layer3.is_moe_layer)


class TestMoEModel(unittest.TestCase):
    """
    Test cases for full MoE models.
    """

    def setUp(self):
        """
        Set up test configuration and model.
        """
        self.config = MoEConfig(
            vocab_size=100,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_local_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=64,
        )

    def test_moe_model_forward(self):
        """
        Test MoEModel forward pass.
        """
        model = MoEModel(self.config)

        batch_size, seq_len = 2, 10
        input_ids = genesis.randint(0, self.config.vocab_size, (batch_size, seq_len))

        hidden_states, past_kv = model(input_ids, use_cache=True)

        self.assertEqual(hidden_states.shape, (batch_size, seq_len, self.config.hidden_size))
        self.assertIsNotNone(past_kv)
        self.assertEqual(len(past_kv), self.config.num_hidden_layers)

    def test_moe_for_causal_lm_forward(self):
        """
        Test MoEForCausalLM forward pass.
        """
        model = MoEForCausalLM(self.config)

        batch_size, seq_len = 2, 10
        input_ids = genesis.randint(0, self.config.vocab_size, (batch_size, seq_len))

        logits = model(input_ids)

        self.assertEqual(logits.shape, (batch_size, seq_len, self.config.vocab_size))

    def test_moe_for_causal_lm_with_labels(self):
        """
        Test MoEForCausalLM forward pass with labels (training mode).
        """
        model = MoEForCausalLM(self.config)

        batch_size, seq_len = 2, 10
        input_ids = genesis.randint(0, self.config.vocab_size, (batch_size, seq_len))
        labels = genesis.randint(0, self.config.vocab_size, (batch_size, seq_len))

        loss, logits = model(input_ids, labels=labels)

        self.assertIsInstance(loss, Tensor)
        self.assertEqual(logits.shape, (batch_size, seq_len, self.config.vocab_size))

    def test_moe_model_parameter_count(self):
        """
        Test that model has expected number of parameters.
        """
        model = MoEForCausalLM(self.config)
        num_params = sum(p.numel() for p in model.parameters())

        # Model should have a reasonable number of parameters
        self.assertGreater(num_params, 0)

    def test_deepseek_style_config(self):
        """
        Test DeepSeek-style configuration with shared experts.
        """
        config = MoEConfig(
            vocab_size=100,
            hidden_size=128,
            num_hidden_layers=2,
            num_local_experts=8,
            num_shared_experts=2,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=128,
        )

        model = MoEForCausalLM(config)

        batch_size, seq_len = 2, 10
        input_ids = genesis.randint(0, config.vocab_size, (batch_size, seq_len))

        logits = model(input_ids)
        self.assertEqual(logits.shape, (batch_size, seq_len, config.vocab_size))

    def test_mixtral_style_config(self):
        """
        Test Mixtral-style configuration without shared experts.
        """
        config = MoEConfig(
            vocab_size=100,
            hidden_size=128,
            num_hidden_layers=2,
            num_local_experts=4,
            num_shared_experts=None,
        )

        model = MoEForCausalLM(config)

        batch_size, seq_len = 2, 10
        input_ids = genesis.randint(0, config.vocab_size, (batch_size, seq_len))

        logits = model(input_ids)
        self.assertEqual(logits.shape, (batch_size, seq_len, config.vocab_size))

    def test_from_pretrained(self):
        """
        Test loading model from pretrained config.
        """
        model = MoEForCausalLM.from_pretrained("moe-small")
        self.assertIsInstance(model, MoEForCausalLM)
        self.assertEqual(model.config.hidden_size, 768)


class TestMoETraining(unittest.TestCase):
    """
    Test cases for MoE model training.
    """

    def test_backward_pass(self):
        """
        Test that backward pass works correctly.
        """
        config = MoEConfig(
            vocab_size=50,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_local_experts=2,
        )

        model = MoEForCausalLM(config)

        # Create dummy data
        batch_size, seq_len = 2, 8
        input_ids = genesis.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = genesis.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward pass
        loss, _ = model(input_ids, labels=labels)

        # Backward pass
        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

    def test_optimizer_step(self):
        """
        Test that optimizer can update model parameters.
        """
        import genesis.optim as optim

        config = MoEConfig(
            vocab_size=50,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_local_experts=2,
        )

        model = MoEForCausalLM(config)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Training step
        batch_size, seq_len = 2, 8
        input_ids = genesis.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = genesis.randint(0, config.vocab_size, (batch_size, seq_len))

        loss, _ = model(input_ids, labels=labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that parameters changed
        for initial_param, current_param in zip(initial_params, model.parameters()):
            if current_param.requires_grad:
                # Parameters should be different after update
                param_changed = not genesis.allclose(initial_param, current_param)
                self.assertTrue(param_changed)


def run_tests():
    """
    Run all tests.
    """
    unittest.main(argv=[""], verbosity=2, exit=False)


if __name__ == "__main__":
    run_tests()
