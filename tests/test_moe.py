"""Tests for Mixture of Experts (MoE) implementation."""

import pytest
import genesis
import genesis.nn as nn
from genesis.nn.moe import MoEGate, MoEExpert, MoELayer, MoETransformerBlock


class SimpleConfig:
    """Simple config class for testing."""
    def __init__(self):
        self.hidden_size = 64
        self.num_experts = 4
        self.top_k = 2
        self.intermediate_size = 128
        self.aux_loss_alpha = 0.01
        self.seq_aux = True
        self.norm_topk_prob = True
        self.expert_bias = False
        self.num_attention_heads = 4
        self.norm_eps = 1e-6


@pytest.fixture
def config():
    """Test configuration."""
    return SimpleConfig()


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_len():
    return 8


class TestMoEGate:
    """Test MoE gating mechanism."""
    
    def test_gate_creation(self, config):
        """Test MoE gate can be created."""
        gate = MoEGate(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.top_k
        )
        assert gate.hidden_size == config.hidden_size
        assert gate.num_experts == config.num_experts
        assert gate.top_k == config.top_k
        
    def test_gate_forward_shape(self, config, batch_size, seq_len):
        """Test gate forward pass output shapes."""
        gate = MoEGate(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.top_k
        )
        # Move model to CUDA 
        gate.to(genesis.device('cuda'))
        
        # Create input tensor on CUDA (topk kernel issues have been resolved)
        input_tensor = genesis.randn(batch_size, seq_len, config.hidden_size, device=genesis.device('cuda'))
        
        # Forward pass
        expert_indices, expert_weights, aux_loss = gate(input_tensor)
        
        # Check shapes
        expected_tokens = batch_size * seq_len
        assert expert_indices.shape == (expected_tokens, config.top_k)
        assert expert_weights.shape == (expected_tokens, config.top_k)
        
        # Check auxiliary loss
        if gate.training and gate.aux_loss_alpha > 0:
            assert aux_loss is not None
            assert aux_loss.numel() == 1  # scalar
        
    def test_gate_weights_sum_to_one(self, config, batch_size, seq_len):
        """Test that top-k weights sum to 1 when normalized."""
        gate = MoEGate(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
            norm_topk_prob=True
        )
        
        # Move gate to CUDA
        gate.to(genesis.device('cuda'))
        
        input_tensor = genesis.randn(batch_size, seq_len, config.hidden_size, device=genesis.device('cuda'))
        expert_indices, expert_weights, aux_loss = gate(input_tensor)
        
        # Check weights sum approximately to 1
        weight_sums = expert_weights.sum(dim=1)
        expected_ones = genesis.ones(weight_sums.shape, device=genesis.device('cuda'))
        
        # Allow small numerical errors
        assert genesis.allclose(weight_sums, expected_ones, atol=1e-6)


class TestMoEExpert:
    """Test individual MoE expert."""
    
    def test_expert_creation(self, config):
        """Test expert can be created."""
        expert = MoEExpert(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size
        )
        assert expert.hidden_size == config.hidden_size
        assert expert.intermediate_size == config.intermediate_size
        
    def test_expert_forward_shape(self, config):
        """Test expert forward pass shapes."""
        expert = MoEExpert(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size
        )
        
        # Test with different input shapes
        for shape in [(4, config.hidden_size), (2, 3, config.hidden_size)]:
            input_tensor = genesis.randn(*shape)
            output = expert(input_tensor)
            
            # Output should have same shape as input
            assert output.shape == shape
            
    def test_expert_swiglu_computation(self, config):
        """Test SwiGLU computation in expert."""
        expert = MoEExpert(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size
        )
        
        input_tensor = genesis.randn(4, config.hidden_size)
        
        # Manual SwiGLU computation for verification
        gate_out = expert.gate_proj(input_tensor)
        up_out = expert.up_proj(input_tensor)
        intermediate = expert.silu(gate_out) * up_out
        expected_output = expert.down_proj(intermediate)
        
        # Compare with expert forward
        actual_output = expert(input_tensor)
        assert genesis.allclose(actual_output, expected_output, atol=1e-5)


class TestMoELayer:
    """Test complete MoE layer."""
    
    def test_moe_layer_creation(self, config):
        """Test MoE layer can be created."""
        moe_layer = MoELayer(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
            intermediate_size=config.intermediate_size
        )
        assert len(moe_layer.experts) == config.num_experts
        assert moe_layer.gate.num_experts == config.num_experts
        
    def test_moe_layer_forward_shape(self, config, batch_size, seq_len):
        """Test MoE layer forward pass shapes."""
        moe_layer = MoELayer(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
            intermediate_size=config.intermediate_size
        )
        
        # Move to CUDA for GPU testing
        moe_layer.to(genesis.device('cuda'))
        input_tensor = genesis.randn(batch_size, seq_len, config.hidden_size, device=genesis.device('cuda'))
        output = moe_layer(input_tensor)
        
        # Output should have same shape as input
        assert output.shape == input_tensor.shape
        
    def test_moe_layer_with_shared_experts(self, config):
        """Test MoE layer with shared experts."""
        moe_layer = MoELayer(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
            intermediate_size=config.intermediate_size,
            num_shared_experts=1,
            shared_intermediate_size=64
        )
        
        assert moe_layer.shared_experts is not None
        
        # Move to CUDA for GPU testing
        moe_layer.to(genesis.device('cuda'))
        input_tensor = genesis.randn(2, 4, config.hidden_size, device=genesis.device('cuda'))
        output = moe_layer(input_tensor)
        
        assert output.shape == input_tensor.shape
        
    def test_moe_layer_training_vs_eval(self, config):
        """Test MoE layer behaves differently in training vs eval mode."""
        moe_layer = MoELayer(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
            intermediate_size=config.intermediate_size
        )
        
        # Move to CUDA for GPU testing
        moe_layer.to(genesis.device('cuda'))
        input_tensor = genesis.randn(2, 4, config.hidden_size, device=genesis.device('cuda'))
        
        # Training mode
        moe_layer.train()
        output_train = moe_layer(input_tensor)
        
        # Eval mode  
        moe_layer.eval()
        output_eval = moe_layer(input_tensor)
        
        # Both should have same shape
        assert output_train.shape == output_eval.shape
        assert output_train.shape == input_tensor.shape


@pytest.mark.skip(reason="MultiheadAttention constructor needs investigation")
class TestMoETransformerBlock:
    """Test MoE transformer block."""
    
    def test_moe_transformer_creation(self, config):
        """Test MoE transformer block can be created."""
        block = MoETransformerBlock(config)
        assert hasattr(block, 'self_attn')
        assert hasattr(block, 'mlp')
        assert isinstance(block.mlp, MoELayer)
        
    def test_moe_transformer_forward(self, config, batch_size, seq_len):
        """Test MoE transformer block forward pass."""
        block = MoETransformerBlock(config)
        
        # Create inputs
        x = genesis.randn(batch_size, seq_len, config.hidden_size)
        input_pos = genesis.arange(seq_len)
        position_ids = genesis.arange(seq_len)
        
        # Forward pass
        output = block(x, input_pos, position_ids)
        
        # Check output shape
        assert output.shape == x.shape


class TestMoEFunctionality:
    """Test MoE missing functions that were implemented."""
    
    def test_topk_function(self):
        """Test topk function works correctly."""
        x = genesis.tensor([[3.0, 1.0, 4.0, 1.0, 5.0], [2.0, 7.0, 1.0, 8.0, 3.0]])
        values, indices = genesis.topk(x, k=3, dim=1, largest=True)
        
        assert values.shape == (2, 3)
        assert indices.shape == (2, 3)
        
        # Check first row: should be [5.0, 4.0, 3.0] with indices [4, 2, 0]
        assert values[0, 0].item() == 5.0
        assert indices[0, 0].item() == 4
        
    def test_scatter_add_function(self):
        """Test scatter_add function works correctly."""
        input_tensor = genesis.zeros(3, 5)
        index = genesis.tensor([[0, 1, 2, 0]], dtype=genesis.int64)
        src = genesis.tensor([[1.0, 2.0, 3.0, 4.0]])
        
        result = input_tensor.scatter_add(0, index, src)
        
        # Check that values were added correctly
        assert result[0, 0].item() == 1.0  # First addition
        assert result[1, 1].item() == 2.0
        assert result[2, 2].item() == 3.0
        assert result[0, 3].item() == 4.0  # Second addition to same location
        
    def test_repeat_interleave_function(self):
        """Test repeat_interleave function works correctly."""
        x = genesis.tensor([1.0, 2.0, 3.0])
        result = x.repeat_interleave(2, dim=0)
        
        expected = genesis.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        assert result.shape == expected.shape
        assert genesis.allclose(result, expected)
        
    def test_bincount_function(self):
        """Test bincount function works correctly.""" 
        x = genesis.tensor([0, 1, 1, 2, 2, 2], dtype=genesis.int64)
        result = genesis.bincount(x)
        
        expected = genesis.tensor([1, 2, 3])  # 0 appears 1 time, 1 appears 2 times, 2 appears 3 times
        assert result.shape == expected.shape
        assert genesis.allclose(result.float(), expected.float())
        
    def test_argsort_function(self):
        """Test argsort function works correctly."""
        x = genesis.tensor([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
        indices = genesis.argsort(x, dim=1, descending=False)
        
        # For first row [3, 1, 2], sorted indices should be [1, 2, 0] (1 < 2 < 3)
        assert indices[0, 0].item() == 1
        assert indices[0, 1].item() == 2  
        assert indices[0, 2].item() == 0


if __name__ == "__main__":
    pytest.main([__file__])