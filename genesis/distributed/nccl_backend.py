"""
NCCL backend for Genesis distributed training.

Uses direct ctypes bindings to NCCL library for high-performance
GPU-to-GPU communication without external dependencies.
"""

import os
import threading
import socket
import time
import logging
from typing import Optional, List
import genesis
from .comm import ReduceOp
from .nccl import (
    NCCLCommunicator, generate_unique_id, is_nccl_available,
    genesis_reduce_op_to_nccl, ncclUniqueId
)

logger = logging.getLogger(__name__)


class NCCLBackend:
    """NCCL backend using direct ctypes bindings.
    
    Provides high-performance distributed communication using
    NVIDIA NCCL library through native ctypes interface.
    """
    
    def __init__(self):
        self.communicator = None
        self.world_size = None
        self.rank = None
        self.local_rank = None
        self.initialized = False
        
    def init_process_group(self, init_method: Optional[str], world_size: int, rank: int):
        """Initialize process group using NCCL."""
        if not is_nccl_available():
            raise RuntimeError(
                "NCCL library not available. Please install NCCL and ensure "
                "libnccl.so is in your library path or set GENESIS_NCCL_PATH."
            )
        
        # Check CUDA availability
        if not genesis.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA. No CUDA devices found.")
            
        self.world_size = world_size
        self.rank = rank
        device_count = genesis.cuda.device_count()
        if device_count == 0:
            raise RuntimeError("No CUDA devices available for distributed training.")
            
        self.local_rank = int(os.environ.get("LOCAL_RANK", rank % device_count))
        
        # Note: Genesis handles CUDA device context automatically
        # No need to explicitly set device in single-process mode
        
        if world_size == 1:
            # Single process - no communication needed
            self.communicator = None
            self.initialized = True
            logger.info(f"NCCL: Single process mode, rank {rank}")
            return
            
        # Multi-process initialization
        if init_method is None:
            init_method = "tcp://localhost:23456"
            
        # Exchange NCCL unique ID between processes
        if rank == 0:
            # Master process: generate unique ID and broadcast
            unique_id = generate_unique_id()
            self._broadcast_unique_id(unique_id, init_method)
        else:
            # Worker processes: receive unique ID
            unique_id = self._receive_unique_id(init_method)
        
        # Initialize NCCL communicator
        self.communicator = NCCLCommunicator()
        self.communicator.init_rank(world_size, unique_id, rank)

        self.initialized = True
        logger.info(f"NCCL: Initialized rank {rank}/{world_size} on device {self.local_rank}")
            
    def destroy(self):
        """Clean up NCCL resources."""
        if self.initialized and self.communicator is not None:
            self.communicator.destroy()
        self.initialized = False
        logger.info("NCCL: Destroyed")
        
    def barrier(self):
        """Synchronization barrier (standard API)."""
        if not self.initialized:
            raise RuntimeError("Backend not initialized")

        if self.communicator is None:
            # Single process - no-op
            return

        # Use all_reduce on dummy data as barrier
        # Use current rank's GPU device
        device = genesis.device(f"cuda:{self.local_rank}")
        dummy = genesis.ones([1], dtype=genesis.float32, device=device)
        self.all_reduce(dummy, ReduceOp.SUM)
        
    def all_reduce(self, tensor: genesis.Tensor, op: ReduceOp, async_op: bool = False):
        """Perform all_reduce operation."""
        if not self.initialized:
            raise RuntimeError("Backend not initialized")
            
        if self.communicator is None:
            # Single process - no communication needed
            return None
            
        # Convert operation
        nccl_op = genesis_reduce_op_to_nccl(op)
        
        # Get CUDA stream for async operation
        stream_ptr = 0  # Use default stream
        if async_op:
            # Could create custom stream here for async operations
            pass
            
        # Perform all_reduce
        self.communicator.all_reduce(tensor, nccl_op, stream_ptr)
        
        return None
        
    def all_gather(self, tensor_list: List[genesis.Tensor], tensor: genesis.Tensor, async_op: bool = False):
        """Perform all_gather operation.""" 
        if not self.initialized:
            raise RuntimeError("Backend not initialized")
            
        if self.communicator is None:
            # Single process - copy tensor to all positions
            for i, out_tensor in enumerate(tensor_list):
                tensor_list[i] = tensor.clone()
            return None
            
        # For now, implement using multiple broadcast operations
        # A full implementation would use ncclAllGather directly
        for i in range(self.world_size):
            if i == self.rank:
                # Copy own tensor to output list (standard API)
                tensor_list[i].copy_(tensor)
            else:
                # Broadcast from rank i to current rank
                tensor_list[i].copy_(tensor)  # Initialize with same shape
                self.broadcast(tensor_list[i], i, async_op)
                
        return None
        
    def broadcast(self, tensor: genesis.Tensor, src: int, async_op: bool = False):
        """Perform broadcast operation using native NCCL broadcast."""
        if not self.initialized:
            raise RuntimeError("Backend not initialized")

        if self.communicator is None:
            # Single process - no-op
            return None

        # Get CUDA stream for async operation
        stream_ptr = 0  # Use default stream
        if async_op:
            # Could create custom stream here for async operations
            pass

        # Use native NCCL broadcast (efficient, single operation)
        self.communicator.broadcast(tensor, src, stream_ptr)

        return None
        
    def reduce_scatter(self, output: genesis.Tensor, input_list: List[genesis.Tensor],
                      op: ReduceOp, async_op: bool = False):
        """Perform reduce_scatter operation."""
        if not self.initialized:
            raise RuntimeError("Backend not initialized")

        if self.communicator is None:
            # Single process - just copy first input to output (standard API)
            if input_list:
                output.copy_(input_list[0])
            return None

        # Simplified implementation: concatenate inputs, reduce, then slice
        if input_list:
            # Concatenate all inputs
            concat_tensor = genesis.cat(input_list, dim=0)

            # All-reduce the concatenated tensor
            self.all_reduce(concat_tensor, op, async_op)

            # Extract the portion for current rank (standard API)
            chunk_size = output.numel()
            start_idx = self.rank * chunk_size
            end_idx = start_idx + chunk_size

            output.copy_(concat_tensor[start_idx:end_idx])
            
        return None
        
    def _get_device_count(self) -> int:
        """Get number of available CUDA devices."""
        if not genesis.cuda.is_available():
            raise RuntimeError("CUDA not available")
        return genesis.cuda.device_count()
            
    def _broadcast_unique_id(self, unique_id: ncclUniqueId, init_method: str):
        """Broadcast NCCL unique ID to other processes."""
        if not init_method.startswith("tcp://"):
            raise ValueError("Only TCP init_method supported currently")
            
        # Extract host and port
        url = init_method[6:]  # Remove 'tcp://'
        if ":" in url:
            host, port = url.split(":")
            port = int(port)
        else:
            host, port = url, 23456
            
        def server_thread():
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            try:
                server.bind((host, port))
                server.listen(self.world_size - 1)
                
                # Convert unique ID to bytes for transmission
                uid_bytes = bytes(unique_id.internal)
                
                for _ in range(self.world_size - 1):
                    conn, addr = server.accept()
                    try:
                        # Send unique ID length first, then the data
                        conn.send(len(uid_bytes).to_bytes(4, "little"))
                        conn.send(uid_bytes)
                    finally:
                        conn.close()
                        
            finally:
                server.close()
                
        thread = threading.Thread(target=server_thread)
        thread.daemon = True
        thread.start()
        
        # Give server time to start
        time.sleep(0.1)
        
    def _receive_unique_id(self, init_method: str) -> ncclUniqueId:
        """Receive NCCL unique ID from master process."""
        if not init_method.startswith("tcp://"):
            raise ValueError("Only TCP init_method supported currently")
            
        url = init_method[6:]
        if ":" in url:
            host, port = url.split(":")
            port = int(port)
        else:
            host, port = url, 23456
            
        # Connect to master and receive unique ID
        for attempt in range(60):  # Try for 1 minute
            try:
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.settimeout(5.0)
                client.connect((host, port))
                
                # Read length first
                length_bytes = client.recv(4)
                if len(length_bytes) != 4:
                    raise RuntimeError("Failed to receive unique ID length")
                    
                length = int.from_bytes(length_bytes, "little")
                
                # Read the unique ID data
                uid_bytes = b''
                while len(uid_bytes) < length:
                    chunk = client.recv(length - len(uid_bytes))
                    if not chunk:
                        raise RuntimeError("Connection closed while receiving unique ID")
                    uid_bytes += chunk
                    
                client.close()
                
                # Reconstruct unique ID
                unique_id = ncclUniqueId()
                for i, byte_val in enumerate(uid_bytes[:128]):  # Only use first 128 bytes
                    unique_id.internal[i] = byte_val
                    
                return unique_id
                
            except Exception as e:
                if attempt == 59:
                    raise RuntimeError(f"Failed to receive unique ID after 60 attempts: {e}")
                time.sleep(1)
                
        raise RuntimeError("Failed to receive unique ID")