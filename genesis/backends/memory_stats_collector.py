"""
Memory statistics collection system for CUDA memory pool optimization.
"""

import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics


@dataclass
class AllocationEvent:
    """Single memory allocation event."""
    timestamp: float
    size: int
    ptr: int
    duration_ns: int
    cache_hit: bool
    thread_id: int = 0


@dataclass 
class DeallocationEvent:
    """Single memory deallocation event."""
    timestamp: float
    size: int
    ptr: int
    lifetime_ms: float
    thread_id: int = 0


class NoOpMemoryStatsCollector:
    """
    No-operation memory statistics collector to avoid deadlock issues.
    """

    def __init__(self, history_size: int = 10000):
        pass

    def record_allocation(self, *args, **kwargs):
        pass

    def record_deallocation(self, *args, **kwargs):
        pass

    def get_enhanced_stats(self):
        return {}

class MemoryStatsCollector:
    """
    Comprehensive memory statistics collector with detailed insights.
    """

    def __init__(self, history_size: int = 10000):
        self.history_size = history_size
        self.start_time = time.time()
        self.lock = threading.Lock()
        
        # Event history
        self.allocation_history = deque(maxlen=history_size)
        self.deallocation_history = deque(maxlen=history_size)
        
        # Active allocations tracking
        self.active_allocations = {}  # ptr -> AllocationEvent
        
        # Aggregated statistics
        self.total_allocations = 0
        self.total_deallocations = 0
        self.peak_concurrent_blocks = 0
        self.peak_memory_mb = 0.0
        self.current_memory_mb = 0.0
        
        # Size distribution
        self.size_distribution = defaultdict(int)
        
        # Thread usage
        self.thread_stats = defaultdict(lambda: {"allocs": 0, "deallocs": 0})
        
    def record_allocation(self, size: int, ptr: int = 0, duration_ns: int = 0, cache_hit: bool = False, 
                         thread_id: int = 0, bucket_size: int = None, allocator_type: str = None, 
                         allocation_time_ns: int = None):
        """Record a memory allocation event."""
        with self.lock:
            timestamp = time.time()
            
            # Use allocation_time_ns if provided, otherwise duration_ns
            actual_duration = allocation_time_ns if allocation_time_ns is not None else duration_ns
            
            event = AllocationEvent(
                timestamp=timestamp,
                size=size,
                ptr=ptr, 
                duration_ns=actual_duration,
                cache_hit=cache_hit,
                thread_id=thread_id
            )
            
            # Add to history
            self.allocation_history.append(event)
            self.active_allocations[ptr] = event
            
            # Update counters
            self.total_allocations += 1
            self.current_memory_mb += size / (1024 * 1024)
            
            # Update peaks
            concurrent_blocks = len(self.active_allocations)
            if concurrent_blocks > self.peak_concurrent_blocks:
                self.peak_concurrent_blocks = concurrent_blocks
                
            if self.current_memory_mb > self.peak_memory_mb:
                self.peak_memory_mb = self.current_memory_mb
            
            # Update distributions
            bucket = self._get_size_bucket(size)
            self.size_distribution[bucket] += 1
            
            # Update thread stats
            self.thread_stats[thread_id]["allocs"] += 1
    
    def record_deallocation(self, size: int, ptr: int = 0, thread_id: int = 0):
        """Record a memory deallocation event."""
        with self.lock:
            timestamp = time.time()
            lifetime_ms = 0.0
            
            # Calculate lifetime if we have allocation record
            if ptr in self.active_allocations:
                alloc_event = self.active_allocations.pop(ptr)
                lifetime_ms = (timestamp - alloc_event.timestamp) * 1000
            
            event = DeallocationEvent(
                timestamp=timestamp,
                size=size,
                ptr=ptr,
                lifetime_ms=lifetime_ms,
                thread_id=thread_id
            )
            
            # Add to history
            self.deallocation_history.append(event)
            
            # Update counters
            self.total_deallocations += 1
            self.current_memory_mb -= size / (1024 * 1024)
            self.current_memory_mb = max(0, self.current_memory_mb)  # Prevent negative
            
            # Update thread stats
            self.thread_stats[thread_id]["deallocs"] += 1
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        with self.lock:
            current_time = time.time()
            collection_duration = current_time - self.start_time
            
            # Basic state
            current_state = {
                "active_blocks": len(self.active_allocations),
                "memory_usage_mb": round(self.current_memory_mb, 3),
                "peak_concurrent_blocks": self.peak_concurrent_blocks,
                "peak_memory_mb": round(self.peak_memory_mb, 3)
            }
            
            # Recent activity (last minute)
            recent_allocs = self._count_recent_events(self.allocation_history, 60)
            recent_deallocs = self._count_recent_events(self.deallocation_history, 60)
            
            recent_activity = {
                "allocations_last_minute": recent_allocs,
                "deallocations_last_minute": recent_deallocs,
                "alloc_rate_per_second": round(recent_allocs / 60.0, 1),
                "dealloc_rate_per_second": round(recent_deallocs / 60.0, 1)
            }
            
            # Performance analysis
            performance = self._analyze_performance()
            
            # Size distribution analysis
            size_dist = self._analyze_size_distribution()
            
            # Lifetime analysis
            lifetime_analysis = self._analyze_lifetimes()
            
            # Thread usage
            thread_usage = {
                "active_threads": len([t for t in self.thread_stats.values() if t["allocs"] > 0]),
                "thread_breakdown": dict(self.thread_stats)
            }
            
            # Memory efficiency
            efficiency = self._calculate_efficiency()
            
            # Temporal patterns
            temporal_patterns = self._analyze_temporal_patterns()
            
            # Fragmentation trend (placeholder)
            fragmentation_trend = self._analyze_fragmentation()
            
            return {
                "collection_metadata": {
                    "start_time": self.start_time,
                    "collection_duration_hours": collection_duration / 3600,
                    "total_events_tracked": len(self.allocation_history) + len(self.deallocation_history),
                    "history_buffer_utilization": f"{len(self.allocation_history)}/{self.history_size}"
                },
                "current_state": current_state,
                "recent_activity": recent_activity,
                "size_distribution": size_dist,
                "performance": performance,
                "lifetime_analysis": lifetime_analysis,
                "temporal_patterns": temporal_patterns,
                "thread_usage": thread_usage,
                "efficiency": efficiency,
                "fragmentation_trend": fragmentation_trend
            }
    
    def _get_size_bucket(self, size: int) -> str:
        """Categorize allocation size into buckets."""
        if size < 1024:
            return f"{size}B"
        elif size < 1024 * 1024:
            return f"{size // 1024}KB"
        else:
            return f"{size // (1024 * 1024)}MB"
    
    def _count_recent_events(self, event_queue: deque, seconds: int) -> int:
        """Count events in the last N seconds."""
        cutoff_time = time.time() - seconds
        count = 0
        for event in reversed(event_queue):
            if event.timestamp >= cutoff_time:
                count += 1
            else:
                break
        return count
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze allocation performance."""
        if not self.allocation_history:
            return {"allocation_latency": "No data"}
        
        # Get recent allocation times
        recent_allocs = list(self.allocation_history)[-1000:]  # Last 1000 allocations
        latencies = [event.duration_ns / 1000 for event in recent_allocs]  # Convert to microseconds
        
        if not latencies:
            return {"allocation_latency": "No data"}
        
        return {
            "allocation_latency": {
                "avg_us": round(statistics.mean(latencies), 2),
                "median_us": round(statistics.median(latencies), 2),
                "p95_us": round(self._percentile(latencies, 95), 2),
                "p99_us": round(self._percentile(latencies, 99), 2),
                "min_us": round(min(latencies), 2),
                "max_us": round(max(latencies), 2)
            }
        }
    
    def _analyze_size_distribution(self) -> Dict[str, Any]:
        """Analyze allocation size patterns."""
        if not self.size_distribution:
            return {"hot_sizes": [], "distribution": {}}
        
        total_allocs = sum(self.size_distribution.values())
        
        # Find hot sizes (>5% of allocations)
        hot_sizes = []
        for size_bucket, count in self.size_distribution.items():
            percentage = count / total_allocs * 100
            if percentage > 5.0:
                hot_sizes.append({
                    "size": size_bucket,
                    "count": count,
                    "percentage": f"{percentage:.1f}%"
                })
        
        # Sort by frequency
        hot_sizes.sort(key=lambda x: x["count"], reverse=True)
        
        return {
            "hot_sizes": hot_sizes[:10],  # Top 10
            "distribution": dict(self.size_distribution)
        }
    
    def _analyze_lifetimes(self) -> Dict[str, Any]:
        """Analyze memory allocation lifetimes."""
        if not self.deallocation_history:
            return {
                "very_short": {"count": 0, "percentage": "0.0%", "total_mb": 0.0, "max_lifetime_ms": 100},
                "short": {"count": 0, "percentage": "0.0%", "total_mb": 0.0, "max_lifetime_ms": 1000}, 
                "medium": {"count": 0, "percentage": "0.0%", "total_mb": 0.0, "max_lifetime_ms": 10000},
                "long": {"count": 0, "percentage": "0.0%", "total_mb": 0.0, "max_lifetime_ms": "unlimited"}
            }
        
        # Categorize by lifetime
        very_short = []  # < 100ms
        short = []       # 100ms - 1s
        medium = []      # 1s - 10s
        long = []        # > 10s
        
        for event in self.deallocation_history:
            if event.lifetime_ms <= 0:
                continue
                
            if event.lifetime_ms < 100:
                very_short.append(event)
            elif event.lifetime_ms < 1000:
                short.append(event)
            elif event.lifetime_ms < 10000:
                medium.append(event)
            else:
                long.append(event)
        
        total_events = len(very_short) + len(short) + len(medium) + len(long)
        if total_events == 0:
            return self._empty_lifetime_analysis()
        
        return {
            "very_short": {
                "count": len(very_short),
                "percentage": f"{len(very_short) / total_events * 100:.1f}%",
                "total_mb": sum(e.size for e in very_short) / (1024 * 1024),
                "max_lifetime_ms": 100
            },
            "short": {
                "count": len(short),
                "percentage": f"{len(short) / total_events * 100:.1f}%", 
                "total_mb": sum(e.size for e in short) / (1024 * 1024),
                "max_lifetime_ms": 1000
            },
            "medium": {
                "count": len(medium),
                "percentage": f"{len(medium) / total_events * 100:.1f}%",
                "total_mb": sum(e.size for e in medium) / (1024 * 1024),
                "max_lifetime_ms": 10000
            },
            "long": {
                "count": len(long),
                "percentage": f"{len(long) / total_events * 100:.1f}%",
                "total_mb": sum(e.size for e in long) / (1024 * 1024),
                "max_lifetime_ms": "unlimited"
            }
        }
    
    def _empty_lifetime_analysis(self) -> Dict[str, Any]:
        """Return empty lifetime analysis."""
        categories = ["very_short", "short", "medium", "long"]
        max_lifetimes = [100, 1000, 10000, "unlimited"]
        
        result = {}
        for cat, max_lt in zip(categories, max_lifetimes):
            result[cat] = {
                "count": 0,
                "percentage": "0.0%",
                "total_mb": 0.0,
                "max_lifetime_ms": max_lt
            }
        return result
    
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal allocation patterns."""
        if not self.allocation_history:
            return {"hourly_distribution": {}}
        
        # Group by hour of day
        hourly_counts = defaultdict(int)
        for event in self.allocation_history:
            hour = time.localtime(event.timestamp).tm_hour
            hourly_counts[hour] += 1
        
        return {
            "hourly_distribution": dict(hourly_counts)
        }
    
    def _calculate_efficiency(self) -> Dict[str, Any]:
        """Calculate memory usage efficiency metrics."""
        if not self.allocation_history:
            return {"efficiency": "No data"}
        
        # Simple efficiency calculation based on cache hit rate
        recent_allocs = list(self.allocation_history)[-1000:]
        if not recent_allocs:
            return {"efficiency": "No data"}
        
        cache_hits = sum(1 for event in recent_allocs if event.cache_hit)
        cache_hit_rate = cache_hits / len(recent_allocs) * 100
        
        return {
            "cache_hit_rate_percent": round(cache_hit_rate, 1),
            "cache_efficiency": "High" if cache_hit_rate > 70 else "Medium" if cache_hit_rate > 30 else "Low"
        }
    
    def _analyze_fragmentation(self) -> Dict[str, str]:
        """Analyze memory fragmentation trends."""
        # Placeholder implementation
        return {
            "trend": "insufficient_data"
        }
    
    def _percentile(self, data: List[float], p: float) -> float:
        """Calculate percentile."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        f = int(k)
        c = k - f
        if f + 1 < len(sorted_data):
            return sorted_data[f] + c * (sorted_data[f + 1] - sorted_data[f])
        else:
            return sorted_data[f]


# Global stats collector instance
_global_stats_collector = None
_stats_lock = threading.Lock()


def get_stats_collector() -> MemoryStatsCollector:
    """Get global memory statistics collector instance."""
    global _global_stats_collector
    with _stats_lock:
        if _global_stats_collector is None:
            # Use NoOp collector to avoid deadlock issues
            _global_stats_collector = NoOpMemoryStatsCollector()
        return _global_stats_collector


def reset_stats_collector():
    """Reset the global stats collector (for testing)."""
    global _global_stats_collector
    with _stats_lock:
        _global_stats_collector = None