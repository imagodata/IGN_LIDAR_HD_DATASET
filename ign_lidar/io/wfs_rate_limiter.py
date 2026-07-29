"""WFS Rate Limiting and Circuit Breaker

This module provides rate limiting and circuit breaker functionality for WFS requests
to prevent overwhelming IGN's WFS service and gracefully handle service unavailability.

Key Features:
- Token bucket rate limiting to control request rate
- Circuit breaker pattern to stop requests when service is down
- Concurrent request throttling to limit parallel requests
- Automatic recovery when service becomes available again

Author: IGN LiDAR HD Development Team
Date: November 28, 2025
Version: 3.7.0
"""

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable, TypeVar, Any

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Service unavailable, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    
    failure_threshold: int = 5  # Failures before opening circuit
    success_threshold: int = 2  # Successes before closing circuit
    timeout_seconds: float = 60.0  # Time before trying half-open
    half_open_max_calls: int = 3  # Max concurrent calls in half-open state


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for controlling request rate.
    
    Allows burst requests up to bucket capacity while maintaining
    average rate over time.
    
    Example:
        >>> limiter = TokenBucketRateLimiter(rate=2.0, capacity=5)
        >>> 
        >>> # This will succeed immediately (burst allowed)
        >>> for i in range(5):
        >>>     limiter.acquire()
        >>>     make_request()
        >>> 
        >>> # This will wait (rate limiting kicks in)
        >>> limiter.acquire()  # Waits ~0.5s
        >>> make_request()
    """
    
    def __init__(self, rate: float = 2.0, capacity: int = 5):
        """
        Initialize rate limiter.
        
        Args:
            rate: Tokens per second (requests per second)
            capacity: Maximum burst size (bucket capacity)
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = threading.Lock()

        logger.info(f"🚦 TokenBucket rate limiter: {rate} req/s, burst={capacity}")

    def __getstate__(self):
        # threading.Lock is not picklable — dropped here so instances can
        # cross a multiprocessing.Pool boundary (each worker process gets
        # its own fresh, unheld lock instead of sharing the parent's).
        state = self.__dict__.copy()
        del state['lock']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lock = threading.Lock()

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a token (blocking if necessary).
        
        Args:
            timeout: Maximum time to wait (None = wait forever)
        
        Returns:
            True if token acquired, False if timeout
        """
        start_time = time.time()
        
        while True:
            with self.lock:
                # Refill tokens based on time elapsed
                now = time.time()
                elapsed = now - self.last_update
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last_update = now
                
                # Check if token available
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True
            
            # Check timeout
            if timeout is not None and (time.time() - start_time) >= timeout:
                return False
            
            # Wait a bit before retrying
            time.sleep(0.05)
    
    def try_acquire(self) -> bool:
        """
        Try to acquire token without blocking.
        
        Returns:
            True if token acquired, False otherwise
        """
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False


class CircuitBreaker:
    """
    Circuit breaker for WFS service availability.
    
    Prevents cascading failures by stopping requests to failing service
    and periodically testing for recovery.
    
    States:
    - CLOSED: Normal operation, all requests allowed
    - OPEN: Service down, requests blocked immediately
    - HALF_OPEN: Testing recovery, limited requests allowed
    
    Example:
        >>> breaker = CircuitBreaker()
        >>> 
        >>> def fetch_data():
        >>>     if not breaker.allow_request():
        >>>         logger.warning("Circuit breaker OPEN, skipping WFS request")
        >>>         return None
        >>>     
        >>>     try:
        >>>         data = make_wfs_request()
        >>>         breaker.record_success()
        >>>         return data
        >>>     except Exception as e:
        >>>         breaker.record_failure()
        >>>         raise
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.
        
        Args:
            config: Circuit breaker configuration
        """
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        self.lock = threading.Lock()

        logger.info(
            f"⚡ Circuit breaker initialized: "
            f"failure_threshold={self.config.failure_threshold}, "
            f"timeout={self.config.timeout_seconds}s"
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['lock']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lock = threading.Lock()

    def allow_request(self) -> bool:
        """
        Check if request should be allowed.
        
        Returns:
            True if request allowed, False if blocked
        """
        with self.lock:
            if self.state == CircuitState.CLOSED:
                return True
            
            elif self.state == CircuitState.OPEN:
                # Check if timeout elapsed
                if self.last_failure_time is None:
                    return False
                
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.config.timeout_seconds:
                    # Try half-open state
                    logger.info("⚡ Circuit breaker → HALF_OPEN (testing recovery)")
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    self.success_count = 0
                    return True
                
                return False
            
            elif self.state == CircuitState.HALF_OPEN:
                # Allow limited requests in half-open
                if self.half_open_calls < self.config.half_open_max_calls:
                    self.half_open_calls += 1
                    return True
                return False
            
            return False
    
    def record_success(self):
        """Record successful request."""
        with self.lock:
            self.failure_count = 0
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    logger.info("✅ Circuit breaker → CLOSED (service recovered)")
                    self.state = CircuitState.CLOSED
                    self.success_count = 0
                    self.half_open_calls = 0
    
    def record_failure(self):
        """Record failed request."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                # Failed during recovery test, back to open
                logger.warning("⚠️  Circuit breaker → OPEN (recovery test failed)")
                self.state = CircuitState.OPEN
                self.failure_count = 0
                self.success_count = 0
                self.half_open_calls = 0
            
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    logger.warning(
                        f"⚠️  Circuit breaker → OPEN "
                        f"({self.failure_count} consecutive failures)"
                    )
                    self.state = CircuitState.OPEN
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        with self.lock:
            return self.state
    
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self.get_state() == CircuitState.OPEN
    
    def reset(self):
        """Manually reset circuit breaker to closed state."""
        with self.lock:
            logger.info("🔄 Circuit breaker manually reset to CLOSED")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.half_open_calls = 0
            self.last_failure_time = None


class ConcurrencyLimiter:
    """
    Semaphore-based limiter for controlling concurrent requests.
    
    Ensures no more than N requests are in-flight simultaneously.
    
    Example:
        >>> limiter = ConcurrencyLimiter(max_concurrent=3)
        >>> 
        >>> with limiter:
        >>>     # Only 3 threads can be here simultaneously
        >>>     make_wfs_request()
    """
    
    def __init__(self, max_concurrent: int = 3):
        """
        Initialize concurrency limiter.
        
        Args:
            max_concurrent: Maximum concurrent requests allowed
        """
        self.max_concurrent = max_concurrent
        self.semaphore = threading.Semaphore(max_concurrent)
        self.active_count = 0
        self.lock = threading.Lock()

        logger.info(f"🔒 Concurrency limiter: max {max_concurrent} concurrent requests")

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['semaphore']
        del state['lock']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.semaphore = threading.Semaphore(self.max_concurrent)
        self.lock = threading.Lock()

    def __enter__(self):
        """Acquire slot for request."""
        self.semaphore.acquire()
        with self.lock:
            self.active_count += 1
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release slot after request."""
        with self.lock:
            self.active_count -= 1
        self.semaphore.release()
    
    def get_active_count(self) -> int:
        """Get number of active concurrent requests."""
        with self.lock:
            return self.active_count


class WFSRateLimiter:
    """
    Comprehensive rate limiting for WFS requests.
    
    Combines:
    - Token bucket rate limiting (average rate)
    - Circuit breaker (service availability)
    - Concurrency limiting (max parallel requests)
    
    This prevents overwhelming IGN's WFS service and provides
    graceful degradation when service is unavailable.
    
    Example:
        >>> limiter = WFSRateLimiter(
        >>>     requests_per_second=2.0,
        >>>     max_concurrent=3,
        >>>     enable_circuit_breaker=True
        >>> )
        >>> 
        >>> def fetch_data():
        >>>     return limiter.execute(
        >>>         lambda: make_wfs_request(),
        >>>         operation_name="fetch_buildings"
        >>>     )
    """
    
    def __init__(
        self,
        requests_per_second: float = 2.0,
        burst_capacity: int = 5,
        max_concurrent: int = 3,
        enable_circuit_breaker: bool = True,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    ):
        """
        Initialize WFS rate limiter.
        
        Args:
            requests_per_second: Average requests per second allowed
            burst_capacity: Maximum burst size
            max_concurrent: Maximum concurrent requests
            enable_circuit_breaker: Enable circuit breaker pattern
            circuit_breaker_config: Custom circuit breaker config
        """
        self.rate_limiter = TokenBucketRateLimiter(
            rate=requests_per_second,
            capacity=burst_capacity
        )
        self.concurrency_limiter = ConcurrencyLimiter(max_concurrent=max_concurrent)
        
        self.enable_circuit_breaker = enable_circuit_breaker
        self.circuit_breaker = (
            CircuitBreaker(circuit_breaker_config) if enable_circuit_breaker else None
        )
        
        # Statistics
        self.stats = {
            'requests_made': 0,
            'requests_blocked': 0,
            'requests_failed': 0,
            'requests_succeeded': 0,
            'circuit_breaker_trips': 0,
        }
        self.stats_lock = threading.Lock()

        logger.info("🎯 WFS Rate Limiter initialized")
        logger.info(f"   - Rate: {requests_per_second} req/s (burst={burst_capacity})")
        logger.info(f"   - Max concurrent: {max_concurrent}")
        logger.info(f"   - Circuit breaker: {'enabled' if enable_circuit_breaker else 'disabled'}")

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['stats_lock']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.stats_lock = threading.Lock()

    def execute(
        self,
        func: Callable[[], T],
        operation_name: str = "WFS request",
        timeout: Optional[float] = None,
    ) -> Optional[T]:
        """
        Execute function with rate limiting and circuit breaker.
        
        Args:
            func: Function to execute (should perform WFS request)
            operation_name: Name for logging
            timeout: Timeout for rate limiter (None = wait forever)
        
        Returns:
            Result of function, or None if blocked/failed
        """
        # Check circuit breaker first
        if self.circuit_breaker and not self.circuit_breaker.allow_request():
            with self.stats_lock:
                self.stats['requests_blocked'] += 1
            
            if self.circuit_breaker.get_state() == CircuitState.OPEN:
                logger.debug(
                    f"⛔ {operation_name} blocked by circuit breaker "
                    f"(service unavailable)"
                )
            return None
        
        # Acquire rate limit token (may block)
        if not self.rate_limiter.acquire(timeout=timeout):
            with self.stats_lock:
                self.stats['requests_blocked'] += 1
            logger.warning(f"⏱️  {operation_name} timed out waiting for rate limit")
            return None
        
        # Execute with concurrency limiting
        try:
            with self.concurrency_limiter:
                with self.stats_lock:
                    self.stats['requests_made'] += 1
                
                # Execute the actual request
                result = func()
                
                # Record success
                if self.circuit_breaker:
                    self.circuit_breaker.record_success()
                
                with self.stats_lock:
                    self.stats['requests_succeeded'] += 1
                
                return result
        
        except Exception as e:
            # Record failure
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
                
                # Check if circuit just opened
                if self.circuit_breaker.is_open():
                    with self.stats_lock:
                        self.stats['circuit_breaker_trips'] += 1
            
            with self.stats_lock:
                self.stats['requests_failed'] += 1
            
            # Re-raise exception
            raise
    
    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        with self.stats_lock:
            stats = self.stats.copy()
        
        if self.circuit_breaker:
            stats['circuit_breaker_state'] = self.circuit_breaker.get_state().value
        
        stats['active_concurrent'] = self.concurrency_limiter.get_active_count()
        
        return stats
    
    def reset(self):
        """Reset circuit breaker and statistics."""
        if self.circuit_breaker:
            self.circuit_breaker.reset()
        
        with self.stats_lock:
            self.stats = {
                'requests_made': 0,
                'requests_blocked': 0,
                'requests_failed': 0,
                'requests_succeeded': 0,
                'circuit_breaker_trips': 0,
            }


__all__ = [
    'CircuitState',
    'CircuitBreakerConfig',
    'TokenBucketRateLimiter',
    'CircuitBreaker',
    'ConcurrencyLimiter',
    'WFSRateLimiter',
]
