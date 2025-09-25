"""
Utilities for handling noisy imports and ALSA error suppression.
"""
import os
from typing import Optional, Tuple, Any


def import_quietly(func):
    """
    Run a function while suppressing ALL stderr output.
    Simple and reliable approach for noisy library imports.
    """
    import sys
    import warnings
    
    # Save original stderr
    original_stderr = sys.stderr
    
    try:
        # Suppress Python warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Redirect stderr to devnull
            with open(os.devnull, 'w') as devnull:
                sys.stderr = devnull
                return func()
    finally:
        # Always restore stderr
        sys.stderr = original_stderr


def import_optional_audio_libs() -> Tuple[Optional[Any], Optional[Any]]:
    """
    DISABLED: Returns (None, None) - was causing segfaults.
    """
    return None, None


# Initialize environment with comprehensive warning suppression
os.environ.setdefault("JACK_NO_START_SERVER", "1")

# Suppress Google Cloud warnings
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")


def with_suppressed_audio_warnings(func):
    """
    Context manager to suppress native audio warnings during a function call.
    This temporarily redirects stderr at the file descriptor level.
    """
    def wrapper(*args, **kwargs):
        # Save original stderr
        try:
            original_stderr_fd = os.dup(2)
            null_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(null_fd, 2)
            os.close(null_fd)
        except Exception:
            original_stderr_fd = None
        
        try:
            # Run the function with suppressed stderr
            return func(*args, **kwargs)
        finally:
            # Always restore stderr
            if original_stderr_fd is not None:
                try:
                    os.dup2(original_stderr_fd, 2)
                    os.close(original_stderr_fd)
                except Exception:
                    pass
    
    return wrapper


# Don't suppress globally - let audio functions use the wrapper instead
