    Monkey-patch Aim's Stat.get_stats to only return GPUs visible to CUDA.
    This ensures that each job only tracks its assigned GPU.

    Must be called before any Aim imports that use ResourceTracker.