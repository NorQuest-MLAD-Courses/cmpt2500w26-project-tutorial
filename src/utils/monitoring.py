"""
monitoring.py — Prometheus metrics for training processes.
Starts a lightweight HTTP server to expose training metrics
so Prometheus can scrape them independently of the Flask API.
"""

import os
import time
import threading
import psutil
from prometheus_client import start_http_server, Counter, Gauge, Histogram


class TrainingMonitor:
    """Base training monitor with common metrics."""

    def __init__(self, port=8002):
        self.port = port

        # Training progress
        self.epochs_completed = Counter(
            "training_epochs_completed_total",
            "Total training epochs completed",
        )
        self.training_duration = Gauge(
            "training_duration_seconds",
            "Time elapsed since training started",
        )

        # Model performance
        self.train_accuracy = Gauge("training_accuracy", "Training set accuracy")
        self.val_accuracy = Gauge("validation_accuracy", "Validation set accuracy")
        self.train_f1 = Gauge("training_f1_score", "Training set F1 score")
        self.val_f1 = Gauge("validation_f1_score", "Validation set F1 score")

        # Feature importance (top features)
        self.feature_importance = Gauge(
            "feature_importance",
            "Feature importance value",
            ["feature_name"],
        )

        # System resources
        self.cpu_usage = Gauge("training_cpu_usage_percent", "CPU usage during training")
        self.memory_usage = Gauge("training_memory_usage_bytes", "Memory usage during training")
        self.memory_percent = Gauge("training_memory_usage_percent", "Memory usage percentage")

        self._resource_thread = None
        self._stop_event = threading.Event()

    def start(self):
        """Start the metrics HTTP server and resource monitoring thread."""
        start_http_server(self.port)
        print(f"Training metrics server started on port {self.port}")

        self._stop_event.clear()
        self._resource_thread = threading.Thread(
            target=self._monitor_resources, daemon=True
        )
        self._resource_thread.start()

    def stop(self):
        """Stop the resource monitoring thread."""
        self._stop_event.set()

    def _monitor_resources(self):
        """Update CPU and memory metrics every 5 seconds."""
        process = psutil.Process(os.getpid())
        while not self._stop_event.is_set():
            self.cpu_usage.set(process.cpu_percent())
            mem = process.memory_info()
            self.memory_usage.set(mem.rss)
            self.memory_percent.set(process.memory_percent())
            time.sleep(5)

    def record_epoch(self, train_acc=None, val_acc=None, train_f1=None, val_f1=None):
        """Record metrics for a completed epoch or evaluation round."""
        self.epochs_completed.inc()
        if train_acc is not None:
            self.train_accuracy.set(train_acc)
        if val_acc is not None:
            self.val_accuracy.set(val_acc)
        if train_f1 is not None:
            self.train_f1.set(train_f1)
        if val_f1 is not None:
            self.val_f1.set(val_f1)

    def record_feature_importance(self, importance_dict, top_n=5):
        """Record top-N feature importances."""
        sorted_features = sorted(
            importance_dict.items(), key=lambda x: x[1], reverse=True
        )[:top_n]
        for name, value in sorted_features:
            self.feature_importance.labels(feature_name=name).set(value)

    def set_duration(self, seconds):
        """Set the elapsed training duration."""
        self.training_duration.set(seconds)
