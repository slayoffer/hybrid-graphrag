"""Real-time Performance Monitoring for LightRAG Unified Neo4j Storage.

This module provides continuous performance monitoring capabilities
for the unified vector storage system with alerting and optimization recommendations.
"""

import asyncio
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import psutil
from loguru import logger

from ..config import Settings
from ..graph.neo4j_client import Neo4jClient


@dataclass
class PerformanceMetric:
    """Single performance metric data point."""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    metadata: Dict[str, Any] = None
    

@dataclass  
class PerformanceAlert:
    """Performance alert definition."""
    level: str  # "info", "warning", "error", "critical"
    message: str
    timestamp: datetime
    metric_name: str
    value: float
    threshold: float
    

class PerformanceMonitor:
    """Real-time performance monitoring system."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = None
        self.metrics_history: List[PerformanceMetric] = []
        self.alerts: List[PerformanceAlert] = []
        self.is_monitoring = False
        
        # Performance thresholds
        self.thresholds = {
            "query_latency_ms": {
                "warning": 400,
                "error": 500,
                "critical": 1000
            },
            "index_latency_ms": {
                "warning": 100,
                "error": 200,
                "critical": 500
            },
            "memory_usage_mb": {
                "warning": 1000,
                "error": 2000,
                "critical": 4000
            },
            "cpu_usage_percent": {
                "warning": 70,
                "error": 85,
                "critical": 95
            },
            "connection_count": {
                "warning": 80,
                "error": 90,
                "critical": 100
            }
        }
        
        # Monitoring configuration
        self.monitoring_interval = 30  # seconds
        self.history_retention = timedelta(hours=24)  # Keep 24 hours of metrics
        
    async def initialize(self):
        """Initialize monitoring system."""
        self.client = Neo4jClient(self.settings)
        await self.client.health_check()
        logger.info("🔍 Performance monitor initialized")
        
    async def start_monitoring(self, duration_minutes: Optional[int] = None):
        """Start continuous performance monitoring."""
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return
            
        self.is_monitoring = True
        logger.info(f"🚀 Starting performance monitoring (interval: {self.monitoring_interval}s)")
        
        if duration_minutes:
            logger.info(f"📅 Monitoring duration: {duration_minutes} minutes")
            
        start_time = time.time()
        
        try:
            while self.is_monitoring:
                # Collect metrics
                await self.collect_metrics()
                
                # Clean old metrics
                self.cleanup_old_metrics()
                
                # Check for alerts
                await self.check_alerts()
                
                # Check duration limit
                if duration_minutes:
                    elapsed_minutes = (time.time() - start_time) / 60
                    if elapsed_minutes >= duration_minutes:
                        logger.info(f"⏰ Monitoring duration reached: {duration_minutes} minutes")
                        break
                        
                # Wait for next interval
                await asyncio.sleep(self.monitoring_interval)
                
        except Exception as e:
            logger.error(f"❌ Monitoring error: {e}")
        finally:
            self.is_monitoring = False
            logger.info("⏹️ Performance monitoring stopped")
            
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        logger.info("⏹️ Stopping performance monitoring...")
        self.is_monitoring = False
        
    async def collect_metrics(self):
        """Collect performance metrics."""
        timestamp = datetime.now()
        
        try:
            # System metrics
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            cpu_usage = process.cpu_percent()
            
            self.add_metric("memory_usage_mb", memory_usage, "MB", timestamp)
            self.add_metric("cpu_usage_percent", cpu_usage, "%", timestamp)
            
            # Neo4j vector index performance
            index_metrics = await self.client.get_vector_index_performance_metrics()
            
            for index_name, metric in index_metrics.items():
                if "error" not in metric:
                    latency = metric.get("latency_ms", 0)
                    self.add_metric(
                        f"index_latency_ms_{index_name}", 
                        latency, 
                        "ms", 
                        timestamp,
                        {"index_name": index_name}
                    )
                    
            # Connection pool metrics (if available)
            try:
                # This would require custom Neo4j driver instrumentation
                # For now, we'll estimate based on system activity
                connection_estimate = min(10, cpu_usage / 10)  # Simple estimation
                self.add_metric("connection_count", connection_estimate, "count", timestamp)
            except Exception:
                pass
                
            # Log monitoring activity
            if len(self.metrics_history) % 10 == 0:  # Every 10th collection
                logger.info(f"📊 Collected {len(self.metrics_history)} metrics (Memory: {memory_usage:.1f}MB, CPU: {cpu_usage:.1f}%)")
                
        except Exception as e:
            logger.error(f"❌ Metric collection failed: {e}")
            
    def add_metric(self, name: str, value: float, unit: str, timestamp: datetime, metadata: Dict = None):
        """Add a performance metric."""
        metric = PerformanceMetric(
            timestamp=timestamp,
            metric_name=name,
            value=value,
            unit=unit,
            metadata=metadata or {}
        )
        self.metrics_history.append(metric)
        
    async def check_alerts(self):
        """Check metrics against thresholds and generate alerts."""
        if not self.metrics_history:
            return
            
        # Get recent metrics (last 5 minutes)
        recent_time = datetime.now() - timedelta(minutes=5)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > recent_time]
        
        if not recent_metrics:
            return
            
        # Group metrics by name
        metric_groups = {}
        for metric in recent_metrics:
            if metric.metric_name not in metric_groups:
                metric_groups[metric.metric_name] = []
            metric_groups[metric.metric_name].append(metric.value)
            
        # Check thresholds
        for metric_name, values in metric_groups.items():
            if not values:
                continue
                
            avg_value = statistics.mean(values)
            max_value = max(values)
            
            # Find applicable threshold
            threshold_key = None
            for key in self.thresholds.keys():
                if metric_name.startswith(key.replace("_ms", "")) or metric_name == key:
                    threshold_key = key
                    break
                    
            if not threshold_key:
                continue
                
            thresholds = self.thresholds[threshold_key]
            
            # Check threshold levels
            for level in ["critical", "error", "warning"]:
                if level in thresholds and avg_value >= thresholds[level]:
                    # Check if we already have a recent alert for this
                    recent_alert_time = datetime.now() - timedelta(minutes=15)
                    has_recent_alert = any(
                        alert.metric_name == metric_name and 
                        alert.level == level and 
                        alert.timestamp > recent_alert_time
                        for alert in self.alerts
                    )
                    
                    if not has_recent_alert:
                        alert = PerformanceAlert(
                            level=level,
                            message=f"{metric_name} average ({avg_value:.1f}) exceeds {level} threshold ({thresholds[level]})",
                            timestamp=datetime.now(),
                            metric_name=metric_name,
                            value=avg_value,
                            threshold=thresholds[level]
                        )
                        self.alerts.append(alert)
                        
                        # Log the alert
                        level_emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🚨"}
                        logger.warning(f"{level_emoji.get(level, '⚠️')} {level.upper()}: {alert.message}")
                        
                    break  # Only trigger the highest level alert
                    
    def cleanup_old_metrics(self):
        """Remove old metrics beyond retention period."""
        cutoff_time = datetime.now() - self.history_retention
        self.metrics_history = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        # Also cleanup old alerts
        alert_cutoff = datetime.now() - timedelta(hours=6)
        self.alerts = [a for a in self.alerts if a.timestamp > alert_cutoff]
        
    async def get_performance_summary(self, window_minutes: int = 30) -> Dict[str, Any]:
        """Get performance summary for the specified time window."""
        window_start = datetime.now() - timedelta(minutes=window_minutes)
        window_metrics = [m for m in self.metrics_history if m.timestamp > window_start]
        
        if not window_metrics:
            return {"error": "No metrics available for the specified window"}
            
        # Group metrics by name
        metric_groups = {}
        for metric in window_metrics:
            if metric.metric_name not in metric_groups:
                metric_groups[metric.metric_name] = []
            metric_groups[metric.metric_name].append(metric.value)
            
        # Calculate summary statistics
        summary = {
            "window_minutes": window_minutes,
            "metrics_count": len(window_metrics),
            "timestamp": datetime.now().isoformat(),
            "metrics": {}
        }
        
        for metric_name, values in metric_groups.items():
            if values:
                summary["metrics"][metric_name] = {
                    "avg": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                    "count": len(values),
                    "unit": next((m.unit for m in window_metrics if m.metric_name == metric_name), "")
                }
                
        # Add alerts
        window_alerts = [a for a in self.alerts if a.timestamp > window_start]
        summary["alerts"] = {
            "total": len(window_alerts),
            "by_level": {}
        }
        
        for alert in window_alerts:
            level = alert.level
            if level not in summary["alerts"]["by_level"]:
                summary["alerts"]["by_level"][level] = 0
            summary["alerts"]["by_level"][level] += 1
            
        return summary
        
    async def get_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on metrics."""
        recommendations = []
        
        # Analyze recent performance
        recent_summary = await self.get_performance_summary(window_minutes=60)
        
        if "error" in recent_summary:
            return ["No recent metrics available for analysis"]
            
        metrics = recent_summary.get("metrics", {})
        
        # Memory usage recommendations
        memory_metrics = [name for name in metrics.keys() if "memory" in name]
        for metric_name in memory_metrics:
            metric = metrics[metric_name]
            avg_memory = metric["avg"]
            max_memory = metric["max"]
            
            if avg_memory > 2000:  # >2GB average
                recommendations.append(f"High memory usage detected (avg: {avg_memory:.0f}MB) - consider enabling vector quantization")
            elif max_memory > 4000:  # >4GB peak
                recommendations.append(f"Memory spikes detected (max: {max_memory:.0f}MB) - monitor for memory leaks")
                
        # Latency recommendations
        latency_metrics = [name for name in metrics.keys() if "latency" in name]
        for metric_name in latency_metrics:
            metric = metrics[metric_name]
            avg_latency = metric["avg"]
            max_latency = metric["max"]
            
            if avg_latency > 500:
                recommendations.append(f"High latency in {metric_name} (avg: {avg_latency:.0f}ms) - consider HNSW parameter tuning")
            elif max_latency > 1000:
                recommendations.append(f"Latency spikes in {metric_name} (max: {max_latency:.0f}ms) - investigate system resources")
                
        # CPU recommendations
        cpu_metrics = [name for name in metrics.keys() if "cpu" in name]
        for metric_name in cpu_metrics:
            metric = metrics[metric_name]
            avg_cpu = metric["avg"]
            
            if avg_cpu > 80:
                recommendations.append(f"High CPU usage (avg: {avg_cpu:.1f}%) - consider scaling or optimization")
                
        # Alert-based recommendations
        alerts = recent_summary.get("alerts", {})
        critical_alerts = alerts.get("by_level", {}).get("critical", 0)
        error_alerts = alerts.get("by_level", {}).get("error", 0)
        
        if critical_alerts > 0:
            recommendations.append(f"Critical alerts detected ({critical_alerts}) - immediate attention required")
        elif error_alerts > 5:
            recommendations.append(f"Multiple error alerts ({error_alerts}) - system tuning recommended")
            
        if not recommendations:
            recommendations.append("System performance within acceptable parameters - no immediate optimizations needed")
            
        return recommendations
        
    async def save_metrics_report(self, filepath: Path, window_minutes: int = 60):
        """Save performance metrics report to file."""
        summary = await self.get_performance_summary(window_minutes)
        recommendations = await self.get_optimization_recommendations()
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "performance_summary": summary,
            "optimization_recommendations": recommendations,
            "system_info": {
                "monitoring_interval": self.monitoring_interval,
                "thresholds": self.thresholds,
                "retention_hours": self.history_retention.total_seconds() / 3600
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"📊 Performance report saved to: {filepath}")
        return filepath
        
    async def close(self):
        """Close monitoring system."""
        if self.is_monitoring:
            await self.stop_monitoring()
            
        if self.client:
            await self.client.close()
            
        logger.info("🔍 Performance monitor closed")


# Utility functions for easy monitoring
async def monitor_performance(duration_minutes: int = 30, settings: Settings = None):
    """Convenience function to monitor performance for a specified duration."""
    if not settings:
        settings = Settings()
        
    monitor = PerformanceMonitor(settings)
    await monitor.initialize()
    
    try:
        await monitor.start_monitoring(duration_minutes=duration_minutes)
        
        # Generate final report
        summary = await monitor.get_performance_summary()
        recommendations = await monitor.get_optimization_recommendations()
        
        logger.info("\n" + "="*50)
        logger.info("PERFORMANCE MONITORING SUMMARY")
        logger.info("="*50)
        
        if "error" not in summary:
            metrics = summary.get("metrics", {})
            alerts = summary.get("alerts", {})
            
            logger.info(f"📊 Metrics collected: {summary['metrics_count']}")
            logger.info(f"⚠️ Alerts generated: {alerts['total']}")
            
            # Show key metrics
            for metric_name, metric_data in metrics.items():
                if "latency" in metric_name:
                    avg_val = metric_data['avg']
                    unit = metric_data['unit']
                    status = "✅" if avg_val <= 500 else "⚠️"
                    logger.info(f"   {metric_name}: {avg_val:.1f}{unit} {status}")
                    
        logger.info("\n💡 Optimization Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"   {i}. {rec}")
            
        return summary, recommendations
        
    finally:
        await monitor.close()


if __name__ == "__main__":
    import argparse
    from ..config import Settings
    
    parser = argparse.ArgumentParser(description="Performance Monitor for LightRAG")
    parser.add_argument("--duration", type=int, default=30, 
                       help="Monitoring duration in minutes (default: 30)")
    parser.add_argument("--interval", type=int, default=30,
                       help="Monitoring interval in seconds (default: 30)")
    
    args = parser.parse_args()
    
    # Run monitoring
    asyncio.run(monitor_performance(duration_minutes=args.duration))