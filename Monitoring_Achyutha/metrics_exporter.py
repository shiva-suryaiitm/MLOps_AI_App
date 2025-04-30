#!/usr/bin/env python3
import subprocess
import time
import logging
import re
from prometheus_client import start_http_server, Gauge, Counter, REGISTRY
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the log file path relative to the script's location
log_file = os.path.join(script_dir, 'logs', 'app.log')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file, mode='a')]
)
logger = logging.getLogger("node_exporter")


# Define metrics for I/O statistics
io_read_rate = Gauge('io_read_rate', 'I/O read rate in KB/s', ['device'])
io_write_rate = Gauge('io_write_rate', 'I/O write rate in KB/s', ['device'])
io_tps = Gauge('io_tps', 'I/O transactions per second', ['device'])
io_read_bytes = Counter('io_read_bytes', 'I/O read bytes in KB', ['device'])
io_write_bytes = Counter('io_write_bytes', 'I/O write bytes in KB', ['device'])

# Define metrics for CPU statistics
cpu_avg_percent = Gauge('cpu_avg_percent', 'CPU usage percentage', ['mode'])

# Define metrics for memory information - all with meminfo_ prefix
meminfo_total = Gauge('meminfo_total', 'Total memory in kB')
meminfo_free = Gauge('meminfo_free', 'Free memory in kB')
meminfo_available = Gauge('meminfo_available', 'Available memory in kB')
meminfo_buffers = Gauge('meminfo_buffers', 'Buffers memory in kB')
meminfo_cached = Gauge('meminfo_cached', 'Cached memory in kB')
meminfo_swap_total = Gauge('meminfo_swap_total', 'Total swap memory in kB')
meminfo_swap_free = Gauge('meminfo_swap_free', 'Free swap memory in kB')

def check_command_exists(command):
    """
    Check if a command exists in the system path
    """
    try:
        subprocess.check_call(['which', command], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def collect_iostat_metrics():
    """
    Collect I/O statistics using iostat command.
    Parse the output and update Prometheus metrics.
    """
    # Check if iostat command exists
    if not check_command_exists('iostat'):
        logger.error("iostat command not found. Please install sysstat package.")
        # Set default values to indicate error state
        cpu_avg_percent.labels('user').set(0)
        cpu_avg_percent.labels('nice').set(0)
        cpu_avg_percent.labels('system').set(0)
        cpu_avg_percent.labels('iowait').set(0)
        cpu_avg_percent.labels('idle').set(100)
        io_tps.labels('none').set(0)
        io_read_rate.labels('none').set(0)
        io_write_rate.labels('none').set(0)
        return

    try:
        # Run iostat command and capture output
        iostat_output = subprocess.check_output(['iostat', '-k'], universal_newlines=True)
        logger.info("Collected iostat data")
        
        # Parse the output
        lines = iostat_output.strip().split('\n')
        
        # Extract CPU statistics from the third line
        if len(lines) >= 3:
            cpu_stats = lines[3].split()
            if len(cpu_stats) >= 6:
                cpu_avg_percent.labels('user').set(float(cpu_stats[0]))
                cpu_avg_percent.labels('nice').set(float(cpu_stats[1]))
                cpu_avg_percent.labels('system').set(float(cpu_stats[2]))
                cpu_avg_percent.labels('iowait').set(float(cpu_stats[3]))
                cpu_avg_percent.labels('idle').set(float(cpu_stats[5]))
                logger.debug(f"Updated CPU metrics: {cpu_stats}")
        
        # Find the line that starts with "Device"
        device_header_index = None
        for i, line in enumerate(lines):
            if line.strip().startswith("Device"):
                device_header_index = i
                break
        
        # If we found the device header, parse the device statistics
        if device_header_index is not None and device_header_index + 1 < len(lines):
            for line in lines[device_header_index + 1:]:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 6:
                        device = parts[0]
                        tps_val = float(parts[1])
                        read_rate_val = float(parts[2])
                        write_rate_val = float(parts[3])
                        
                        # Update metrics
                        io_tps.labels(device).set(tps_val)
                        io_read_rate.labels(device).set(read_rate_val)
                        io_write_rate.labels(device).set(write_rate_val)
                        io_read_bytes.labels(device).inc(read_rate_val)
                        io_write_bytes.labels(device).inc(write_rate_val)
                        
                        logger.debug(f"Updated I/O metrics for {device}: tps={tps_val}, read={read_rate_val}, write={write_rate_val}")
    
    except Exception as e:
        logger.error(f"Error collecting iostat metrics: {e}")

def collect_memory_metrics():
    """
    Collect memory statistics from /proc/meminfo.
    Parse the output and update Prometheus metrics with meminfo_ prefix.
    """
    try:
        # Check if /proc/meminfo exists (should exist in WSL)
        if not os.path.exists('/proc/meminfo'):
            logger.error("/proc/meminfo not found. Are you running in WSL?")
            # Set default values to indicate error state
            meminfo_total.set(0)
            meminfo_free.set(0)
            meminfo_available.set(0)
            return
            
        # Read memory information from /proc/meminfo
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            
        logger.info("Collected memory data")
        
        # Parse memory information using regular expressions
        mem_total_match = re.search(r'MemTotal:\s+(\d+)', meminfo)
        mem_free_match = re.search(r'MemFree:\s+(\d+)', meminfo)
        mem_available_match = re.search(r'MemAvailable:\s+(\d+)', meminfo)
        mem_buffers_match = re.search(r'Buffers:\s+(\d+)', meminfo)
        mem_cached_match = re.search(r'Cached:\s+(\d+)', meminfo)
        mem_swap_total_match = re.search(r'SwapTotal:\s+(\d+)', meminfo)
        mem_swap_free_match = re.search(r'SwapFree:\s+(\d+)', meminfo)
        
        # Update metrics if matches are found - with meminfo_ prefix
        if mem_total_match:
            meminfo_total.set(float(mem_total_match.group(1)))
        if mem_free_match:
            meminfo_free.set(float(mem_free_match.group(1)))
        if mem_available_match:
            meminfo_available.set(float(mem_available_match.group(1)))
        if mem_buffers_match:
            meminfo_buffers.set(float(mem_buffers_match.group(1)))
        if mem_cached_match:
            meminfo_cached.set(float(mem_cached_match.group(1)))
        if mem_swap_total_match:
            meminfo_swap_total.set(float(mem_swap_total_match.group(1)))
        if mem_swap_free_match:
            meminfo_swap_free.set(float(mem_swap_free_match.group(1)))
        
        logger.debug("Updated memory metrics with meminfo_ prefix")
    
    except Exception as e:
        logger.error(f"Error collecting memory metrics: {e}")

def main():
    """
    Main function to start the HTTP server and collect metrics periodically.
    """
    # Start HTTP server to expose metrics
    start_http_server(18000)
    logger.info("Started Prometheus HTTP server on port 18000")
    
    # Log environment information
    if os.path.exists('/proc/version'):
        with open('/proc/version', 'r') as f:
            logger.info(f"Running on: {f.read().strip()}")
    
    logger.info("Verifying required commands and files...")
    if check_command_exists('iostat'):
        logger.info("iostat command found")
    else:
        logger.warning("iostat command not found. Please install 'sysstat' package.")
    
    if os.path.exists('/proc/meminfo'):
        logger.info("/proc/meminfo found")
    else:
        logger.warning("/proc/meminfo not found. Memory metrics will not be collected.")
    
    # Collect metrics periodically
    while True:
        collect_iostat_metrics()
        collect_memory_metrics()
        time.sleep(1)  # Collect metrics every second

if __name__ == "__main__":
    main()
