# Basic Resource Usage Monitoring

## 1. Introduction

Monitoring basic resource usage (CPU, memory, disk, network) is crucial for understanding the performance, stability, and cost-effectiveness of your application. It helps in identifying bottlenecks, preventing outages due to resource exhaustion, and optimizing resource allocation.

## 2. Local Development / Bare Metal Servers

When running the application on a local machine for development or on a bare-metal server, several command-line tools are readily available:

*   **`top` / `htop`**:
    *   Provide a dynamic real-time view of a running system.
    *   Show CPU usage, memory usage, running processes, and more.
    *   `htop` is an interactive version of `top` with a more user-friendly interface.
    *   Usage: Simply type `top` or `htop` in the terminal.

*   **`vmstat`**:
    *   Reports virtual memory statistics.
    *   Provides information about processes, memory, paging, block IO, traps, and CPU activity.
    *   Usage: `vmstat [interval] [count]` (e.g., `vmstat 1 10` for updates every second, 10 times).

*   **`psutil` (Python)**:
    *   A cross-platform Python library for retrieving information on running processes and system utilization (CPU, memory, disks, network, sensors).
    *   Useful for building custom monitoring scripts or integrating resource checks directly into your Python application if needed.
    *   Example:
        ```python
        import psutil
        # CPU usage
        print(f"CPU Percent: {psutil.cpu_percent(interval=1)}%")
        # Memory usage
        mem = psutil.virtual_memory()
        print(f"Memory: Total={mem.total}, Available={mem.available}, Used={mem.used} ({mem.percent}%)")
        ```

## 3. Cloud Provider Tools

Most cloud providers offer built-in monitoring services for their compute instances:

*   **AWS CloudWatch Metrics**:
    *   For EC2 instances, CloudWatch provides default metrics like `CPUUtilization`.
    *   To monitor memory (`MemoryUtilization`), disk space, and other OS-level metrics, you need to install and configure the CloudWatch Agent on the EC2 instance.
    *   Metrics can be viewed in the AWS Management Console, and alarms can be set up.

*   **Google Cloud Monitoring**:
    *   For Compute Engine VM instances, Cloud Monitoring provides a suite of metrics, including CPU usage, disk I/O, and network traffic by default.
    *   Similar to AWS, an agent (Ops Agent) can be installed for more detailed OS-level metrics like memory usage.
    *   Dashboards and alerting policies can be configured in the Google Cloud Console.

*   **Azure Monitor**:
    *   For Azure Virtual Machines, Azure Monitor provides host-level metrics like CPU, network, and disk utilization.
    *   The Azure Monitor agent (or older Log Analytics agent) can be used to collect more detailed guest OS metrics, including memory.

## 4. Containerized Environments

Monitoring resource usage in containerized environments requires tools that understand container orchestration:

*   **`docker stats`**:
    *   Provides a live stream of resource usage statistics for running Docker containers.
    *   Shows CPU usage, memory usage & limit, network I/O, and block I/O for each container.
    *   Usage: `docker stats [container_id_or_name...]`

*   **Kubernetes**:
    *   **Metrics Server**: A cluster-wide aggregator of resource usage data. It provides basic CPU and memory metrics for nodes and pods, accessible via `kubectl top node` and `kubectl top pod`. This is often used for autoscaling.
    *   **Prometheus & Grafana**: A more comprehensive and advanced solution.
        *   Prometheus scrapes detailed metrics from Kubernetes components and applications (if instrumented).
        *   Grafana is used to visualize these metrics in dashboards.
        *   This setup is powerful but requires more configuration. Many Kubernetes distributions or cloud-managed Kubernetes services offer easier integration paths for Prometheus and Grafana.

## 5. Sentry Resource Monitoring

Sentry's primary focus is on application performance monitoring (APM) and error tracking, including performance issues like N+1 queries, slow database queries, and frontend performance metrics.

While Sentry APM provides insights into how your application code utilizes resources (e.g., time spent in functions, DB query times), it does not typically provide direct host-level resource monitoring (CPU, memory, disk usage of the server/container itself) in the same way that tools like CloudWatch, Prometheus, or `htop` do.

For comprehensive host-level resource monitoring, it's recommended to use dedicated infrastructure monitoring tools, often those provided by your cloud/hosting environment or specialized solutions like Prometheus/Grafana, Datadog, etc., in conjunction with Sentry for application-specific performance and error insights. Some APM tools offer agents that can collect system metrics, but Sentry's Python SDK is primarily focused on the application layer.
