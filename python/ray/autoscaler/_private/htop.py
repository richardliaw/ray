import json
import os  # noqa: F401
from queue import Empty, Queue
from typing import Dict, Iterable, Optional, Tuple
from enum import Enum, auto

import datetime
import sys
import threading
import time

import requests
from rich import get_console
from rich.columns import Columns
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Task, TaskID, TextColumn
from rich.table import Table
from rich.text import Text

import ray.ray_constants as ray_constants
from ray.autoscaler._private.load_metrics import LoadMetricsSummary
from ray.autoscaler._private.autoscaler import AutoscalerSummary
from ray.autoscaler._private.util import DEBUG_AUTOSCALING_STATUS

try:
    # Windows
    import msvcrt
    termios = tty = None
except ImportError:
    # Unix
    msvcrt = None
    import termios
    import tty


def wait_key_press() -> str:
    if msvcrt:
        return msvcrt.getch()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def _fmt_bytes(bytes: float) -> str:
    """Pretty format bytes"""
    pwr = 2**10
    suffix = ["B", "KB", "MB", "GB", "TB"]
    keep = bytes
    times = 0
    while bytes >= 1. and times < len(suffix):
        keep = bytes
        bytes /= pwr
        times += 1

    return f"{keep:.2f} {suffix[times-1]}"


def _fmt_timedelta(delta: datetime.timedelta):
    """Pretty format timedelta"""
    h = int(delta.seconds // 3600)
    m = int(delta.seconds / 60 % 60)
    s = int(delta.seconds % 60)

    if delta.days:
        return f"{delta.days}:{h:02d}:{m:02d}:{s:02d}"

    return f"{h:02d}:{m:02d}:{s:02d}"


class Page(Enum):
    PAGE_NODE_INFO = auto()
    PAGE_LOGICAL_VIEW = auto()


class TUIPart:
    def __init__(self, data_manager: "DataManager"):
        self.data_manager = data_manager


class Display(TUIPart):
    def __init__(self, data_manager: "DataManager", event_queue: Queue):
        super(Display, self).__init__(data_manager)

        self.event_queue = event_queue
        self.current_page = Page.PAGE_NODE_INFO

    def handle_queue(self):
        try:
            action = self.event_queue.get_nowait()
        except Empty:
            return

        if action == "page_node_info":
            self.current_page = Page.PAGE_NODE_INFO
        elif action == "page_logical_view":
            self.current_page = Page.PAGE_LOGICAL_VIEW
        else:
            raise RuntimeError(f"Unknown action: {action}")

    def display(self):
        root = Layout(name="root")

        root.split(
            Layout(name="header", size=10), Layout(name="body", ratio=1),
            Layout(name="footer", size=3))

        root["header"].update(Header(self.data_manager))

        if self.current_page == Page.PAGE_NODE_INFO:
            root["body"].update(NodeInfoView(self.data_manager))
        elif self.current_page == Page.PAGE_LOGICAL_VIEW:
            root["body"].update(LogicalView(self.data_manager))
        else:
            raise RuntimeError(f"Unknown page: {self.current_page}")

        root["footer"].update(Footer(self.data_manager))

        return root


class Header(TUIPart):
    def __rich__(self) -> Layout:
        layout = Layout()
        layout.split(
            Layout(name="cluster_resources"),
            Layout(name="autoscaler_status"),
            Layout(name="meta", size=30),
            direction="horizontal")

        layout["cluster_resources"].update(ClusterResources(self.data_manager))
        layout["autoscaler_status"].update(AutoscalerStatus(self.data_manager))
        layout["meta"].update(Meta(self.data_manager))

        return layout


class Meta(TUIPart):
    def __rich__(self) -> Panel:
        return Panel(
            Text(
                f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
                justify="right"),
            title="")


class ClusterResources(TUIPart):
    def __rich__(self) -> Panel:
        return Panel(
            Text(f"Cluster resources", justify="center"),
            title="Cluster resources")


class AutoscalerStatus(TUIPart):
    def __rich__(self) -> Panel:
        return Panel(
            Text(f"Autoscaler status", justify="center"),
            title="Autoscaler status")


class Footer(TUIPart):
    def __rich__(self) -> Layout:
        layout = Layout()

        commands = [
            f"[b]{key}[/b] {desc}"
            for key, desc, _ in DisplayController.bindings
        ]

        layout.update(Columns(commands, equal=True, expand=True))

        return layout


class NodeInfoView(TUIPart):
    def __init__(self, data_manager: "DataManager"):
        super(NodeInfoView, self).__init__(data_manager)
        self.sort_by = "cpu"

    def __rich__(self) -> Layout:
        layout = Layout()

        table = NodeTable(self.data_manager, self.sort_by)
        layout.update(Panel(table, title="Cluster node overview"))

        return layout


class NodeTable(TUIPart):
    def __init__(self,
                 data_manager: "DataManager",
                 sort_by: Optional[str] = None):
        super(NodeTable, self).__init__(data_manager)

        self.sort_by = sort_by

    def __rich__(self) -> Table:
        table = Table(show_header=True, header_style="bold magenta")

        table.add_column("Host", justify="center")
        table.add_column("PID", justify="center")
        table.add_column("Uptime", justify="center")
        table.add_column("CPU", justify="center")
        table.add_column("RAM", justify="center")
        table.add_column("GPU", justify="center")
        table.add_column("Plasma", justify="center")
        table.add_column("Disk", justify="center")
        table.add_column("Sent", justify="center")
        table.add_column("Received", justify="center")
        table.add_column("Logs", justify="center")
        table.add_column("Errors", justify="center")

        for node in self.data_manager.nodes:
            table.add_row(
                *node.node_row(extra=None)[1:], end_section=not node.expanded)

            if node.expanded:
                workers_rows = list(node.worker_rows(extra=self.sort_by))
                for i, row in enumerate(
                        sorted(workers_rows, key=lambda item: -item[0])):
                    table.add_row(
                        *row[1:], end_section=i == len(workers_rows) - 1)

        return table


class LogicalView(TUIPart):
    def __rich__(self) -> Layout:
        layout = Layout()

        layout.update(Panel(Text("No content"), title="Logical  view"))

        return layout


class DataManager:
    def __init__(self, url: str, redis_address="127.0.0.1:6379"):
        self.url = url

        mock = os.environ.get("RAY_HTOP_MOCK")
        self.mock = os.path.expanduser(mock) if mock else None

        self.mock_cache = None
        self.mock_memory_cache = None
        if self.mock:
            with open(self.mock, "rt") as f:
                self.mock_cache = json.load(f)

        self.nodes = []

        # Autoscaler info
        self.ray_address = redis_address
        self.redis_client = None
        mock_a6s = os.environ.get("RAY_HTOP_AS_MOCK")
        self.mock_autoscaler = os.path.expanduser(mock_a6s) if mock_a6s else None
        self.autoscaler_summary = None
        self.lm_summary = None
        self.update()

    def _create_redis_client(
            self, address,
            redis_password=ray_constants.REDIS_DEFAULT_PASSWORD):
        import ray._private.services as services
        if not address:
            address = services.get_ray_address_to_use_or_die()
        redis_client = services.create_redis_client(address, redis_password)
        self.redis_client = redis_client

    def update(self):
        self._load_nodes()
        self._load_autoscaler_state()
        self._load_memory_info()

    def _load_nodes(self):
        if self.mock_cache:
            resp_json = self.mock_cache
        else:
            resp = requests.get(f"{self.url}/nodes?view=details")
            resp_json = resp.json()

        resp_data = resp_json["data"]

        self.nodes = [Node(node_dict) for node_dict in resp_data["clients"]]

    def _load_memory_info(self):
        # Fetch core memory worker stats, store as a dictionary
        if self.mock_memory_cache:
            resp_json = self.mock_memory_cache
        else:
            resp = requests.get(f"{self.url}/memory/memory_table")
            resp_json = resp.json()

    def _load_autoscaler_state(self):
        as_dict = None
        if self.mock_autoscaler:
            if isinstance(self.mock_autoscaler, str):
                with open(self.mock_autoscaler) as f:
                    as_dict = json.loads(f.read())
        else:
            if not self.redis_client:
                self._create_redis_client(self.ray_address)
            status = self.redis_client.hget(DEBUG_AUTOSCALING_STATUS, "value")
            if status:
                status = status.decode("utf-8")
                as_dict = json.loads(status)

        if as_dict:
            self.lm_summary = LoadMetricsSummary(
                **as_dict["load_metrics_report"])
            if "autoscaler_report" in as_dict:
                self.autoscaler_summary = AutoscalerSummary(
                    **as_dict["autoscaler_report"])
            # TODO: process the autoscaler data.


class StaticProgress:
    def __init__(self, *columns, task):
        self.columns = columns
        self.task = task

    def __rich__(self):
        table_columns = [
            column.get_table_column().copy() for column in self.columns
        ]

        table = Table.grid(*table_columns, padding=(0, 1), expand=True)
        table.add_row(*((column.format(
            task=self.task) if isinstance(column, str) else column(self.task))
                        for column in self.columns))
        return table


class Node:
    percent_only = (
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"))

    def _make_task(self):
        task = Task(
            id=TaskID(0),
            description="",
            total=100,
            completed=0,
            _get_time=get_console().get_time)
        task.start_time = 0
        return task

    def __init__(self, data: Optional[Dict] = None):
        self.expanded = True

        self.cpu_task = self._make_task()
        self.cpu_progress = StaticProgress(
            *self.percent_only, task=self.cpu_task)

        self.memory_task = self._make_task()
        self.memory_progress = StaticProgress(
            *self.percent_only, task=self.memory_task)

        self.plasma_task = self._make_task()
        self.plasma_progress = StaticProgress(
            *self.percent_only, task=self.plasma_task)

        self.disk_task = self._make_task()
        self.disk_progress = StaticProgress(
            *self.percent_only, task=self.disk_task)

        self.worker_progresses = {}

        if data:
            self.update(data)

    def update(self, data: Dict):
        self.workers = data["workers"]

        self.bootTime = data["bootTime"]
        self.cpu = data["cpu"]
        self.cpus = data["cpus"]
        self.disk = data["disk"]
        self.hostname = data["hostname"]
        self.mem = data["mem"]
        self.network = data["network"]
        self.now = data["now"]
        self.raylet = data["raylet"]

        self.logCount = data["logCount"]
        self.errorCount = data["errorCount"]

        self.cpu_task.completed = self.cpu
        self.memory_task.completed = self.mem[2]

        plasma_used = self.raylet["objectStoreUsedMemory"]
        plasma_avail = self.raylet["objectStoreAvailableMemory"]

        self.plasma_task.completed = plasma_used / plasma_avail / 100

        self.disk_task.completed = self.disk["/"]["percent"]

        for worker in self.workers:
            pid = worker["pid"]

            if pid not in self.worker_progresses:
                self._make_worker_progresses(pid)

            (cpu_progress, cpu_task), \
                (memory_progress, memory_task) = self.worker_progresses[pid]

            memory_percent = worker["memoryInfo"]["rss"] / \
                worker["memoryInfo"]["vms"]

            cpu_task.completed = worker["cpuPercent"]
            memory_task.completed = memory_percent

    def node_row(self, extra: Optional[str] = None) -> Tuple:
        """Create node row for table.

        First element is ``extra`` element, used for sorting."""
        num_workers = len(self.workers)
        num_cores, num_cpus = self.cpus

        uptime = datetime.timedelta(seconds=self.now - self.bootTime)

        sent, received = self.network

        if extra == "cpu":
            extra_val = self.cpu
        else:
            extra_val = None

        return (
            extra_val,
            Text(self.hostname, justify="left"),
            Text(f"{num_workers} workers / {num_cores} cores", justify="left"),
            Text(f"{_fmt_timedelta(uptime)}", justify="right"),
            self.cpu_progress,
            self.memory_progress,
            "",
            self.plasma_progress,
            self.disk_progress,
            Text(_fmt_bytes(sent), justify="right"),
            Text(_fmt_bytes(received), justify="right"),
            Text(str(self.logCount), justify="right"),
            Text(str(self.errorCount), justify="right"),
        )

    def _make_worker_progresses(self, pid: int):
        cpu_task = self._make_task()
        cpu_progress = StaticProgress(*self.percent_only, task=cpu_task)

        memory_task = self._make_task()
        memory_progress = StaticProgress(*self.percent_only, task=memory_task)

        self.worker_progresses[pid] = (
            (cpu_progress, cpu_task),
            (memory_progress, memory_task),
        )

    def worker_rows(self, extra: Optional[str] = None) -> Iterable[Tuple]:
        """Create worker row for table.

        First element is ``extra`` element, used for sorting."""
        for worker in self.workers:
            uptime = datetime.timedelta(seconds=self.now -
                                        worker["createTime"])

            if not worker["pid"] in self.worker_progresses:
                self._make_worker_progresses(worker["pid"])

            (cpu_progress, cpu_task), \
                (memory_progress, memory_task) = \
                self.worker_progresses[worker["pid"]]

            if extra == "cpu":
                extra_val = worker["cpuPercent"]
            else:
                extra_val = None

            yield (
                extra_val,
                Text(str(worker["pid"]), justify="right"),
                Text(worker["cmdline"][0], justify="right", no_wrap=True),
                Text(f"{_fmt_timedelta(uptime)}", justify="right"),
                cpu_progress,
                memory_progress,
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            )


class DisplayController:
    bindings = [
        ("N", "Node view", "cmd_view_node"),
        ("L", "Logical view", "cmd_view_logical"),
        ("q", "Quit", "cmd_stop"),
    ]

    def __init__(self, display: Display, stop_event: threading.Event,
                 event_queue: Queue):
        self.display = display
        self.stop_event = stop_event
        self.event_queue = event_queue
        self.thread = None

    def listen(self):
        def _run():
            while not self.stop_event.is_set():
                key = wait_key_press()
                self.on_press(key)

        self.thread = threading.Thread(target=_run)
        self.thread.start()

    def shutdown(self):
        self.cmd_stop()
        self.thread.join()

    def on_press(self, key):
        for k, d, cmd in self.bindings:
            if key == k:
                fn = getattr(self, cmd, "cmd_invalid")
                fn()

    def cmd_stop(self):
        self.stop_event.set()

    def cmd_view_node(self):
        self.event_queue.put("page_node_info")

    def cmd_view_logical(self):
        self.event_queue.put("page_logical_view")

    def cmd_invalid(self):
        print("Invalid command called.")


def live():
    should_stop = threading.Event()
    event_queue = Queue()

    data_manager = DataManager("http://localhost:8265")
    display = Display(data_manager, event_queue)

    controller = DisplayController(display, should_stop, event_queue)
    controller.listen()

    with Live(
            display.display(),
            refresh_per_second=4,
            screen=False,
            transient=False,
            redirect_stderr=False,
            redirect_stdout=False) as live:
        while not should_stop.is_set():
            time.sleep(.25)
            data_manager.update()
            display.handle_queue()
            live.update(display.display())

    controller.shutdown()
