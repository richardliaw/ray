from queue import Empty, Queue
from typing import Dict, Optional, Tuple
from enum import Enum, auto

import datetime
import sys
import threading
import time

from rich.columns import Columns
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
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
    def __rich__(self) -> Layout:
        layout = Layout()

        table = NodeTable(self.data_manager)
        layout.update(Panel(table, title="Cluster node overview"))

        return layout


class NodeTable(TUIPart):
    def __init__(self, data_manager: "DataManager"):
        super(NodeTable, self).__init__(data_manager)

    def __rich__(self) -> Table:
        table = Table(show_header=True, header_style="bold magenta")

        table.add_column("Host", justify="center")
        table.add_column("PID", justify="center")
        table.add_column("Uptime (s)", justify="center")
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
            table.add_row(*node.node_row())

        return table


class LogicalView(TUIPart):
    def __rich__(self) -> Layout:
        layout = Layout()

        layout.update(Panel(Text("No content"), title="Logical  view"))

        return layout


class DataManager:
    def __init__(self, mock_autoscaler_data=None):
        # Node info
        self.nodes = []

        # Autoscaler info
        self.redis_client = None
        self.mock_autoscaler_data = mock_autoscaler_data
        self.autoscaler_summary = None
        self.lm_summary = None
        self.update()

    def _create_redis_client(self, address, redis_password=ray_constants.REDIS_DEFAULT_PASSWORD):
        import ray._private.services as services
        if not address:
            address = services.get_ray_address_to_use_or_die()
        redis_client = services.create_redis_client(address, redis_password)
        self.redis_client = redis_client

    def update(self):
        # Todo: fetch data
        # from ray.new_dashboard.datacenter import DataOrganizer
        # import asyncio
        # node_details = asyncio.run(DataOrganizer.get_all_node_details())
        # raise RuntimeError(node_details)

        import json
        with open("/tmp/test.json", "rt") as f:
            self.nodes = [Node(json.load(f))]

        self._load_autoscaler_state()

    def _load_autoscaler_state(self):
        as_dict = None
        if self.mock_autoscaler_data:
            with open(self.mock_autoscaler_data) as f:
                as_dict = json.loads(f.read())
        else:
            if not self.redis_client:
                self._create_redis_client()
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


class Node:
    def __init__(self, data: Optional[Dict] = None):
        percent_only = (
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"))

        self.cpu_progress = Progress(*percent_only)
        self.cpu_task = self.cpu_progress.add_task("CPU", total=100)

        self.memory_progress = Progress(*percent_only)
        self.memory_task = self.memory_progress.add_task("Memory", total=100)

        self.plasma_progress = Progress(*percent_only)
        self.plasma_task = self.plasma_progress.add_task("Plasma", total=100)

        self.disk_progress = Progress(*percent_only)
        self.disk_task = self.disk_progress.add_task("Disk", total=100)

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
        self.net = data["net"]
        self.now = data["now"]
        self.raylet = data["raylet"]

        self.logCount = data["logCount"]
        self.errorCount = data["errorCount"]

        self.cpu_progress.update(self.cpu_task, completed=self.cpu)
        self.memory_progress.update(self.memory_task, completed=self.mem[2])

        plasma_used = self.raylet["objectStoreUsedMemory"]
        plasma_avail = self.raylet["objectStoreAvailableMemory"]
        self.plasma_progress.update(
            self.plasma_task, completed=plasma_used / plasma_avail / 100)

        self.disk_progress.update(
            self.disk_task, completed=self.disk["/"]["percent"])

    def node_row(self) -> Tuple:
        num_workers = len(self.workers)
        num_cores, num_cpus = self.cpus

        uptime = datetime.timedelta(seconds=self.now - self.bootTime)

        sent, received = self.net

        return (
            Text(self.hostname, justify="left"),
            Text(f"{num_workers} workers / {num_cores} cores", justify="left"),
            Text(f"{uptime} seconds", justify="left"),
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

    data_manager = DataManager()
    display = Display(data_manager, event_queue)

    controller = DisplayController(display, should_stop, event_queue)
    controller.listen()

    with Live(
            display.display(),
            refresh_per_second=4,
            screen=False,
            redirect_stderr=False,
            redirect_stdout=False) as live:
        while not should_stop.is_set():
            time.sleep(0.25)
            display.handle_queue()
            live.update(display.display())

    controller.shutdown()
