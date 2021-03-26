import json
import os  # noqa: F401
from queue import Empty, Queue
from typing import Dict, Iterable, Optional, Tuple
from enum import Enum, auto

import datetime
import sys
import threading
import time

import ray
import requests
from ray.state import GlobalState
from rich import get_console
from rich.columns import Columns
from rich.console import RenderGroup
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Task, TaskID, TextColumn
from rich.table import Table
from rich.text import Text

import ray.ray_constants as ray_constants
from ray.internal.internal_api import node_stats

from ray.autoscaler._private.load_metrics import LoadMetricsSummary
from ray.autoscaler._private.autoscaler import AutoscalerSummary

from ray.new_dashboard.memory_utils import construct_memory_table, \
    get_group_by_type, get_sorting_type
from ray.new_dashboard.modules.stats_collector.stats_collector_head import \
    node_stats_to_dict

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


def camel_to_snake(string):
    parts = ["_" + i.lower() if i.isupper() else i for i in string]
    return "".join(parts).lstrip("_")


def camel_to_snake_dict(dict_):
    result = {}
    for k, v in dict_.items():
        result[camel_to_snake(k)] = v
    return result


class Page(Enum):
    PAGE_NODE_INFO = auto()
    PAGE_MEMORY_VIEW = auto()


class TUIPart:
    def __init__(self, data_manager: "DataManager"):
        self.data_manager = data_manager


class Display(TUIPart):
    def __init__(self, data_manager: "DataManager", event_queue: Queue):
        super(Display, self).__init__(data_manager)

        self.event_queue = event_queue
        self.current_page = Page.PAGE_NODE_INFO
        self.current_sorting = None

    def handle_queue(self):
        try:
            action, val = self.event_queue.get_nowait()
        except Empty:
            return

        if action == "page":
            self.current_page = val
        elif action == "sort":
            self.current_sorting = val
        else:
            raise RuntimeError(f"Unknown action: {action}")

    def display(self):
        root = Layout(name="root")

        root.split(
            Layout(name="header", size=10), Layout(name="body", ratio=1),
            Layout(name="footer", size=3))

        root["header"].update(Header(self.data_manager))

        if self.current_page == Page.PAGE_NODE_INFO:
            root["body"].update(
                NodeInfoView(self.data_manager, self.current_sorting))
        elif self.current_page == Page.PAGE_MEMORY_VIEW:
            root["body"].update(MemoryView(self.data_manager))
        else:
            raise RuntimeError(f"Unknown page: {self.current_page}")

        root["footer"].update(Footer(self.data_manager, self.current_page))

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
    def __init__(self, data_manager: "DataManager", current_page: Page):
        super(Footer, self).__init__(data_manager)
        self.current_page = current_page

    def __rich__(self) -> Layout:
        layout = Layout()

        commands = [
            f"[b]{key}[/b] {desc}"
            for key, desc, _, page in DisplayController.bindings
            if page == 0 or page == self.current_page
        ]

        layout.update(Columns(commands, equal=True, expand=True))

        return layout


class NodeInfoView(TUIPart):
    def __init__(self, data_manager: "DataManager", sort_by: str = "cpu"):
        super(NodeInfoView, self).__init__(data_manager)
        self.sort_by = sort_by

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

        table.add_column("Host/PID", justify="center")
        table.add_column("Process", justify="center")
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


class MemoryView(TUIPart):
    def __rich__(self) -> Layout:
        layout = Layout()

        layout.update(
            Panel(MemoryTable(self.data_manager), title="Memory view"))

        return layout


class MemoryTable(TUIPart):
    def __rich__(self) -> Table:
        table = Table(
            show_header=False,
            box=None,
            show_edge=False,
        )
        table.add_column()
        table.add_column()

        for row in self.summary_row():
            table.add_row(*row)

        return table

    def summary_row(self, group_by="NODE_ADDRESS", sort_by="OBJECT_SIZE"):
        group_by, sort_by = get_group_by_type(group_by), get_sorting_type(
            sort_by)
        for key, group in self.data_manager.memory_data["group"].items():
            yield self.summary_table(group), self.object_table(group)

    def summary_table(self, group) -> Table:
        table = Table(
            title="Node summary",
            show_header=False,
            expand=False,
            pad_edge=True,
            padding=(0, 1))
        table.add_column()
        table.add_column()
        table.add_column()
        table.add_column()

        columns = [
            "Memory used",
            "Local refs",
            "Pinned count",
            "Pending tasks",
            "Captured in objs",
            "Actor handles",
        ]

        colvals = list(zip(columns, self.summary_data(group["summary"])))

        for i in range(3):
            table.add_row(
                Text(colvals[i][0], style="bold magenta", justify="right"),
                colvals[i][1],
                Text(colvals[i + 3][0], style="bold magenta", justify="right"),
                colvals[i + 3][1],
            )

        return table

    def summary_data(self, summary):
        summary = camel_to_snake_dict(summary)
        # import ipdb; ipdb.set_trace()
        return (
            Text(_fmt_bytes(summary["total_object_size"]), justify="right"),
            Text(str(summary["total_local_ref_count"]), justify="right"),
            Text(str(summary["total_pinned_in_memory"]), justify="right"),
            Text(str(summary["total_used_by_pending_task"]), justify="right"),
            Text(str(summary["total_pinned_in_memory"]), justify="right"),
            Text(str(summary["total_actor_handles"]), justify="right"),
        )

    def object_table(self, group) -> Table:
        table = Table(
            show_header=True,
            title="Node objects",
            header_style="bold magenta")
        table.add_column("IP", justify="center")
        table.add_column("PID", justify="center")
        table.add_column("Type", justify="center")
        table.add_column("Call site", justify="center")
        table.add_column("Size", justify="center")
        table.add_column("Reference Type", justify="center")
        table.add_column("Object Reference", justify="center")

        for row in self.object_rows(group["entries"]):
            table.add_row(*row)

        return table

    def object_rows(self, entries):
        for entry in entries:
            entry = camel_to_snake_dict(entry)
            yield (
                Text(entry["node_ip_address"], justify="right"),
                Text(str(entry["pid"]), justify="right"),
                Text(entry["type"], justify="right"),
                Text(entry["call_site"], justify="right"),
                Text(_fmt_bytes(entry["object_size"]), justify="right"),
                Text(entry["reference_type"], justify="right"),
                Text(entry["object_ref"], justify="right"),
            )


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
                self.mock_cache["data"]["clients"] *= 2

        self.nodes = []

        # Autoscaler info
        self.ray_address = redis_address or \
            ray.services.get_ray_address_to_use_or_die()
        self.redis_client = None
        mock_a6s = os.environ.get("RAY_HTOP_AS_MOCK")
        self.mock_autoscaler = os.path.expanduser(
            mock_a6s) if mock_a6s else None
        self.autoscaler_summary = None
        self.lm_summary = None

        self._fetch_enabled = False
        self.memory_data = None

        self.update()

    def _create_global_state(
            self, address,
            redis_password=ray_constants.REDIS_DEFAULT_PASSWORD):
        state = GlobalState()
        state._initialize_global_state(address, redis_password)
        return state

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
        # state = self._create_global_state(self.ray_address)
        # core_worker_stats = []
        # for raylet in state.node_table():
        #     stats = node_stats_to_dict(
        #         node_stats(raylet["NodeManagerAddress"],
        #                    raylet["NodeManagerPort"]))
        #     core_worker_stats.extend(stats["coreWorkersStats"])
        #     assert type(stats) is dict and "coreWorkersStats" in stats
        # print(core_worker_stats)
        # self.memory_data = core_worker_stats

        if self.mock_memory_cache:
            resp_json = self.mock_memory_cache
        else:
            # NOTE: This may not load instantaneously.
            resp = requests.get(f"{self.url}/memory/memory_table")
            resp_json = resp.json()
        resp_data = resp_json["data"]
        self.memory_data = resp_data["memoryTable"]

    def _load_autoscaler_state(self):
        as_dict = None
        if self.mock_autoscaler:
            if isinstance(self.mock_autoscaler, str):
                with open(self.mock_autoscaler) as f:
                    as_dict = json.loads(f.read())
        else:
            resp = requests.get(f"{self.url}/api/cluster_status")
            resp_json = resp.json()
            as_dict = resp_json["data"]["clusterStatus"]

        if as_dict:
            load_metrics = as_dict["loadMetricsReport"]
            load_metrics = {
                camel_to_snake(k): v
                for k, v in load_metrics.items()
            }
            self.lm_summary = LoadMetricsSummary(**load_metrics)
            if "autoscalingStatus" in as_dict:
                autoscaling_status = as_dict["autoscalingStatus"]
                autoscaling_status = {
                    camel_to_snake(k): v
                    for k, v in autoscaling_status.items()
                }
                self.autoscaler_summary = AutoscalerSummary(autoscaling_status)
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

        self.gpu_task_progresses = []

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
        self.mem = data["mem"]  # total, available, pct, used
        self.network = data["network"]  # sent, recv
        self.now = data["now"]
        self.raylet = data["raylet"]

        self.logCount = data["logCount"]
        self.errorCount = data["errorCount"]

        self.cpu_task.completed = self.cpu
        self.memory_task.completed = self.mem[2]

        for i, gpu_dict in enumerate(data["gpus"]):
            if i >= len(self.gpu_task_progresses):
                task = self._make_task()
                self.gpu_task_progresses.append((task,
                                                 StaticProgress(
                                                     *self.percent_only,
                                                     task=task)))
            task, progress = self.gpu_task_progresses[i]
            task.completed = gpu_dict["memory_used"]
            task.total = gpu_dict["memory_total"]

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

            memory_percent = worker["memoryInfo"]["rss"] / self.mem[0]

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
            Text(
                f"{num_workers} workers / {num_cores} cores", justify="right"),
            Text(f"{_fmt_timedelta(uptime)}", justify="right"),
            self.cpu_progress,
            self.memory_progress,
            RenderGroup(
                *[progress for _, progress in self.gpu_task_progresses]),
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
            elif extra == "memory":
                extra_val = worker["memoryInfo"]["rss"] / self.mem[0]
            else:
                extra_val = 0

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
        ("N", "Node view", "cmd_view_node", 0),
        ("M", "Memory view", "cmd_view_memory", 0),
        ("q", "Quit", "cmd_stop", 0),
        ("c", "Sort by CPU", "cmd_sort_cpu", Page.PAGE_NODE_INFO),
        ("m", "Sort by Memory", "cmd_sort_memory", Page.PAGE_NODE_INFO),
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
        for k, d, cmd, _ in self.bindings:
            if key == k:
                fn = getattr(self, cmd, "cmd_invalid")
                fn()

    def cmd_stop(self):
        self.stop_event.set()

    def cmd_view_node(self):
        self.event_queue.put(("page", Page.PAGE_NODE_INFO))

    def cmd_view_memory(self):
        self.event_queue.put(("page", Page.PAGE_MEMORY_VIEW))

    def cmd_sort_cpu(self):
        self.event_queue.put(("sort", "cpu"))

    def cmd_sort_memory(self):
        self.event_queue.put(("sort", "memory"))

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
