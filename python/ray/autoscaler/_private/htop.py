import threading
import time

from rich.console import RenderGroup
from rich.live import Live
from rich.table import Table


class Display:
    def __init__(self):
        pass

    def display(self):
        node_table = self.create_node_table()

        return RenderGroup(node_table)

    def create_node_table(self):
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Date", style="dim", width=12)
        table.add_column("Title")
        table.add_column("Production Budget", justify="right")
        table.add_column("Box Office", justify="right")
        table.add_row("Dev 20, 2019", "Star Wars: The Rise of Skywalker",
                      "$275,000,000", "$375,126,118")
        table.add_row(
            "May 25, 2018",
            "[red]Solo[/red]: A Star Wars Story",
            "$275,000,000",
            "$393,151,347",
        )
        table.add_row(
            "Dec 15, 2017",
            "Star Wars Ep. VIII: The Last Jedi",
            "$262,000,000",
            "[bold]$1,332,539,889[/bold]",
        )
        return table


class DisplayController:
    def __init__(self, display: Display, stop_event: threading.Event):
        self.display = display
        self.stop_event = stop_event

    def on_press(self, key):
        if key == "q":
            self.stop_event.set()


def live():
    display = Display()
    should_stop = threading.Event()
    controller = DisplayController(display, should_stop)

    with Live(display.display(), refresh_per_second=4) as live:
        while not should_stop.is_set():
            live.update(display.display())
            time.sleep(0.25)
            controller.on_press("q")
