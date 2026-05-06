import sys
import time
from loguru import logger

from app.core.config import settings


class PlainConsole:
    def print(self, *args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        text = sep.join(str(arg) for arg in args)
        sys.stdout.write(text + end)


console = PlainConsole()

logger.remove()
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG" if settings.debug_pipeline else "INFO",
)


def print_module_start(name: str):
    if settings.debug_pipeline:
        console.print(f"\n>>> Starting Module: {name}")


def print_module_summary(
    module_name: str,
    status: str,
    duration_ms: float,
    input_summary: dict,
    output_summary: dict,
    error: str = None,
):
    line = f"[{status.upper():<7}] Module: {module_name:<10} | Time: {duration_ms:.2f}ms"

    if status == "success":
        in_str = ", ".join([f"{k}: {v}" for k, v in input_summary.items()])
        out_str = ", ".join([f"{k}: {v}" for k, v in output_summary.items()])
        line += f" | Input: {in_str} | Result: {out_str}"

    console.print(line)

    if settings.debug_pipeline and status == "success" and output_summary.get("preview"):
        preview = output_summary["preview"]
        console.print(f"--- {module_name} Detail Output ---")
        console.print(preview)
        console.print(f"--- End {module_name} Detail Output ---")

    if error:
        console.print(f"ERROR: {error}")


def print_final_answer(answer: str, run_id: str = None):
    console.print("\n" + "=" * 80)
    console.print(f"Assistant:\n\n{answer}")
    if run_id:
        console.print(f"\nRun ID: {run_id}")
    console.print("=" * 80 + "\n")


class PipelineProgress:
    def __init__(self, query: str):
        self.query = query
        self.start_time = time.time()

    def __enter__(self):
        console.print(f"\nQuery: {self.query}\n")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (time.time() - self.start_time) * 1000
        console.print(f"Total pipeline time: {duration:.2f}ms")
