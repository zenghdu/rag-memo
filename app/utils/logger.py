import sys
import time
from typing import Any, List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.status import Status
from rich.live import Live
from rich.box import ROUNDED
from loguru import logger

from app.core.config import settings

console = Console()

# 配置 loguru 输出到 rich
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG" if settings.debug_pipeline else "INFO"
)

def print_module_start(name: str):
    """打印模块开始运行"""
    if settings.debug_pipeline:
        console.print(f"\n[bold cyan]▶ Starting Module: [inverse]{name}[/inverse][/bold cyan]")

def print_module_summary(
    module_name: str, 
    status: str, 
    duration_ms: float, 
    input_summary: dict, 
    output_summary: dict,
    error: str = None
):
    """显示模块摘要"""
    color = "green" if status == "success" else "red"
    icon = "🟢" if status == "success" else "🔴"
    
    # 基础信息
    line = f"{icon} [bold {color}]Module: {module_name: <10}[/bold {color}] | Status: {status: <8} | Time: [yellow]{duration_ms:.2f}ms[/yellow]"
    
    # 输入输出摘要
    if status == "success":
        in_str = ", ".join([f"{k}: {v}" for k, v in input_summary.items()])
        out_str = ", ".join([f"{k}: {v}" for k, v in output_summary.items()])
        line += f" | Input: [dim]{in_str}[/dim] | Result: [white]{out_str}[/white]"
    
    console.print(line)
    
    # 调试模式下的详细成果展示
    if settings.debug_pipeline and status == "success" and output_summary.get("preview"):
        preview = output_summary["preview"]
        panel = Panel(
            f"[dim]{preview}[/dim]",
            title=f"[cyan]{module_name} Detail Output[/cyan]",
            border_style="cyan",
            box=ROUNDED
        )
        console.print(panel)
    
    if error:
        console.print(f"   [bold red]❌ Error: {error}[/bold red]")

def print_final_answer(answer: str, run_id: str = None):
    """打印最终回答"""
    console.print("\n" + "="*80)
    console.print(f"🤖 [bold green]Assistant:[/bold green]\n\n{answer}")
    if run_id:
        console.print(f"\n[dim italic]Run ID: {run_id}[/dim italic]")
    console.print("="*80 + "\n")

class PipelineProgress:
    """流水线进度展示"""
    def __init__(self, query: str):
        self.query = query
        self.start_time = time.time()
        
    def __enter__(self):
        console.print(f"\n[bold yellow]🔍 Query:[/bold yellow] [bold white]{self.query}[/bold white]\n")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (time.time() - self.start_time) * 1000
        console.print(f"[dim]Total pipeline time: {duration:.2f}ms[/dim]")
