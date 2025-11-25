import sys
import time
import json
import threading
import numpy as np
import torch
import psutil
from datetime import datetime
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QTextEdit, QFrame, QMessageBox,
    QTabWidget, QGroupBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QRunnable, QThreadPool
from PyQt6.QtGui import QFont, QColor, QPalette, QPainter, QLinearGradient, QPen


# ================================
# 🔧 Configuration & Constants
# ================================
REFERENCE = {
    'cpu_single': 2100,   # ~Geekbench 6 Single-Core (i7-12700K)
    'cpu_multi': 14500,   # ~Geekbench 6 Multi-Core
    'gpu_score': 22000,   # Synthetic: RTX 3060 (FP32 GEMM+FFT mix)
}

# Worker Signals
class Signals(QObject):
    progress = pyqtSignal(int)
    result = pyqtSignal(object)
    message = pyqtSignal(str)

# ================================
# 🧠 Benchmark Engines
# ================================

class CPUBenchmark(QRunnable):
    def __init__(self):
        super().__init__()
        self.signals = Signals()
        self.is_cancelled = False

    def mandelbrot_single(self, max_iter=500):
        """Heavy FP workload — simulates single-core rendering (Cinebench-style)"""
        xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
        width = height = 800
        x = np.linspace(xmin, xmax, width)
        y = np.linspace(ymin, ymax, height)
        C = x[np.newaxis, :] + 1j * y[:, np.newaxis]
        Z = np.zeros_like(C)
        M = np.full(C.shape, True, dtype=bool)
        for _ in range(max_iter):
            Z[M] = Z[M] ** 2 + C[M]
            diverged = np.abs(Z) > 2
            M[diverged] = False
        return np.count_nonzero(M)

    def prime_search(self, target_count=3000):
        """Integer-heavy — mimics Geekbench CPU tests"""
        def is_prime(n):
            if n < 2:
                return False
            if n % 2 == 0:
                return n == 2
            r = int(n ** 0.5)
            f = 3
            while f <= r:
                if n % f == 0:
                    return False
                f += 2
            return True

        count = 0
        num = 2
        while count < target_count and not self.is_cancelled:
            if is_prime(num):
                count += 1
            num += 1
        return count

    def run_multi(self, thread_id, reps=8):
        total = 0
        for i in range(reps):
            if self.is_cancelled:
                break
            total += self.mandelbrot_single(max_iter=300)
            self.signals.progress.emit(int((thread_id * 100 + i * 10) / (psutil.cpu_count() * reps)))
        return total

    def run(self):
        try:
            # ——— SINGLE-CORE ———
            self.signals.message.emit("Running CPU Single-Core Test (Mandelbrot + Prime)...")
            start_single = time.time()
            _ = self.mandelbrot_single(600)
            _ = self.prime_search(2500)
            single_duration = time.time() - start_single
            single_score = (REFERENCE['cpu_single'] / single_duration) * 1.0  # Scale to reference

            # ——— MULTI-CORE ———
            self.signals.message.emit("Running CPU Multi-Core Stress (Parallel Rendering)...")
            threads = []
            pool = QThreadPool.globalInstance()
            num_threads = psutil.cpu_count(logical=True)

            # Launch workers
            results = []
            lock = threading.Lock()

            def worker_wrapper(tid):
                res = self.run_multi(tid, reps=6)
                with lock:
                    results.append(res)

            start_multi = time.time()
            for i in range(num_threads):
                t = threading.Thread(target=worker_wrapper, args=(i,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            multi_duration = time.time() - start_multi
            multi_score = (REFERENCE['cpu_multi'] / multi_duration) * 1.0

            # Finalize
            result = {
                'single_core': round(single_score, 1),
                'multi_core': round(multi_score, 1),
                'threads_used': num_threads,
                'single_duration': round(single_duration, 2),
                'multi_duration': round(multi_duration, 2)
            }
            self.signals.result.emit(result)

        except Exception as e:
            self.signals.result.emit({'error': str(e)})


class GPUBenchmark(QRunnable):
    def __init__(self):
        super().__init__()
        self.signals = Signals()
        self.is_cuda = torch.cuda.is_available()

    def run(self):
        try:
            if not self.is_cuda:
                self.signals.result.emit({
                    'score': 0,
                    'status': 'No CUDA GPU detected',
                    'details': {}
                })
                return

            self.signals.message.emit("Warming up GPU...")
            torch.cuda.empty_cache()
            a = torch.randn(100, 100, device='cuda')
            b = torch.randn(100, 100, device='cuda')
            _ = torch.mm(a, b)
            torch.cuda.synchronize()

            # ——— GEMM STRESS ———
            self.signals.message.emit("GPU Test 1/2: Matrix Multiplication (FP32)...")
            size = 3072
            a = torch.randn(size, size, device='cuda', dtype=torch.float32)
            b = torch.randn(size, size, device='cuda', dtype=torch.float32)
            start = time.time()
            for i in range(12):
                c = torch.mm(a, b)
                torch.cuda.synchronize()
                self.signals.progress.emit(int((i + 1) / 12 * 50))
            gemm_time = time.time() - start

            # ——— FFT STRESS ———
            self.signals.message.emit("GPU Test 2/2: Large FFTs...")
            n = 2 ** 21
            x = torch.randn(n, dtype=torch.complex64, device='cuda')
            start = time.time()
            for i in range(40):
                y = torch.fft.fft(x)
                torch.cuda.synchronize()
                self.signals.progress.emit(50 + int((i + 1) / 40 * 50))
            fft_time = time.time() - start

            total_time = gemm_time + fft_time
            raw_score = (1 / total_time) * 1e6
            normalized_score = (raw_score / REFERENCE['gpu_score']) * 1000

            result = {
                'score': round(normalized_score, 1),
                'status': 'Success',
                'details': {
                    'gemm_time_s': round(gemm_time, 2),
                    'fft_time_s': round(fft_time, 2),
                    'gpu_name': torch.cuda.get_device_name(0),
                    'cuda_ver': torch.version.cuda
                }
            }
            self.signals.result.emit(result)

        except Exception as e:
            self.signals.result.emit({
                'score': 0,
                'status': f'Error: {str(e)}',
                'details': {}
            })


# ================================
# 🎨 Custom Widgets
# ================================

class AnimatedProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444;
                border-radius: 4px;
                text-align: center;
                color: white;
                background: #2a2a2a;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00b0ff, stop:1 #0077ff
                );
                border-radius: 3px;
            }
        """)
        self.setValue(0)


class ScoreDisplay(QFrame):
    def __init__(self, title="Score", value="—", unit="pts", color="#00b0ff"):
        super().__init__()
        self.setFixedHeight(100)
        self.setStyleSheet(f"""
            QFrame {{
                background: #1e1e1e;
                border-radius: 8px;
                border: 1px solid #333;
            }}
        """)
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 14px; color: #aaa;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.value_label = QLabel(str(value))
        self.value_label.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {color};")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        unit_label = QLabel(unit)
        unit_label.setStyleSheet("font-size: 12px; color: #888; padding-left: 2px;")
        unit_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(unit_label)
        self.setLayout(layout)

    def set_value(self, value, color=None):
        self.value_label.setText(str(value))
        if color:
            self.value_label.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {color};")


class CyberBenchUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CyberBench Pro v1.2")
        self.setFixedSize(720, 580)
        self.setStyleSheet("""
            QMainWindow { background: #121212; }
            QLabel { color: #e0e0e0; }
            QPushButton {
                background: #2a2a2a;
                color: white;
                border: none;
                padding: 12px 20px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover { background: #3a3a3a; }
            QPushButton:disabled { background: #333; color: #777; }
            QTextEdit {
                background: #1a1a1a;
                color: #ccc;
                border: 1px solid #333;
                border-radius: 4px;
            }
        """)

        # Central Widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # Header
        header = QLabel("⚡ CyberBench")
        header.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        header.setStyleSheet("color: #00b0ff;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header)

        subheader = QLabel("Professional System Benchmark Suite")
        subheader.setFont(QFont("Segoe UI", 11))
        subheader.setStyleSheet("color: #888;")
        subheader.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(subheader)

        # System Info Bar
        sys_info = QGroupBox("System Overview")
        sys_layout = QHBoxLayout()
        cpu_threads = psutil.cpu_count(logical=True)
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
        sys_layout.addWidget(QLabel(f"💻 CPU: {psutil.cpu_count(logical=False)}C/{cpu_threads}T"))
        sys_layout.addWidget(QLabel(f"🧠 RAM: {ram_gb} GB"))
        sys_layout.addWidget(QLabel(f"🎮 GPU: {gpu_name}"))
        sys_info.setLayout(sys_layout)
        main_layout.addWidget(sys_info)

        # Tab Widget
        tabs = QTabWidget()
        tabs.setStyleSheet("QTabBar::tab { height: 32px; }")
        main_layout.addWidget(tabs)

        # ——— CPU TAB ———
        cpu_tab = QWidget()
        cpu_layout = QVBoxLayout(cpu_tab)
        cpu_layout.setContentsMargins(15, 15, 15, 15)

        cpu_title = QLabel("Central Processing Unit (CPU) Benchmark")
        cpu_title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        cpu_layout.addWidget(cpu_title)

        # Score Row
        score_row = QHBoxLayout()
        self.cpu_single_display = ScoreDisplay("Single-Core", "—", "pts", "#4caf50")
        self.cpu_multi_display = ScoreDisplay("Multi-Core", "—", "pts", "#2196f3")
        score_row.addWidget(self.cpu_single_display)
        score_row.addWidget(self.cpu_multi_display)
        cpu_layout.addLayout(score_row)

        # Progress & Button
        self.cpu_progress = AnimatedProgressBar()
        cpu_layout.addWidget(self.cpu_progress)

        self.cpu_btn = QPushButton("▶️ Run CPU Benchmark")
        self.cpu_btn.clicked.connect(self.run_cpu_benchmark)
        cpu_layout.addWidget(self.cpu_btn)

        # Log
        self.cpu_log = QTextEdit()
        self.cpu_log.setReadOnly(True)
        self.cpu_log.setMaximumHeight(100)
        cpu_layout.addWidget(self.cpu_log)

        tabs.addTab(cpu_tab, "💻 CPU")

        # ——— GPU TAB ———
        gpu_tab = QWidget()
        gpu_layout = QVBoxLayout(gpu_tab)
        gpu_layout.setContentsMargins(15, 15, 15, 15)

        gpu_title = QLabel("Graphics Processing Unit (GPU) Benchmark")
        gpu_title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        gpu_layout.addWidget(gpu_title)

        self.gpu_score_display = ScoreDisplay("GPU Compute Score", "—", "pts", "#ff5722")
        gpu_layout.addWidget(self.gpu_score_display)

        self.gpu_progress = AnimatedProgressBar()
        gpu_layout.addWidget(self.gpu_progress)

        self.gpu_btn = QPushButton("▶️ Run GPU Benchmark")
        self.gpu_btn.setEnabled(torch.cuda.is_available())
        if not torch.cuda.is_available():
            self.gpu_btn.setText("GPU Benchmark (CUDA Required)")
        self.gpu_btn.clicked.connect(self.run_gpu_benchmark)
        gpu_layout.addWidget(self.gpu_btn)

        self.gpu_log = QTextEdit()
        self.gpu_log.setReadOnly(True)
        self.gpu_log.setMaximumHeight(100)
        gpu_layout.addWidget(self.gpu_log)

        tabs.addTab(gpu_tab, "🎮 GPU")

        # ——— FOOTER ———
        footer = QLabel("© 2025 CyberBench Labs | Inspired by Cinebench & Geekbench")
        footer.setFont(QFont("Segoe UI", 9))
        footer.setStyleSheet("color: #666;")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(footer)

        # Thread pool
        self.thread_pool = QThreadPool()
        self.cpu_worker = None
        self.gpu_worker = None

    def log_cpu(self, msg):
        self.cpu_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        self.cpu_log.verticalScrollBar().setValue(self.cpu_log.verticalScrollBar().maximum())

    def log_gpu(self, msg):
        self.gpu_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        self.gpu_log.verticalScrollBar().setValue(self.gpu_log.verticalScrollBar().maximum())

    def run_cpu_benchmark(self):
        self.cpu_btn.setEnabled(False)
        self.cpu_progress.setValue(0)
        self.cpu_single_display.set_value("—")
        self.cpu_multi_display.set_value("—")
        self.cpu_log.clear()
        self.log_cpu("Initializing CPU benchmark...")

        self.cpu_worker = CPUBenchmark()
        self.cpu_worker.signals.progress.connect(self.cpu_progress.setValue)
        self.cpu_worker.signals.message.connect(self.log_cpu)
        self.cpu_worker.signals.result.connect(self.on_cpu_finished)
        self.thread_pool.start(self.cpu_worker)

    def on_cpu_finished(self, result):
        self.cpu_btn.setEnabled(True)
        if 'error' in result:
            QMessageBox.critical(self, "CPU Benchmark Error", f"Failed: {result['error']}")
            return

        self.cpu_single_display.set_value(result['single_core'], "#4caf50")
        self.cpu_multi_display.set_value(result['multi_core'], "#2196f3")
        self.log_cpu(f"✅ Completed! Single: {result['single_core']} | Multi: {result['multi_core']}")

        # Save result
        result.update({'type': 'cpu', 'timestamp': datetime.now().isoformat()})
        self.save_result(result)

    def run_gpu_benchmark(self):
        if not torch.cuda.is_available():
            QMessageBox.warning(self, "GPU Unavailable", "CUDA-capable GPU not detected.")
            return

        self.gpu_btn.setEnabled(False)
        self.gpu_progress.setValue(0)
        self.gpu_score_display.set_value("—")
        self.gpu_log.clear()
        self.log_gpu("Initializing GPU benchmark...")

        self.gpu_worker = GPUBenchmark()
        self.gpu_worker.signals.progress.connect(self.gpu_progress.setValue)
        self.gpu_worker.signals.message.connect(self.log_gpu)
        self.gpu_worker.signals.result.connect(self.on_gpu_finished)
        self.thread_pool.start(self.gpu_worker)

    def on_gpu_finished(self, result):
        self.gpu_btn.setEnabled(True)
        score = result.get('score', 0)
        status = result.get('status', 'Unknown')
        self.gpu_score_display.set_value(score if score > 0 else "—", "#ff5722" if score > 0 else "#f44336")
        self.log_gpu(f"✅ {status} | Score: {score}")

        # Save
        result.update({'type': 'gpu', 'timestamp': datetime.now().isoformat()})
        self.save_result(result)

    def save_result(self, data):
        Path("results").mkdir(exist_ok=True)
        fname = f"results/cyberbench_{data['type']}_{int(time.time())}.json"
        with open(fname, 'w') as f:
            json.dump(data, f, indent=2)


# ================================
# 🚀 Launch App
# ================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = CyberBenchUI()
    window.show()
    sys.exit(app.exec())