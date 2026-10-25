"""The keyboard fallback: a small always-on-top window that asks the question.

It runs as its own process (`python -m quackback.dialog`): the server's stdin and stdout belong to
MCP, and Tk wants a main thread of its own. The request arrives as JSON on stdin,
`{"question": str, "context": str | null, "options": [str], "close_after": float | null,
"parent_pid": int | null}`, and the answer leaves as JSON on stdout, `{"answer": str}`. Exit
status DISMISSED (10) means the window was closed unanswered.

Normally the server kills the window when the question times out. `close_after` and
`parent_pid` cover the server dying first: the window closes itself when either runs out.
"""

import json
import os
import sys

# Not 1: that is also what Python exits with on an uncaught exception (say, no display).
DISMISSED = 10
WRAP = 480
_PARENT_CHECK_MS = 1000


def main() -> int:
    request = json.load(sys.stdin)
    answer = run_dialog(
        request["question"],
        request.get("context"),
        request.get("options") or [],
        close_after=request.get("close_after"),
        parent_pid=request.get("parent_pid"),
    )
    if answer is None:
        return DISMISSED
    sys.stdout.write(json.dumps({"answer": answer}))
    return 0


def run_dialog(
    question: str,
    context: str | None,
    options: list[str],
    close_after: float | None = None,
    parent_pid: int | None = None,
) -> str | None:
    # Imported here so the server (which imports DISMISSED) never needs Tk itself.
    import tkinter as tk
    from tkinter import font, ttk

    _sharp_on_high_dpi()
    root = tk.Tk()
    root.title("quackback: your coding agent has a question")
    root.attributes("-topmost", True)
    root.resizable(False, False)
    answer: list[str] = []

    def submit(text: str) -> None:
        text = text.strip()
        if text:
            answer.append(text)
            root.destroy()

    frame = ttk.Frame(root, padding=16)
    frame.grid(sticky="nsew")
    if context:
        ttk.Label(frame, text=context, wraplength=WRAP, foreground="#555555").grid(sticky="w")
    heading = font.nametofont("TkDefaultFont").copy()
    size = heading.cget("size")  # negative means pixels (the X11 default): grow away from zero
    heading.configure(size=size + 2 if size > 0 else size - 2, weight="bold")
    ttk.Label(frame, text=question, wraplength=WRAP, font=heading).grid(sticky="w", pady=(4, 12))
    for option in options:
        button = ttk.Button(frame, text=option, command=lambda o=option: submit(o))
        button.grid(sticky="ew", pady=2)
    entry = ttk.Entry(frame, width=64)
    entry.grid(sticky="ew", pady=(12, 4))
    ttk.Button(frame, text="Answer", command=lambda: submit(entry.get())).grid(sticky="e")

    root.bind("<Return>", lambda _event: submit(entry.get()))
    root.bind("<Escape>", lambda _event: root.destroy())
    root.after(100, lambda: (root.lift(), root.focus_force(), entry.focus_set()))
    if close_after is not None:
        root.after(max(0, int(close_after * 1000)), root.destroy)
    if parent_pid is not None:

        def watch_parent() -> None:
            if process_alive(parent_pid):
                root.after(_PARENT_CHECK_MS, watch_parent)
            else:
                root.destroy()

        root.after(_PARENT_CHECK_MS, watch_parent)
    root.bell()
    root.mainloop()
    return answer[0] if answer else None


def process_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        synchronize, wait_timeout = 0x00100000, 0x102
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel32.OpenProcess(synchronize, False, pid)
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == wait_timeout
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # it exists, it is just not ours
    return True


def _sharp_on_high_dpi() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass


if __name__ == "__main__":
    sys.exit(main())
