def main():
    import subprocess
    import sys
    from pathlib import Path

    try:
        import streamlit  # noqa: F401
    except ImportError:
        sys.stderr.write(
            "\nError: streamlit is not installed.\n"
            "Install the GUI dependencies with:\n\n"
            "  pip install xlron[gui]\n\n"
            "or:\n\n"
            "  uv sync --extra gui\n\n"
        )
        sys.exit(1)

    app_path = str(Path(__file__).parent / "app.py")

    # ANSI colour codes
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"

    msg = f"""
{CYAN}{'=' * 60}{RESET}
{BOLD}  XLRON - Optical Network RL Training GUI{RESET}
{CYAN}{'=' * 60}{RESET}

  {GREEN}Streamlit is starting...{RESET}
  Open the URL shown below in your browser.

  {YELLOW}Remote server?{RESET} Use SSH port forwarding:
    ssh -L <port>:localhost:<port> user@remote-host
  Replace <port> with the port number shown by Streamlit.

  {CYAN}Docs:{RESET} https://micdoh.github.io/XLRON/
{CYAN}{'=' * 60}{RESET}
"""
    sys.stderr.write(msg)
    sys.stderr.flush()

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", app_path, "--server.headless", "true"]
    )
