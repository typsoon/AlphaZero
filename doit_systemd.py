import os

def task_setup_service():
    """Install the inference server as a systemd user service."""
    repo_dir = os.getcwd()
    home_dir = os.path.expanduser("~")
    return {
        "actions": [
            f"mkdir -p {home_dir}/.config/systemd/user {home_dir}/.config/alphazero/models",
            f"sed 's|{{REPO_DIR}}|{repo_dir}|g' inference_server/alphazero-inference.service > {home_dir}/.config/systemd/user/alphazero-inference.service",
            f"test -f {home_dir}/.config/alphazero/inference.env || sed -e 's|{{REPO_DIR}}|{repo_dir}|g' -e 's|{{HOME_DIR}}|{home_dir}|g' inference_server/inference.env.example > {home_dir}/.config/alphazero/inference.env",
            "systemctl --user daemon-reload",
        ]
    }

def task_enable_service():
    """Enable the inference server systemd service to start on boot."""
    return {
        "actions": [
            "systemctl --user enable alphazero-inference.service",
        ],
        "task_dep": ["setup_service"]
    }

def task_disable_service():
    """Disable the inference server systemd service."""
    return {
        "actions": [
            "systemctl --user disable alphazero-inference.service",
        ]
    }

def task_start_service():
    """Start the inference server systemd service immediately."""
    check_cmd = "(systemctl --user is-active --quiet alphazero-inference.service || { echo '\\033[31mService failed to stay running! Logs:\\033[0m'; journalctl --user -xeu alphazero-inference.service -n 20; exit 1; })"
    return {
        "actions": [
            f"systemctl --user is-active --quiet alphazero-inference.service && echo '\\033[33mService is already running! Use `doit restart_service` to apply changes.\\033[0m' || (systemctl --user start alphazero-inference.service && sleep 2 && {check_cmd})",
        ],
        "task_dep": ["setup_service", "build"]
    }

def task_restart_service():
    """Restart the inference server systemd service to apply config changes."""
    check_cmd = "(systemctl --user is-active --quiet alphazero-inference.service || { echo '\\033[31mService failed to stay running! Logs:\\033[0m'; journalctl --user -xeu alphazero-inference.service -n 20; exit 1; })"
    return {
        "actions": [
            f"systemctl --user restart alphazero-inference.service && sleep 2 && {check_cmd}",
        ],
        "task_dep": ["setup_service", "build"]
    }

def task_stop_service():
    """Stop the inference server systemd service."""
    return {
        "actions": [
            "systemctl --user stop alphazero-inference.service",
        ]
    }
