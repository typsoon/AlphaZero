from pathlib import Path
from python.utils import PROJ_ROOT


def task_setup_service():
    """Install the inference server as a systemd user service."""
    home_dir = Path.home()
    config_systemd = home_dir / ".config" / "systemd" / "user"
    config_alphazero = home_dir / ".config" / "alphazero"
    models_dir = config_alphazero / "models"
    service_file = config_systemd / "alphazero-inference.service"
    env_file = config_alphazero / "inference.env"

    source_service = PROJ_ROOT / "inference_server" / "alphazero-inference.service"
    source_env = PROJ_ROOT / "inference_server" / "inference.env.example"

    return {
        "actions": [
            f"mkdir -p {config_systemd} {models_dir}",
            f"sed 's|{{REPO_DIR}}|{PROJ_ROOT}|g' {source_service} > {service_file}",
            f"test -f {env_file} || sed -e 's|{{REPO_DIR}}|{PROJ_ROOT}|g' -e 's|{{HOME_DIR}}|{home_dir}|g' {source_env} > {env_file}",
            "systemctl --user daemon-reload",
        ]
    }


def task_enable_service():
    """Enable the inference server systemd service to start on boot."""
    return {
        "actions": [
            "systemctl --user enable alphazero-inference.service",
        ],
        "task_dep": ["setup_service"],
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
        "task_dep": ["setup_service", "build"],
    }


def task_restart_service():
    """Restart the inference server systemd service to apply config changes."""
    check_cmd = "(systemctl --user is-active --quiet alphazero-inference.service || { echo '\\033[31mService failed to stay running! Logs:\\033[0m'; journalctl --user -xeu alphazero-inference.service -n 20; exit 1; })"
    return {
        "actions": [
            f"systemctl --user restart alphazero-inference.service && sleep 2 && {check_cmd}",
        ],
        "task_dep": ["setup_service", "build"],
    }


def task_stop_service():
    """Stop the inference server systemd service."""
    return {
        "actions": [
            "systemctl --user stop alphazero-inference.service",
        ]
    }
