from pathlib import Path
from python.utils import PROJ_ROOT, BUILD_DIR
import platformdirs


def task_install_service_binary():
    """Copy the inference server binary to ~/.local/bin."""
    home_dir = Path.home()
    bin_dir = home_dir / ".local" / "bin"
    installed_bin = bin_dir / "alphazero-inference-server"
    compiled_bin = BUILD_DIR / "inference_server" / "inference_server"
    return {
        "actions": [
            f"mkdir -p {bin_dir}",
            f"cp {compiled_bin} {installed_bin}",
            f"chmod +x {installed_bin}",
        ],
        "task_dep": ["build"],
    }


def task_setup_service():
    """Install the inference server as a systemd user service."""
    home_dir = Path.home()
    config_systemd = home_dir / ".config" / "systemd" / "user"
    config_alphazero = Path(platformdirs.user_config_dir("alphazero"))
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
        ],
        "task_dep": ["install_service_binary"],
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
        "task_dep": ["setup_service", "install_service_binary"],
    }


def task_restart_service():
    """Restart the inference server systemd service to apply config changes."""
    check_cmd = "(systemctl --user is-active --quiet alphazero-inference.service || { echo '\\033[31mService failed to stay running! Logs:\\033[0m'; journalctl --user -xeu alphazero-inference.service -n 20; exit 1; })"
    return {
        "actions": [
            f"systemctl --user restart alphazero-inference.service && sleep 2 && {check_cmd}",
        ],
        "task_dep": ["setup_service", "install_service_binary"],
    }


def task_stop_service():
    """Stop the inference server systemd service."""
    return {
        "actions": [
            "systemctl --user stop alphazero-inference.service",
        ]
    }
