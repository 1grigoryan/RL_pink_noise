import subprocess
import os

# --- 1. Установка системных библиотек (APT) ---
subprocess.run("sudo apt-get update -y > /dev/null 2>&1", shell=True, check=True)
subprocess.run(
    "sudo apt-get install -y "
    "python3.9-dev python3-dev build-essential swig libosmesa6-dev libegl1-mesa-dev "
    "libgles2-mesa-dev libglfw3 libgl1-mesa-glx patchelf ffmpeg > /dev/null 2>&1",
    shell=True,
    check=True
)

# --- 2. Установка/обновление необходимых Python-пакетов ---
subprocess.run("pip install --upgrade pip setuptools wheel --quiet", shell=True, check=True)
subprocess.run("pip uninstall mujoco_py gym --yes --quiet", shell=True, check=True)

# Устанавливаем пакеты (без acme).
# gymnasium[box2d] уже включает Box2D-поддержку.
subprocess.run(
    "pip install stable-baselines3[extra] gymnasium[box2d] gymnasium[other] moviepy gym box2d-py swig "
    "colorednoise plotly seaborn tqdm torch optuna dm_control mujoco imageio "
    "sb3-contrib --quiet",
    shell=True,
    check=True
)

# Дополнительно обновляем PyOpenGL
subprocess.run("pip install --upgrade PyOpenGL PyOpenGL-accelerate --quiet", shell=True, check=True)

# --- 3. Установка MuJoCo 2.1.0 ---
os.makedirs(os.path.expanduser("~/.mujoco"), exist_ok=True)
subprocess.run("wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz", shell=True, check=True)
subprocess.run("tar -xzf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco > /dev/null 2>&1", shell=True, check=True)
subprocess.run("rm mujoco210-linux-x86_64.tar.gz", shell=True, check=True)

os.environ["MUJOCO_PY_MUJOCO_PATH"] = os.path.expanduser("~/.mujoco/mujoco210")
os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":" + os.path.expanduser("~/.mujoco/mujoco210/bin")

# --- 4. Клонируем репозиторий pink-noise-rl ---
subprocess.run("git clone https://github.com/martius-lab/pink-noise-rl.git tmp_repo", shell=True)

# Переносим все файлы (включая скрытые)
subprocess.run("mv tmp_repo/* tmp_repo/.[!.]* . 2>/dev/null || echo 'Некоторые файлы уже существуют, пропускаем...'", shell=True)
subprocess.run("rm -rf tmp_repo", shell=True)

print("Установка завершена.")