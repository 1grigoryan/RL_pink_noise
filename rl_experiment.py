import warnings
warnings.filterwarnings("ignore")

import os
import time
import json
import numpy as np
import torch
import zipfile
import gymnasium as gym
from dm_control import suite
from stable_baselines3 import SAC, TD3, DDPG

from pink import PinkNoiseDist, ColoredNoiseDist
from ou_noise import OrnsteinUhlenbeckNoise
import optuna


class RLModel:
    def __init__(
        self,
        env_name="Hopper-v5",
        agent_cls=SAC,
        total_timesteps=50_000,
        seed=0,
        noise_type="white",
        beta=None
    ):
        """
        Параметры:
         - env_name: Gym или dm_control среда
         - agent_cls: класс алгоритма (SAC, TD3, DDPG и т.д.)
         - total_timesteps: сколько шагов обучения по умолчанию
         - seed: random seed
         - noise_type: "white", "ou" или "colored"
         - beta: параметр, если noise_type == "colored"
        """
        self.env_name = env_name
        self.agent_cls = agent_cls
        self.total_timesteps = total_timesteps
        self.seed = seed
        self.noise_type = noise_type
        self.beta = beta

        # «Тег» для называния файлов/папок
        self.model_tag = f"{env_name}_{agent_cls.__name__}"

        self.env = None
        self.action_dim = None
        self.seq_len = None
        self.agent = None
        self.ou_noise = None

        self._setup_environment()
        self._initialize_agent()

    def _setup_environment(self):
        """
        Создаём обычную Gym/DM Control среду без SubprocVecEnv
        (во избежание ошибок сериализации).
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if "dm_control" in self.env_name:
            # DM Control
            domain, task = self.env_name.split('-')
            dm_env = suite.load(domain_name=domain, task_name=task)
            self.env = dm_env
            self.action_dim = dm_env.action_spec().shape[-1]
            self.seq_len = 1000
        else:
            # Gymnasium
            env = gym.make(self.env_name)
            self.env = env
            self.action_dim = env.action_space.shape[-1]
            self.seq_len = getattr(env, "_max_episode_steps", 1000)

    def _initialize_agent(self, learning_rate=3e-4, batch_size=256, gamma=0.99):
        """Создаём SB3-агента (SAC/TD3/DDPG) и подключаем шум (если нужно)."""
        self.agent = self.agent_cls(
            "MlpPolicy",
            self.env,
            seed=self.seed,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gamma=gamma,
        )
        # Если требуется шум
        if self.noise_type == "colored" and self.beta is not None:
            self.agent.actor.action_dist = ColoredNoiseDist(
                self.beta, self.seq_len, self.action_dim,
                rng=np.random.default_rng(self.seed)
            )
        elif self.noise_type == "ou":
            self.ou_noise = OrnsteinUhlenbeckNoise(size=self.action_dim, seed=self.seed)

    def train(self, total_timesteps=None, hyperparameters=None):
        """
        Обучаем агента. Stable-Baselines3>=1.7.0 позволяет progress_bar=True для наглядности.
        """
        if total_timesteps is None:
            total_timesteps = self.total_timesteps
        if hyperparameters:
            self._initialize_agent(**hyperparameters)

        start_time = time.time()
        self.agent.learn(total_timesteps=total_timesteps, progress_bar=True)
        elapsed_time = time.time() - start_time
        print(f"[TRAIN] {self.noise_type} noise - {total_timesteps} steps - {elapsed_time:.2f}s")

    def evaluate(self, num_episodes=5, bootstrap_repeats=1):
        """
        Оцениваем модель: несколько эпизодов, собираем суммарные награды.
        """
        all_results = []
        for _ in range(bootstrap_repeats):
            results = []
            for _ in range(num_episodes):
                if "dm_control" in self.env_name:
                    # DM Control
                    timestep = self.env.reset()
                    total_reward = 0
                    done = False
                    steps = 0
                    while not done and steps < 10000:
                        action, _ = self.agent.predict(timestep.observation, deterministic=True)
                        if self.noise_type == "ou":
                            action += self.ou_noise.sample()
                        timestep = self.env.step(action)
                        total_reward += float(timestep.reward or 0)
                        done = timestep.last()
                        steps += 1
                    results.append(total_reward)
                else:
                    # Gym
                    obs, info = self.env.reset(seed=self.seed + 999)
                    total_reward = 0
                    done = False
                    steps = 0
                    while not done and steps < 10000:
                        action, _ = self.agent.predict(obs, deterministic=True)
                        if self.noise_type == "ou":
                            action += self.ou_noise.sample()
                        obs, reward, terminated, truncated, info = self.env.step(action)
                        total_reward += reward
                        done = terminated or truncated
                        steps += 1
                    results.append(total_reward)
            all_results.extend(results)
        return all_results

    def optimize_hyperparameters(self, n_trials=5, train_timesteps=1000):
        """
        Оптимизация гиперпараметров с Optuna. n_jobs=1 -> без параллели.
        """
        def objective(trial):
            lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
            gamma = trial.suggest_float("gamma", 0.9, 0.999)

            # Переинициализируем
            self._initialize_agent(
                learning_rate=lr,
                batch_size=batch_size,
                gamma=gamma
            )
            # Короткое обучение
            self.agent.learn(total_timesteps=train_timesteps, progress_bar=True)
            # Быстрая оценка
            rewards = self.evaluate(num_episodes=3, bootstrap_repeats=1)
            return np.mean(rewards)

        import optuna
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, n_jobs=1)

        print("[OPTUNA] Best hyperparameters:", study.best_params)
        return study.best_params

    def save_zip_model(self, save_dir="./results"):
        """
        Сохраняем всё в один .zip:
          1) Сохраняем SB3-агента (policy) через self.agent.save(...)
          2) Пишем JSON c метаданными (env_name, noise_type, seed, beta, ...),
             добавляя его в тот же zip.
        """
        import json
        import zipfile

        # Папка для <env>_<alg>:
        target_folder = os.path.join(save_dir, self.model_tag)
        os.makedirs(target_folder, exist_ok=True)

        # Основное имя файла: <env>_<alg>.zip
        zip_path = os.path.join(target_folder, f"{self.model_tag}.zip")

        # 1) Сохраняем агента SB3 (policy и т.д.) в .zip
        #    Это создаст zip-файл (если он не существует) или перезапишет.
        self.agent.save(zip_path)
        print(f"[SAVE] SB3-модель сохранена в: {zip_path}")

        # 2) Теперь добавим наш JSON (метаданные) внутрь того же zip
        metadata = {
            "env_name": self.env_name,
            "noise_type": self.noise_type,
            "seed": self.seed,
            "beta": self.beta,
            "total_timesteps": self.total_timesteps,
            "agent_cls": self.agent_cls.__name__
        }

        # Откроем zip в режиме append и добавим "rl_model_metadata.json"
        with zipfile.ZipFile(zip_path, mode="a") as archive:
            # Превратим словарь metadata в json-строку
            json_str = json.dumps(metadata, indent=2)
            archive.writestr("rl_model_metadata.json", json_str)

        print(f"[ZIP] Метаданные добавлены в {zip_path}")

    def generate_videos(self, indices, video_folder="./videos", video_length=300):
        """
        Генерация видео для шумов: {noise_label: sb3_model}.
        Используем DummyVecEnv + VecVideoRecorder. (Не для dm_control).
        """
        if "dm_control" in self.env_name:
            print("[VIDEO] dm_control не поддерживается VecVideoRecorder. Пропускаем.")
            return

        os.makedirs(video_folder, exist_ok=True)

        from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
        import gymnasium as gym

        for noise_label, sb3_model in indices.items():
            print(f"[VIDEO] Генерация для шума: {noise_label}")

            def make_env():
                return gym.make(self.env_name, render_mode="rgb_array")

            vec_env = DummyVecEnv([make_env])

            prefix = f"{self.model_tag}_{noise_label}"  # напр. "Hopper-v5_SAC_white"
            vec_env = VecVideoRecorder(
                vec_env,
                video_folder,
                record_video_trigger=lambda step: step == 0,
                video_length=video_length,
                name_prefix=prefix
            )

            obs = vec_env.reset()
            for step_i in range(video_length):
                action, _ = sb3_model.predict(obs, deterministic=True)
                obs, _, dones, _ = vec_env.step(action)
                if dones[0]:
                    break

            vec_env.close()
            mp4_name = f"{prefix}-step-0-to-step-{video_length}.mp4"
            print(f"[VIDEO] Сохранено: {os.path.join(video_folder, mp4_name)}")


