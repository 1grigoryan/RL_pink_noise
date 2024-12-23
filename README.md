# Исследование розового шума в глубоком обучении с подкреплением

Этот репозиторий содержит реализации, эксперименты и результаты исследований **розового шума** в глубоком обучении с подкреплением (RL), вдохновленные статьей *"Pink Noise is All You Need: Colored Noise Exploration in Deep Reinforcement Learning"*, представленной на конференции ICLR 2023. Проект исследует влияние различных процессов цветного шума, с акцентом на **розовый шум**, в качестве шума действий для RL-агентов.

## Структура проекта

```plaintext
├── Experiments.ipynb        # Ноутбук для запуска и анализа экспериментов
├── HalfCheetah/             # Результаты и артефакты для среды HalfCheetah (пример ниже для DDPG-алгоритма)
│   ├── HalfCheetah-v5_DDPG.zip # Модель в zip-формате
│   ├── HalfCheetah-v5_DDPG_{noise_type}.mp4  # Видеоролики для различных шумов
│   ├── HalfCheetah-v5_DDPG_combined_plot.png   # Общий график производительности
│   ├── HalfCheetah-v5_SAC.zip
│   └── HalfCheetah-v5_TD3.zip
├── Hopper/                  # Результаты и артефакты для среды Hopper
├── Walker2d/                # Результаты и артефакты для среды Walker2d
├── LICENSE
├── README.md
├── mlc_notebook_readme.md   
├── ou_noise.py              # Реализация шума Орнштейна-Уленбека
├── pyproject.toml           # Зависимости Python-проекта
├── rl_experiment.py         # Скрипт для обучения RL-агентов с цветным шумом
└── set_env.py               # Скрипт настройки окружения
```

## Основные особенности

1. **Цветной шум в RL**: Реализация исследований с процессами цветного шума, включая белый шум, шум Орнштейна-Уленбека и розовый шум.
2. **Поддержка нескольких алгоритмов**: Демонстрация влияния шума на такие алгоритмы, как SAC, TD3 и DDPG.
3. **Разнообразие сред**: Эксперименты в средах HalfCheetah, Hopper и Walker2d из OpenAI Gym и DeepMind Control Suite.
4. **Визуализация**: Видеоролики и графики, иллюстрирующие влияние различных типов шума.

## Начало работы

### Требования

- Python 3.8+
- Среды Gym (например, Mujoco для задач с непрерывным управлением)
- Необходимые Python-пакеты: см. `pyproject.toml`

### Установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/1grigoryan/pink-noise-rl.git
   cd pink-noise-rl
   ```

2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

3. Настройте окружение:
   ```bash
   python set_env.py
   ```

### Запуск экспериментов

1. **Обучение RL-агентов**:
   Используйте `rl_experiment.py` для обучения агентов с различными типами шума:
   ```bash
   python rl_experiment.py --env HalfCheetah-v5 --algorithm SAC --noise pink
   ```

2. **Визуализация результатов**:
   Откройте `Experiments.ipynb` для анализа и визуализации производительности.

3. **Воспроизведение видео**:
   Проверьте видеоролики в папках сред (например, `HalfCheetah`), чтобы увидеть поведение агентов при различных условиях шума.

### Основные параметры экспериментов

- `--env`: Среда для обучения (например, `HalfCheetah-v5`)
- `--algorithm`: Алгоритм RL для использования (SAC, TD3, DDPG)
- `--noise`: Тип шума для исследования (white, pink, OU и др.)
- `--beta`: Параметр beta для цветного шума (например, 0.1, 1.0 для розового шума)

## Результаты

### Обзор
- **Розовый шум** обеспечивает баланс между локальным и глобальным исследованием, превосходя белый и шум Орнштейна-Уленбека в различных задачах.
- Графики и видеоролики демонстрируют улучшенное покрытие пространства состояний и более быстрое сходимость.

### Основные моменты
- **HalfCheetah**: Розовый шум последовательно достигает более высоких наград.
- **Hopper**: Сбалансированное исследование приводит к более плавным улучшениям политики.
- **Walker2d**: Розовый шум предотвращает преждевременную сходимость, наблюдаемую с белым шумом.

## Ссылки

- Статья: [Pink Noise is All You Need](https://arxiv.org/abs/2301.12345)
- Репозиторий: [martius-lab/pink-noise-rl](https://github.com/martius-lab/pink-noise-rl)

## Лицензия

Этот проект лицензирован под лицензией MIT. Подробности см. в файле LICENSE.




## Citing
If you use this code in your research, please cite our paper:
```bibtex
@inproceedings{eberhard-2023-pink,
  title = {Pink Noise Is All You Need: Colored Noise Exploration in Deep Reinforcement Learning},
  author = {Eberhard, Onno and Hollenstein, Jakob and Pinneri, Cristina and Martius, Georg},
  booktitle = {Proceedings of the Eleventh International Conference on Learning Representations (ICLR 2023)},
  month = may,
  year = {2023},
  url = {https://openreview.net/forum?id=hQ9V5QN27eS}
}
```

If there are any problems, or if you have a question, don't hesitate to open an issue here on GitHub.
>>>>>>> 317f308 (Hopper + HalfCheetah + Walker2d)
