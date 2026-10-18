## Установка

1. Создайте собственный репозиторий
2. Сделайте ```git@github.com:SPBUnited-Junior/StrategyTemplate.git``` 
3. Обновите upstream: ```git fetch upstream```
4. Подтяните изменения из upstream и ребазируйтесь на них: ```git rebase upstream/master```

## Подтянуть изменения
1. Чтобы подтянуть изменения, сделанные в upstream, выполните ```git pull upstream master```

---

# Описание репозитория

## 📦 Installation

Clone the repository with submodules and install dependencies:

```bash
git clone git@github.com:SPbUnited/strategy.git
cd strategy
make init
```

## ⚙️ Configuration

After the first run of the project, a `.env` file will be generated. This file contains all the main configuration parameters of the program, each of which is described in detail. You can modify the values in `.env` to adjust the behavior of the project without changing the code directly.

## ▶️ Running the Project

Для реального поля запустите PAcmaCS с протоколом ваших роботов, затем стратегию
в другом терминале:

```bash
cd ../PAcmaCS
make up-real CTRL=FB4  # для FB2: CTRL=FB2
```

```bash
cd StrategyTemplate_oxxy
make run-real
```

`run-real` задаёт `IS_SIMULATOR_USED=False` и `DEBUG_MODE=False`: начальное состояние
HALT, команды поступают от судьи. Эти значения также выставлены в шаблоне `.env`.
Цвет команды, номера вратарей и сторону поля задавайте в `.env`.
После изменения кода перезапустите PAcmaCS и ранее запущенный процесс стратегии.

Для симулятора используйте PAcmaCS `make up-sim` (или `make up-all` вместе с grSim)
и стратегию `make run-sim`. Судейские команды обрабатываются и в этом режиме.
`make run-sim-debug` включает принудительный RUN для отладки без судьи.

Свои KICKOFF/FREE_KICK начинаются пасом, PENALTY — ударом в створ; при подготовке
и чужом розыгрыше команда занимает позиции. При HALT/TIMEOUT движение, дриблер и
удар выключаются. В телеметрии STRATEGY показаны `game_state`, `active_team` и
`simulator`, чтобы сразу видеть выбранный режим.

Обычный запуск с параметрами из `.env`:

```bash
make run
```
or 
```bash
python3 main.py
```

## 🔍 Static Analysis

Check the entire project:

```bash
make syntax
```

Check all Python files in the `bridge/strategy/` directory:

```bash
make syntax_strategy
```


## 🛠️ Auto Formatting

Install [pre-commit](https://pre-commit.com/) and run auto-formatting:

```bash
make auto_format
```

Pre-commit hooks will automatically:

* Remove unused imports and variables with `autoflake`
* Sort imports according to `isort`
* Format code with `black`
* Check code style with `flake8`
* Perform static type checking with `mypy`


## 🤖 Recommended Program: PAcmaCS

For full control of the robots, data visualization, and telemetry topics, it is recommended to use [PAcmaCS](https://github.com/SPbUnited/PAcmaCS). This framework is designed for developing strategies in RoboCup SSL.

PAcmaCS uses ZeroMQ for inter-module communication, automates dependency installation, builds, and runs the services. With Serviz, you can visualize the game field with objects, control robots via keyboard, and configure display layers.

Follow the [PAcmaCS README](https://github.com/SPbUnited/PAcmaCS) for installation and setup. Once running, you can access the interface at [http://localhost:8000](http://localhost:8000).
