# Maritime AI Navigation System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An advanced AI-powered maritime navigation system that leverages reinforcement learning and sophisticated control algorithms to optimize vessel navigation in complex marine environments.

## 🌊 Overview

The Maritime AI Navigation System is a comprehensive solution for autonomous vessel navigation that integrates:

- **Reinforcement Learning**: AI agents that learn optimal navigation strategies
- **Maneuvering Mathematical Group (MMG) Model**: Accurate ship dynamics simulation
- **Line of Sight (LOS) Guidance**: Precise waypoint following
- **Advanced Reward Systems**: Multi-objective optimization for safety and efficiency
- **Real-time Policy Evaluation**: Continuous learning and adaptation

## 🚀 Features

### Core Capabilities

- ✅ **Autonomous Navigation**: AI-driven pathfinding and obstacle avoidance
- ✅ **Real-time Environment Simulation**: Dynamic marine environment modeling
- ✅ **Waypoint Management**: Intelligent route planning and optimization
- ✅ **Safety Systems**: Collision avoidance and emergency protocols
- ✅ **Performance Analytics**: Comprehensive navigation metrics and reporting
- ✅ **Weather Integration**: Environmental factor consideration
- ✅ **Multi-vessel Coordination**: Fleet management capabilities

### Technical Features

- 🔧 **Modular Architecture**: Easy to extend and customize
- 🔧 **Real-time Processing**: Low-latency decision making
- 🔧 **Scalable Design**: Supports multiple vessel types and sizes
- 🔧 **Robust Error Handling**: Fault-tolerant navigation systems
- 🔧 **Configurable Parameters**: Adaptable to various operational requirements

## 📋 System Requirements

### Hardware Requirements

- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 2GB available space
- **GPU**: Optional (CUDA-compatible for accelerated training)

### Software Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux Ubuntu 18.04+
- **Dependencies**: Listed in requirements.txt

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/vishnu-thirumurugan/Maritime-AI-Navigation-System.git
cd Maritime-AI-Navigation-System
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv maritime_env
source maritime_env/bin/activate  # On Windows: maritime_env\Scripts\activate

# Using conda
conda create -n maritime_env python=3.8
conda activate maritime_env
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import numpy, matplotlib, scipy; print('Dependencies installed successfully')"
```

## 🚀 Quick Start

### Basic Navigation Example

```python
from Env import MaritimeEnvironment
from PolicyEvaluation import PolicyEvaluator
from waypoints import WaypointManager

# Initialize the maritime environment
env = MaritimeEnvironment()

# Set up waypoints
waypoint_manager = WaypointManager()
waypoints = waypoint_manager.generate_route(
    start=(0, 0),
    end=(100, 100),
    obstacles=[(50, 50, 10)]  # x, y, radius
)

# Initialize policy evaluator
evaluator = PolicyEvaluator(env)

# Run navigation simulation
results = evaluator.evaluate_policy(
    waypoints=waypoints,
    max_episodes=100
)

print(f"Navigation completed with {results['success_rate']}% success rate")
```

### Advanced Configuration

```python
# Custom environment configuration
env_config = {
    'vessel_type': 'cargo_ship',
    'weather_conditions': 'moderate',
    'current_strength': 0.5,
    'wind_speed': 15,
    'visibility': 'good'
}

env = MaritimeEnvironment(config=env_config)
```

## 📁 Project Structure

```
Maritime-AI-Navigation-System/
├── 📄 README.md                 # Project documentation
├── 📄 LICENSE                   # MIT License
├── 📄 requirements.txt          # Python dependencies
├── 🐍 Env.py                    # Environment simulation
├── 🐍 MMG.py                    # Maneuvering Mathematical Group model
├── 🐍 LOS.py                    # Line of Sight guidance system
├── 🐍 Reward.py                 # Reward function definitions
├── 🐍 PolicyEvaluation.py       # RL policy evaluation
├── 🐍 waypoints.py              # Waypoint management
├── 🐍 wp_analysis.py            # Waypoint analysis tools
└── 🐍 Results.py                # Results processing and visualization
```

## 🔧 Configuration

### Environment Configuration

Create a config.yaml file to customize system behavior:

```yaml
environment:
  simulation_step: 0.1
  max_episode_length: 1000
  boundary_limits: [-200, 200, -200, 200]

vessel:
  length: 100.0
  beam: 20.0
  draft: 8.0
  displacement: 15000.0

navigation:
  lookahead_distance: 50.0
  acceptance_radius: 10.0
  max_speed: 15.0

ai_training:
  learning_rate: 0.001
  batch_size: 32
  memory_size: 10000
  exploration_rate: 0.1
```

## 🧪 Testing

### Run Unit Tests

```bash
python -m pytest tests/ -v
```

### Run Integration Tests

```bash
python -m pytest tests/integration/ -v
```

### Performance Benchmarks

```bash
python tests/benchmark.py
```

## 📊 Usage Examples

### 1. Waypoint Navigation

```python
from waypoints import WaypointManager
from wp_analysis import WaypointAnalyzer

# Generate optimal route
wm = WaypointManager()
route = wm.optimize_route(
    waypoints=[(0,0), (50,30), (100,60), (150,90)],
    constraints={'max_turn_angle': 45, 'min_segment_length': 20}
)

# Analyze route efficiency
analyzer = WaypointAnalyzer()
metrics = analyzer.analyze_route(route)
print(f"Total distance: {metrics['total_distance']:.2f} nm")
print(f"Estimated time: {metrics['eta']:.2f} hours")
```

### 2. Policy Training

```python
from PolicyEvaluation import train_policy

# Train new navigation policy
training_config = {
    'episodes': 1000,
    'learning_rate': 0.001,
    'exploration_decay': 0.995,
    'target_success_rate': 0.95
}

trained_policy = train_policy(
    environment=env,
    config=training_config,
    save_path='models/navigation_policy.pkl'
)
```

### 3. Results Visualization

```python
from Results import ResultsVisualizer

# Visualize navigation results
visualizer = ResultsVisualizer()
visualizer.plot_trajectory(results)
visualizer.plot_performance_metrics(results)
visualizer.generate_report('navigation_report.html')
```

## 🗺️ Roadmap

### Version 2.0 (Q4 2025)

- [ ] Full autonomous operation
- [ ] International waters compliance
- [ ] Satellite communication support
- [ ] Predictive maintenance integration

## 🏆 Awards & Recognition

- 🥇 **All India Research Scholars meet 2024** - Recognized as best research work
- 🏆 **Published in ASME 2024** - International Conference on Ocean, Offshore and Arctic Engineering OMAE 2024


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Research Team**: Indian Institute of Technology, Madras
- **Special Thanks**: Dr. Suresh Rajendran from IIT Madras and Mr. Sivaraman Sivaraj from zf friedrichshafen

---
