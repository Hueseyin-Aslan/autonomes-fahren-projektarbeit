# Autonomes Fahren im Testfeld Digitalisierung

Projektarbeit — Training eines Reinforcement-Learning-Agenten (PPO) zur Hindernisausweichung mit dem TurtleBot3 Burger in ROS 2 / Gazebo.

**Autor:** Huseyin Aslan
**Betreuung:** Dr. Eric Wagner · Christoph Karls, M.Sc. · Prof. Dr. Steffen Knapp
**Abgabe:** 26.08.2026

Die vollständige Dokumentation (PDF/DOCX) beschreibt Vorgehen, Entwicklungsprozess und Ergebnisse ausführlich und liegt separat bei der Abgabe bei.

---

## Kurzüberblick

Ein simulierter TurtleBot3 Burger lernt in einer eigens gebauten Gazebo-Welt, Hindernissen auszuweichen und eine Ziellinie zu erreichen. Trainiert wird mit Proximal Policy Optimization (PPO, via Stable-Baselines3) und einem Curriculum-Learning-Ansatz: Das Ziel beginnt nah und rückt automatisch weiter, sobald der Agent eine Stufe zuverlässig meistert.

**Ergebnis (finales Modell, ≈850.000 Trainingsschritte):** bis zu 88 % Erfolgsquote auf der jeweils aktuellen Curriculum-Stufe, unter 10 % Kollisionsrate. Die volle 14-Meter-Strecke wird noch nicht durchgängig zuverlässig gemeistert (siehe Dokumentation, Kapitel „Ergebnisse" und „Ausblick").

---

## Projektstruktur

```
├── scripts/
│   ├── robot_env.py              # Gym-Environment: LiDAR, Odometrie, Reward, Reset
│   ├── train_ppo.py               # Frisches Training starten (inkl. Curriculum Learning)
│   ├── continue_training.py       # Bestehendes Modell fortsetzen
│   ├── test_env.py                # Environment mit Zufallsaktionen testen (Verifikation)
│   ├── demo_run.py                # Trainiertes Modell deterministisch vorführen (Live-Demo)
│   └── real_robot_inference.py    # Inferenz auf echter Hardware — UNGETESTET, siehe Hinweis unten
├── worlds/
│   └── projektwelt.sdf            # Gazebo-Teststrecke (Straße, Hindernisse, Start/Ziel)
├── launch/
│   └── projektwelt.launch.py      # Startet Simulation + ROS2-Bridges + Roboter-Spawn
├── training/
│   ├── models/                    # Trainierte Modelle (finaler Checkpoint enthalten)
│   └── logs/                      # TensorBoard-Protokolle aller Trainingsläufe
├── package.xml
└── CMakeLists.txt
```

---

## Voraussetzungen

- Ubuntu 24.04 (getestet unter WSL2)
- ROS 2 Jazzy
- Gazebo Harmonic
- Python 3.12, `stable-baselines3`, `gymnasium`, `tensorboard`

```bash
pip install stable-baselines3 gymnasium tensorboard tqdm rich --break-system-packages
```

---

## Verwendung

**1. Paket bauen**
```bash
cd ~/projektarbeit/ros2_ws
colcon build --packages-select autonomes_fahren_gazebo
source install/setup.bash
```

**2. Simulation starten** (eigenes Terminal, offen lassen)
```bash
export TURTLEBOT3_MODEL=burger
ros2 launch autonomes_fahren_gazebo projektwelt.launch.py
```

**3. Environment testen** (zweites Terminal)
```bash
python3 scripts/test_env.py
```

**4. Training starten**
```bash
python3 scripts/train_ppo.py          # frisch, mit Curriculum Learning
python3 scripts/continue_training.py  # setzt neuesten Checkpoint fort
```

**5. Trainingsverlauf ansehen**
```bash
tensorboard --logdir training/logs/
```
Im Browser: `http://localhost:6006`

**6. Trainiertes Modell vorführen (Live-Demo)**
```bash
python3 scripts/demo_run.py
```

---

## Hinweis zu `real_robot_inference.py`

Dieses Skript überträgt die Environment-Logik auf echte ROS2-Topics (`/scan`, `/odom`, `/cmd_vel`) und wurde vorbereitet, falls das trainierte Modell auf der realen Roboterplattform erprobt werden soll. **Es wurde bislang nicht auf echter Hardware getestet.** Vor der Nutzung unbedingt:

- `ros2 topic list` prüfen (sind `/scan`, `/odom`, `/cmd_vel` vorhanden?)
- `ros2 topic type /cmd_vel` prüfen und `CMD_VEL_TYPE` im Skript ggf. anpassen (`Twist` vs. `TwistStamped`)
- Roboter auf sicherer, freier Fläche testen; Fernbedienung/Not-Aus griffbereit halten

Das Skript enthält zusätzlich einen harten Sicherheits-Stopp bei Hindernisabstand < 15 cm, unabhängig vom gelernten Verhalten.

---

## Wichtigste Lektionen aus dem Projekt

Ausführlich in der Dokumentation behandelt, hier nur die Kernpunkte:

- **Reward Hacking:** Der Agent fand wiederholt Strategien, die die Belohnungsfunktion formal erfüllten, ohne das gewünschte Verhalten zu zeigen (z. B. Rückzug kurz vor dem Ziel).
- **„Zombie-Modus":** Passivitäts-/Stillstandsverhalten, da Nichtstun rechnerisch günstiger war als Kollisionsrisiko — behoben durch automatischen Episodenabbruch bei fehlendem Fortschritt.
- **Curriculum Learning** war der entscheidende Ansatz, um von seltenen Zufallserfolgen zu regelmäßigen, wiederholbaren Zielerreichungen zu kommen.
- Reinforcement-Learning-Training ist **nicht monoton stabil** — ein einmal guter Modellstand kann sich auch ohne Codeänderung wieder verschlechtern. Deshalb: regelmäßig Zwischenstände sichern und den besten, nachweislich stabilen Stand für die Demo auswählen (nicht zwingend den letzten).

---

## Lizenz / Kontext

Projektarbeit im Modul „Projektarbeit — Praktische Informatik", entstanden im Rahmen des Themenkomplexes „Testfeld Digitalisierung in der Produktion".
