# Halvorsen mit zwei unabhängigen Parametern a und b

Das Experiment trainiert ein ALRNN mit `phi_dim=2` auf der ganzen Box
`[1.4,2.1] × [2,4]`, bei festem `c=4` und `drive=0`. Die rechte Seite lautet

```text
dx = -a*x - b*y - c*z - y²
dy = -a*y - b*z - c*x - z²
dz = -a*z - b*x - c*y - x²
```

Die beiden lokalen Simulator-Dateien in `generate_trainingdata/time_varying_bifurcation_data`
und `generate_trainingdata/diffrentbutconst_prams` werden synchron gehalten. Ihre
`halvorsen_rhs` akzeptiert jetzt `b` und `c` als Schlüsselwortargumente mit den
alten Standardwerten `b=c=4`. Der neue Generator verwendet genau diese Funktion.
Andere Generatoren behalten ihre bisherigen eindimensionalen Schnittstellen.

## Vorbereitung

Alle folgenden Befehle laufen aus dem Repository `ALRNN-DSR-extended`. Python 3.10+
und die Pakete aus `requirements.txt` werden benötigt; getestet wurde mit Python
3.12. Für das volle Training wird eine CUDA-fähige PyTorch-Installation empfohlen.

```bash
python -m pip install -r alrnn_python/halvorsen_ab/requirements.txt
python alrnn_python/halvorsen_ab/sync_simulator.py
```

Der letzte Befehl prüft die vorhandene Simulator-Datei oder beide lokalen Kopien.
Mit `--apply` wird die Erweiterung auf die vorhandenen Dateien angewendet. Sind
beide Kopien vorhanden, müssen ihre Ausgangsdateien bytegleich sein.

**Auf dem Server reicht der vorhandene Ordner `diffrentbutconst_prams` mit seiner
`generate_dynamic_regime_trajectories.py`.** Weder `time_varying_bifurcation_data`
noch `mixed_parameter_trajectories` muss zusätzlich auf den Server kopiert werden.

Der Loader sucht automatisch zuerst nach `generate_trainingdata` neben dem
ALRNN-Repository, dann nach `diffrentbutconst_prams` daneben oder direkt im
Home-Verzeichnis. Explizit lässt sich über `--simulator-root` ein solcher Ordner
oder direkt die Simulator-Datei auswählen. Auch `GENERATE_TRAININGDATA_ROOT`
akzeptiert diese Orte. Der tatsächlich geladene Dateipfad wird ausgegeben.

Wenn der Server-Ordner unter `$HOME/diffrentbutconst_prams` liegt:

```bash
# Einmalig die vorhandene Kopie erweitern; bei bereits aktualisierter Datei nur prüfen.
python alrnn_python/halvorsen_ab/sync_simulator.py --simulator-root "$HOME/diffrentbutconst_prams" --apply

# Trainingsdaten erzeugen, mit vier Threads für die numerischen Bibliotheken.
python -u alrnn_python/halvorsen_ab/run_experiment.py --stage generate --simulator-root "$HOME/diffrentbutconst_prams" --cpu-threads 4

# Anschließend Training und Auswertung; der vorhandene Datensatz wird wiederverwendet.
python -u alrnn_python/halvorsen_ab/run_experiment.py --simulator-root "$HOME/diffrentbutconst_prams" --gpu-id 0
```

## Datensatz und Training

Die Einstellungen für die Simulation stehen in `config.json`:

| Einstellung | Standard |
| --- | --- |
| a-Gitter | 1.4, 1.575, 1.75, 1.925, 2.1 |
| b-Gitter | 2, 2.5, 3, 3.5, 4 |
| Anfangszustände je Paar | 8, unabhängig gezogen um (-4,0,0) |
| Gespeicherte Schritte je Trajektorie | 100000 |
| Burn-in vor der Speicherung | 5000 Schritte |
| dt | 0.01 |
| Training / zeitlicher Test | 80000 / 20000 Schritte je Trajektorie |
| Beobachtungsrauschen | keines |
| Zustandsnormalisierung | global, nur aus den Trainingsabschnitten |

Damit entstehen 200 lange Trajektorien. Das kompakte Format benötigt ungefähr
560 MB für Zustände, normiertes `phi` und rohe Parameter. Burn-in wird bei jedem
Paar mit dessen eigenen konstanten Parametern integriert. Fehlgeschlagene
Anfangszustände werden protokolliert und höchstens zehnmal neu gezogen. Es gibt
keine künstliche Begrenzung der ODE-Ableitungen.

Die Parameterkoordinaten sind fest und unabhängig:

```text
phi_a = (a - 1.75) / 0.35
phi_b = (b - 3.0) / 1.0
```

`trajectory_phi.npy` und `trajectory_phi_raw.npy` haben jeweils die Form
`(sum(trajectory_lengths),2)`, Kanalreihenfolge `[a,b]`. Konstante Parameterpaare
werden im Batch gleich häufig gezogen. Das Standardtraining nutzt ausschließlich
`C phi`, `M=300`, `P=150`, Batchgröße 100, Sequenzlänge 200 und 1000 Epochen mit
je 100 Batches. Es speichert den letzten Zustand (`--no-best-model`); die bisherige
Modellauswahl anhand nur der ersten Trainingstrajektorie entscheidet hier nicht
über die gesamte Box.

Plan anzeigen, danach den kompletten Lauf auf GPU 0 starten:

```bash
python alrnn_python/halvorsen_ab/run_experiment.py --dry-run
python -u alrnn_python/halvorsen_ab/run_experiment.py --gpu-id 0
```

Alternativ mit Bash:

```bash
bash alrnn_python/halvorsen_ab/run_halvorsen_ab.sh 0
```

Für den vollständigen Server-Ablauf einschließlich Aktivierung der bisherigen
CUDA-Umgebung, Paketinstallation, Simulator-Update, Datengenerierung, Training
und Auswertung gibt es einen eigenen Launcher:

```bash
bash alrnn_python/halvorsen_ab/run_halvorsen_ab_server.sh 0 4
```

Das erste Argument ist die GPU-ID (Standard 0), das zweite die Zahl der
CPU-Threads (Standard 4). Die Limits für numerische Bibliotheken und Build-Jobs
gelten bereits vor dem ersten Python-Aufruf. Ist `taskset` vorhanden, werden alle
Python-Schritte und ihre Kindprozesse außerdem auf höchstens diese Zahl bereits
zugelassener CPUs beschränkt. Vorhandene Slurm-/Container-CPU-Masken bleiben dabei
berücksichtigt. Das Skript bricht bei einem fehlgeschlagenen Schritt ab.

Standardpfade sind `$HOME/DynaMix-python-b-tipping/venv` und
`$HOME/diffrentbutconst_prams`. Andere Orte lassen sich mit `VENV_DIR` und
`SIMULATOR_ROOT` setzen. Der Launcher wechselt selbst in das ALRNN-Repository.

Die Schritte lassen sich einzeln ausführen:

```bash
python alrnn_python/halvorsen_ab/run_experiment.py --stage generate
python alrnn_python/halvorsen_ab/run_experiment.py --stage train --gpu-id 0
python alrnn_python/halvorsen_ab/run_experiment.py --stage analyze --gpu-id 0
```

Pfade: `--data-dir`, `--run-dir` und `--output-dir`. Ohne diese Argumente liegen
die Daten unter `alrnn_python/runs/halvorsen_ab/data_grid5x5`, das Modell unter
`alrnn_python/runs/halvorsen_ab/cphi_m300_p150` und die Auswertung in dessen
Unterordner `analysis`. Für erneutes Training einen neuen `--run-dir` verwenden.
Ein vorhandener Datensatz wird nur bei übereinstimmender Konfiguration und
Quellcode-Prüfsumme wiederverwendet. Das Training legt außerdem
`halvorsen_ab_training.json` mit der Datensatzidentität und den Koordinaten ab.

Weitere Modellterme sind z.B. mit `--phi-terms additive shifted-relu` möglich.
Modellgröße, Epochen, Batchgröße und CPU-Threads sind per CLI einstellbar.

## Interaktive Bifurkationen und Trajektorien

Die Auswertung berechnet standardmäßig ein **41×41-Gitter** innerhalb derselben
Box mit zwei Anfangszuständen je Paar. Dieses Gitter enthält alle Trainingswerte.
Nach dem ODE-Burn-in starten Ground Truth und Modell im selben beobachteten
Zustand. Es folgen 1000 Einschwing-Schritte und 10000 ausgewertete Zustände.
Die Modellrollouts werden in Batches auf der gewählten GPU gerechnet.

`analysis/bifurcation.html` enthält:

- Die drehbare Plotly-Punktwolke mit Achsen `a`, `b`, lokale Maxima von `z`.
- Ground Truth und ALRNN als getrennt schaltbare Datensätze in Rohkoordinaten.
- Eine verschiebbare Ebene bei festem `a` oder `b` und daneben das zugehörige
  zweidimensionale Bifurkationsdiagramm.
- Zahlenfeld, Schieberegler und Punktklick zur Wahl des fixierten Parameters.
  Es werden tatsächlich berechnete Rasterwerte ausgewählt; es wird nicht zwischen
  Bifurkationszweigen interpoliert. Für feinere Schnitte das Auswertungsgitter erhöhen.
- CSV-Export des dargestellten Schnitts.
- Einen auswählbaren 3D-Vergleich der freien Trajektorien für alle 25 Trainingspaare,
  einschließlich Dstsp, Spektralfehler DH und RMSE.

Die Trajektorienvergleiche verwenden jeweils den Testabschnitt der ersten
Trajektorie eines Paars. Der beobachtete Anfangszustand entspricht `prediction[0]`;
ab `prediction[1]` wird das Modell frei iteriert. Für chaotische Langzeitvergleiche
sind Attraktorgeometrie und Spektren aussagekräftiger als phasengenaue RMSE.
Bei konstanten Signalen bleiben undefinierte Histogramm-/Spektralmetriken leer.

Stationäre Zustände erscheinen als gesonderte Rauten. Fehlgeschlagene oder über
die konfigurierte Zustandsgrenze ausbrechende Rollouts werden kenntlich gemacht;
aus ihnen werden keine vollständigen Maxima-Verteilungen erzeugt. Maxima werden
an den gespeicherten dt-Rasterpunkten bestimmt. Die HTML zeigt höchstens 128
zeitlich gleichmäßig ausgewählte Maxima je Paar, Anfangszustand und System;
vollständige Maxima stehen in den NPZ-Dateien.

Die HTML funktioniert nach dem Öffnen im Browser vollständig offline. Alle
Plotly-Skripte und Daten sind eingebettet. Weitere Ergebnisse:

| Datei | Inhalt |
| --- | --- |
| `ground_truth.npz` | Wiederverwendbare ODE-Maxima, stationäre Werte und Startzustände |
| `analysis.npz` | Vollständige GT-/Modell-Maxima, Trajektorien, Fehlerstatus und Metadaten |
| `metrics.json` | Modell-/Datenidentität, Einstellungen und Vergleichsmetriken |
| `trajectories.png` | Statischer Vergleich der vier Boxecken und der Mitte |

Analyseauflösung und Laufzeit steuern:

```bash
python alrnn_python/halvorsen_ab/run_experiment.py --stage analyze --grid-a 81 --grid-b 81 --gpu-id 0
```

Der GT-Cache wird anhand von Gleichungen, Einstellungen und Quellcode geprüft.
HTML und PNG können ohne erneute Simulation/Modellrollouts aus `analysis.npz`
erzeugt werden:

```bash
python alrnn_python/halvorsen_ab/run_experiment.py --stage plot
```

Eine erste Ground-Truth-HTML ist schon vor dem Training möglich:

```bash
python alrnn_python/halvorsen_ab/run_experiment.py --stage ground-truth --device cpu
```

Das ist eine Rekonstruktion innerhalb der vollständig abgedeckten Box.
Parameterextrapolation und zeitlich driftende Parameter sind hier nicht Teil des Experiments.

## Kleine Funktionsprüfung

Ein kurzer Lauf prüft Datengenerierung, zweidimensionale Eingaben, Training,
Checkpoint-Laden und die gesamte Auswertung auf der CPU. Dabei bleiben alle
25 Trainingspaare erhalten, aber mit nur einer kurzen Trajektorie pro Paar.
Die HTML ist ausdrücklich als Smoke-Test markiert.

```bash
python alrnn_python/halvorsen_ab/run_experiment.py --smoke --device cpu --data-dir alrnn_python/runs/halvorsen_ab_smoke/data --run-dir alrnn_python/runs/halvorsen_ab_smoke/model
python -m unittest discover -s alrnn_python/halvorsen_ab -p test_experiment.py -v
```

Optionaler tatsächlicher Browser-Test (beide Schnittrichtungen, Zahlenfeld,
Trajektorienwahl, CSV-Download, keine Netzwerkanfragen):

```bash
python -m pip install playwright
python -m unittest discover -s alrnn_python/halvorsen_ab -p test_report_browser.py -v
```

Chrome/Chromium/Edge wird lokal gesucht; alternativ `HALVORSEN_BROWSER` auf die
Browser-Executable setzen. Das volle 1000-Epochen-Training wird durch Tests
nicht gestartet.
