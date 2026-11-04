# Isolierter Lorenz-Bifurkationstest

Dieser Ordner ergänzt einen eigenen Vergleich nach dem Verfahren in
[Anhang D des Papers](https://arxiv.org/html/2507.19036v2#A4). Die bestehenden
Trainings- und Bifurkationsskripte werden weder bearbeitet noch ausgeführt.
Es werden vorhandene Checkpoints ausgewertet, keine Modelle neu trainiert.

## Serverstart

Aus dem Hauptordner von `ALRNN-DSR-extended`:

```bash
bash alrnn_python/lorenz_paper_test/run_lorenz_paper_test.sh 2 3
```

Das Skript aktiviert die b-tipping-venv und verwendet standardmäßig den bereits
trainierten Lauf:

```text
alrnn_python/runs/term_ablations/lorenz_rho/combined_lorenz_rho_balanced/20261103_165123_417235_497351
```

Beide Varianten werden ausgewertet: `cphi` und `cphi_shifted_relu`, jeweils mit
den beiden Modellen für rho in [5,23] und [1,30]. Eine GPU funktioniert ebenfalls;
dann laufen die Varianten nacheinander. Mehr als zwei GPUs beschleunigen diese
zwei unabhängigen Auswertungen nicht. Die Ground Truth wird zuvor einmal auf der
CPU berechnet und von beiden Varianten verwendet.

Andere Pfade oder eine kleinere Parameterzahl lassen sich voranstellen:

```bash
RUN_ROOT=/path/to/existing_run \
TRAINING_DATA_ROOT=/path/to/combined_lorenz_rho \
OUTPUT_ROOT=/path/to/new_output NUM_PARAMETERS=50 \
bash alrnn_python/lorenz_paper_test/run_lorenz_paper_test.sh 2 3
```

Ohne Override kommt der Trainingsdatenpfad aus `plan.json`, sonst aus dem
bisherigen Serverpfad. Benötigt werden die Kalibrierungsdateien und die
Parameterarrays des Trainingsdatensatzes, um die tatsächliche Trainingsabdeckung
zu schattieren. An diesen Dateien wird nichts verändert. Der neue Code benötigt
keine Kopie des externen Bifurkations-Backends und keine neuen Python-Pakete.

## Verfahren

- Lorenz-Gleichungen mit sigma=10 und beta=8/3, konstantem rho pro Simulation.
- 290 äquidistante rho-Werte in [1,30], inklusive der Randpunkte.
- Für **jeden** rho-Wert genau die beiden physikalischen Anfangszustände
  `(1,1,1)` und `(-1,-1,1)`, ohne zufällige Perturbationen.
- Ground Truth **und Modell** beginnen bei t=0 aus dem jeweiligen Anfangszustand.
  Das Modell initialisiert `latent = normalized_x0 @ B` und überschreibt die ersten
  drei latenten Koordinaten mit `normalized_x0`. Danach läuft es vollständig frei.
- Beide Systeme durchlaufen ihre **eigene** Einschwingphase bis t=200.
  Die Auswertung umfasst t=200 bis t=250, einschließlich beider Randzustände.
  Insbesondere wird das Modell nicht aus einem Ground-Truth-Zustand bei t=200
  gestartet und erhält keine Teacher-Forcing- oder Kontextkorrekturen.
- Es werden **lokale Minima und Maxima von x** gesammelt. Die Ground Truth bleibt
  bis zur Extrema-Erkennung in float64. Die ALRNN-Rekurrenz behält ihre gelernte
  float32-Rechenweise.
- Die globale Trainingsstandardisierung bleibt erhalten. Jedes Modell bekommt
  Eingaben in seinen eigenen Trainingskoordinaten; alle Plotwerte werden in die
  festen Referenzkoordinaten des Datensatzes [5,23] übertragen.
- Zeitschritt ist das dt der Trainingsdaten, standardmäßig 0.01. Bei ALRNNs
  verändert ein anderer Zeitschritt die gelernte diskrete Dynamik; daher wird ein
  davon abweichendes dt abgelehnt. Warm-up und Beobachtungsdauer entsprechen
  dennoch 200 und 50 physikalischen Zeiteinheiten.

Der Test übernimmt die beiden Anfangszustände, die Zeitfenster und die Wahl
beider Extremtypen aus dem Paper. Normierung, Raster inklusive Randpunkten,
trainiertes dt und die explizite Stationaritätserkennung sind hier dokumentierte
eigene Einstellungen; dies ist keine bitgenaue Reproduktion der Autoren-Software.

## Stationäre Lösungen und Diagnose

Fehlende Extrema allein bedeuten **nicht**, dass ein Fixpunkt erreicht wurde.
Wir prüfen stattdessen über das gesamte Beobachtungsfenster:

1. Die Spannweite **jeder** der drei beobachteten Koordinaten muss kleiner sein als
   `stationary_atol + stationary_rtol * max(1, max(abs(coordinate)))`.
2. Die abschließende Änderungsrate muss in allen drei Koordinaten kleiner als
   `stationary_rate` sein. Bei der Ground Truth wird dafür das Lorenz-Vektorfeld
   am Endzustand verwendet; beim Modell der letzte Rekurrenzschritt geteilt durch dt.

Defaults sind atol=1e-5, rtol=1e-5 und Rate=1e-3 in standardisierten Beobachtungen
pro Zeiteinheit. Wenn beide Bedingungen erfüllt sind, wird ein einzelner Punkt
beim mittleren x des Auswertefensters eingetragen, separat je Anfangszustand.
Andernfalls werden die tatsächlich vorhandenen lokalen Extrema verwendet.
Monotone, nicht stationäre Verläufe können daher weiterhin ohne Punkt bleiben;
dies wird als `no_extrema_not_stationary` protokolliert.

Der Test klassifiziert numerisch stationäre **Beobachtungen**. Er ist kein Beweis
für mathematische Konvergenz oder Stabilität und verlangt bei ALRNNs nicht, dass
alle unbeobachteten latenten Dimensionen konstant sind. Die Toleranzen und
Diagnosen werden mitgespeichert. NaN/Inf in einem latenten Modellzustand beendet
die gültige Trajektorie; deren Punkte werden vollständig ausgeschlossen und die
erste fehlerhafte Schrittzahl wird protokolliert. Sehr große, aber endliche
Werte werden nicht als numerische Divergenz verborgen oder abgeschnitten.

## Ausgabe und Wiederverwendung

Default, relativ zu `alrnn_python/lorenz_paper_test`:

```text
results/20261103_165123_417235_497351/
  ground_truth.npz
  cphi/
    points.npz
    bifurcation_x_compare.png
    bifurcation_x_ground_truth.png
    bifurcation_x_model_01.png
    bifurcation_x_model_02.png
    bifurcation_sources.json
  cphi_shifted_relu/
    ...dieselben Dateien...
  logs/<timestamp>/
```

Die Vergleichsbilder zeigen Ground Truth und beide Modelle in getrennten
Feldern untereinander mit gemeinsamen Achsen. Trainingsbereiche werden wie
bisher blass blau hinterlegt; außerdem gibt es alle Felder einzeln. Die
Punktarrays enthalten `rho`, `value`, `kind` und `initial_condition`.
`kind=-1` bedeutet Minimum, `kind=+1` Maximum, `kind=0` stationäre Beobachtung;
`initial_condition=0/1` ordnet den Punkt einem der beiden Startwerte zu.
Im Modellcache heißen die entsprechenden Arrays `model_0_rho` usw.
Eine JSON pro Variantenordner dokumentiert die Checkpoints und alle Bilder;
Diagnosen je rho/Anfangszustand liegen zusätzlich in den Cache-Metadaten.

Diese eigenen Caches enthalten beide Extremtypen und beide Anfangszustände.
Die alten reinen Maxima-Caches können dafür nicht wiederverwendet werden und
werden nicht verändert. Ein unterbrochener Ground-Truth-Lauf muss neu berechnet
werden; vollständige Caches werden atomar geschrieben und beim nächsten Start
nach Prüfung ihrer Einstellungen wiederverwendet. Bereits vollständig
berechnete Modellvarianten können ebenfalls wiederverwendet werden.

```bash
# Nur Einstellungen anzeigen; weder Simulation noch CUDA erforderlich.
DRY_RUN=1 bash alrnn_python/lorenz_paper_test/run_lorenz_paper_test.sh 2 3

# Aus den eigenen fertigen Caches erneut plotten; kein CUDA nötig.
PLOT_ONLY=1 bash alrnn_python/lorenz_paper_test/run_lorenz_paper_test.sh
```

Für weitere Optionen die Python-Datei direkt verwenden, z.B.:

```bash
python alrnn_python/lorenz_paper_test/evaluate_lorenz.py --help
```

Die Python-CLI unterstützt u.a. `--plot-only`, `--recompute`, `--device cpu`,
`--warmup-time`, `--observation-time`, `--parameter-min/max`, die drei
Stationaritätstoleranzen und `--batch-size`. Modellinitialisierung und Plot-
Skalierung bleiben bei diesen Optionen unverändert. Änderungen an
Simulationsparametern, Code oder Checkpoints werden anhand von Hashes geprüft;
bei einem nicht passenden Cache wird ein neuer Ausgabeordner oder explizites
`--recompute` verlangt. Zum Wiederplotten müssen die Checkpoints und die
Trainingskalibrierung weiterhin lesbar sein, damit ihre Identität geprüft wird.

Tests aus diesem Ordner:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest test_lorenz_paper_test
```
