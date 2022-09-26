# Guideline zu Clusterung und Klassifikationsskript

### Vorbemerkungen:

- **Primäres Ziel** ist die Klassifizierung von neuen (unbekannten) Input-Daten (nFk-Verläufen) in eine entsprechende Klasse/ Label
- Damit dies erfolgen kann, muss für das verwendete Klassifikationsmodell, a priori die Klassen bzw. Labels festgelegt sein.
  -> Dies ist inbesondere relevant für das trainieren (oder lernen) des Klassifikationsmodells, welche damit dann die neuen Input-Daten klassifizieren wird
- Dementsprechend ist der gesamte Prozess als zweistufig zu betrachten:
  1. Zunächst werden die Rohdaten (nFK-Jahresverläufe) in Gruppen hinsichtlich ihrer ähnlichen Jahresverläufe geclustert. Damit dienen die klassifiziert Daten als Basis-Input für die Klassifizierung von neuen nFK-Daten 
  2. Sobald das Cluserung erfolgt und somit die Klassen/ Labels definiert sind, kann diese in der zweiten Prozessstufe als Basis-Input verwendet werden, um das Klassifikationsmodell zu trainieren/ lernen, damit "es" weiß wie "es" neue unbekannte Input-Daten in welche Klasse zuordnet.
  3. Wenn das Klassifizieren von neuen Input-Objekten abgeschlossen ist, kann via einer Sichtprüfung die klassifizierten Daten mit dem Referenzverlauf, welcher als Plot im Clusterung-Skript erstellt wird, verglichen werden.


---

#### Generelle Anmerkungen zum Skript Clustering_script.py und Classifikation_script.py: 

- Es empfielt sich das Skript Peu-a-Peu auszuführen, dabei markieren "###" die wichtigen Abschnitte, die ausgführt werden müssen!
- Des Weiteren wird ebenso empfohlen, beide Skripte auf einer Umgebung mit entsprechenden und ausreichenden Hardware-Ressourcen auszuführen, da die Berechungen rechenintensiv sind

---

### Gruppierung von nFK-Jahresvberläufen mittels Clustering-Skript:

1. Die benötigten Libraries einladen **(Zeile 2-19)**
2. Voreinstellungen für die spätere Plot-Erstellung ausführen -> hier individuell adjustierbar **(Zeile 22-24)**
3. Entsprechende Arbeitsverzeichnisse erstellen, falls noch nicht verhanden (**Zeile 30-57)**
   - Achte hierbei auf die Bezeichungen der Arbeitverzeichnisse -> Es wird ein "working_dir\raw_data"-Verzeichnis benötigt, indem die Rohdaten abgelegt werden
   - Das "done"-Verzeichnis wird im Prinzip nicht benötigt, aber kann mit erstellt werden, wenn bspw. gewollt ist, dass die aufbereiteten Daten als csv-Datei exportiert wird (evtl. kurze Sichtprüfung der Daten, ob diese korrekt transformiert wurden, dazu später mehr)
5. Rohdaten aufbereiten, dazu zählen Zeitreihenlänge von 366 auf 365 Tage vereinheitlichen, überflüssige Metainformation aussschließen etc. **(Zeile 61-151)**
6. Als nächstes jede Zeitreihe vom Typ "data frame" zum Typ "Series" umkonvertieren **(Zeile 157-161)**
7. Spalte "year" erstellen und ein Data frame erstellen, wo später die nFK-Jahresverläufe gespeichert werden **(Zeile 165-171)**
8. Schnelltest, welcher prüft ob die Zeitreihen alle der Länge 365 haben **(Zeile 175-180)**
9. Normalisierung der Daten im Intervall [0,1], welcher für die spätere Clustering erforderlich ist **(Zeile 187-198)**
10. Anzahl der Cluster setzen sowie einen Seed; für die Reprodzierbarkeit -> Clusteranzahl adjustierbar **(Zeile 205-207)**
11. Start des Clustering mit K-Means **(Zeile 210-219)** 
    - Je nach Größe bzw. Anzahl von Zeitreihenobjekte kann die Clusterberechung von Minuten bis Stunden reichen
    - Zeile 216 - Adjustiberare Parameter: 
      - "metric": 'euclidean' (per default und bitte so lassen) -> Weitere Metriken sind 'dtw'
      - "max_iter": Gibt die maximale an Iterationen/ Durchläufe 
      - "n_init": Oder auch Multistart genannt, gibt die Anzahl an Wiederholung von gesetzten (zufälligen) Clusterzentroiden im Datenpunkten -> Empfohlen min. einen Wert von 50 Multistart zu wählen, damit keine lokale Clusterergebnisse enstehen!
      - "random_state": dient zur Reprodzierkeit
      - "n_jobs": Bei '-1' heißt eine Parallisierung der Berechung bei "dtw" -> ist optional bzw. Param kann entfernt werden
12. Wenn Clusterberchung abgeschlossen ist, dann die Cluster-Labels dem vorher erstellen Date Frame in Zeile 171 den jeweiligen nFk-Jahresvberlauf zuordnen **(Zeile 224)**
13. Export der Ergebnisse als hdf5 File **(Zeile 228)** bzw. falls eine kurze Sichtprüfung gewünscht auch als csv-Datei (**Zeile 229)**
14. Finaler Schritt ist das PLotten der Clusterergebnisse (Zeile 247 - 274)
    - Achtung: Wenn die Clusteranzahl in Zeile 207 höher oder neideriger gewählt sein sollte, dann ensprechende die Anzahl an Subplots in Zeile 248 adjustieren 
      - 1. Bsp.: n_cluster = 80 -> plot_count = 9 (es werden 81 (9x9) Subplots für die Clusterergbnisse erstellt) 
      - 2. Bsp.: n_cluster = 120 -> plot_count = 11 (121 Subplots) usw.
    - Der Plot kann dann genutzt werden, um die neuen klassifizerten Input-Daten (Classification_scipt.py), mit den aus dem Plot erstellten Referenzverläufen (rot) abzugleichen

---

### Klassifizerung von neuen unekannten nFK-Jahresverläufen mittels Classification_script.py:




