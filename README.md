# 64 Facets Autonomous Luxury Studio

## Übersicht

Dieses Projekt automatisiert die Erstellung hochwertiger Luxus-Juwelenbilder mit minimalem menschlichen Eingriff. Das System nutzt Google Vertex AI (Gemini 1.5 Pro, Imagen 3) und eine Directed Graph Pipeline, um Produktbilder zu analysieren, künstlerisch in Szene zu setzen, digital zu integrieren und automatisch auf Qualität zu prüfen.  

Das Ergebnis: End-to-End 4K-Bilder, die exakt den realen Produkten entsprechen, begleitet von vollständigen Berichten.

---

## Architektur

Die Pipeline ist als Directed Cyclic Graph konzipiert. Sie wiederholt bestimmte Schritte, bis ein Qualitätsziel erreicht wird.

### Nodes

1. **Analyst (Gemologist)**
   - Modell: Gemini 1.5 Pro Vision
   - Aufgabe: Analyse des Originalprodukts
   - Output: `ProductSpecs` JSON mit Metalltyp, Hauptstein (Cut, Farbe, Klarheit), Setting-Stil und einzigartigen Makeln

2. **Art Director**
   - Modell: Gemini 1.5 Pro
   - Aufgabe: Übersetzung der Specs in ein visuelles Szenenkonzept
   - Output: `ScenePlan` JSON mit Prompt, Negative Prompt, Beleuchtungskarte, Inpaint-Koordinaten

3. **Producer (Studio)**
   - Modell: Imagen 3 (Editing/Inpainting)
   - Aufgabe: Szenengenerierung und präzises Einfügen des Produkts
   - Schritte:
     1. Base Scene ohne Schmuck generieren
     2. Original-PNG ins Szenenbild einfügen
   - Output: `candidate_image_vX.png`

4. **Judge (Quality Officer)**
   - Modell: Gemini 1.5 Pro Vision
   - Aufgabe: Vergleich Original vs. Kandidat, Bewertung von Cut, Form, Metall, Anatomie
   - Output: Score 0-100 und Feedback JSON

### Feedback Loop

- Score ≥ 90 → Pipeline endet, Bild wird ausgegeben
- Score < 90 → Feedback zurück an Producer für Re-Generation
- Max Retries: 3

---

## Vorteile

- **Hallucination Firewall**: Analyst verhindert falsche Darstellung von Schmuckdetails  
- **Autonome Selbstkorrektur**: Feedback Loop korrigiert automatisch fehlerhafte Bilder  
- **Composite Method**: Editing/Inpainting bewahrt die Originalität der Produkte

---

## Ordnerstruktur

```

backend/
│
├─ agents/
│   ├─ analyst.py
│   ├─ art_director.py
│   ├─ judge.py
│   └─ producer.py
│
├─ graph/
│   └─ graph_workflow.py
│
├─ llm/
│   └─ gemini_pipeline.py
│
├─ routes.py
├─ config.py
├─ schemas.py
├─ main.py
├─ requirements.txt
├─ .env
│
├─ tests/
│   ├─ test_analyst.py
│   ├─ test_director.py
│   ├─ test_graph_workflow.py
│   ├─ test_judge.py
│   └─ test_producer.py

````

---

## Installation

```bash
git clone <repo-url>
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

`.env` Datei anpassen:

```
ANALYST_MODEL="gemini-1.5-pro"
ART_DIRECTOR_MODEL="gemini-1.5-pro"
JUDGE_MODEL="gemini-1.5-pro"
PRODUCER_MODEL="gemini-1.5-pro"
GEMINI_API_KEY="<DEIN_API_KEY>"
OUTPUT_DIR="./results"
INPUT_DIR="./input"
MIN_ACCEPTED_SCORE="90"
MAX_RETRIES="3"
```

---

## Nutzung

### Einzelbild verarbeiten

```python
from routes import run_single
from pathlib import Path

image_path = Path("Input/ring.png")
state = run_single(image_path)
```

### Batch-Processing

```bash
curl -X POST "http://localhost:8000/process/folder"
```

oder über Upload:

```bash
curl -F "files=@ring.png" http://localhost:8000/process/upload-batch
```

---

## Tests

```bash
pytest tests/ --maxfail=1 --disable-warnings -v
```

---

## Technologie-Stack

* **Python 3.11+**
* **Google Vertex AI**: Gemini 1.5 Pro, Imagen 3
* **FastAPI** für API Endpunkte
* **LangGraph** für Graph-basierte Pipeline
* **Pydantic** für strikte Datenvalidierung
* **pytest** für Unit Tests

---

## Ergebnis

* 4K Endbilder
* JSON-Reports mit Analyse, Szenenplan, Feedback und Korrekturversuchen
* Vollständig autonomer Ablauf mit Feedbackschleifen
