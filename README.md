# RAG

Alle Kommandos werden im Wurzelverzeichnis der Applikation ausgeführt.

## Voraussetzungen

* Node.js: https://nodejs.org/en
* Ollama: https://ollama.com/
* Docker: https://www.docker.com/
* Modelle laden
    * `ollama pull nomic-embed-text`
    * `ollama pull llama3.2`

## Applikation bauen
* Abhängigkeiten installieren: `npm install`
* `.env.sample` nach `.env` kopieren und gegebenenfalls die Werte anpassen
* Applikation bauen `npm run build`

## Vektordatenbank starten
Im infrastructure-Verzeichnis: `docker compose up`

## Ausführung

### Vorbereitung
* (PDF-) Dateien im `files`-Verzeichnis speichern und in der `.env` die Variable `INPUT_FILE` anpassen
* `npm run prepare`

### RAG
* Prompt anpassen: in der `.env` die Variable `PROMPT` anpassen
* `npm run rag`