# Football Predictor AI

Een gratis prototype van een voetbalvoorspeller voor Europese competities.

## Wat doet het?
- Haalt openbare historische wedstrijdresultaten (waar beschikbaar vanaf 1993) op van Football-Data.co.uk.
- Bouwt kenmerken op basis van eerdere wedstrijden van teams.
- Trained drie modellen: wedstrijdresultaat, thuisdoelpunten en uitdoelpunten.
- Schat winstkansen, een score, doelpunten en beide teams scoren.

## Belangrijke beperkingen
- De dataset bevat niet letterlijk alle voetbalwedstrijden uit de geschiedenis.
- Beschikbaarheid verschilt per competitie en seizoen.
- Dit prototype voorspelt op basis van historische vorm en competitiecode. Het gebruikt nog geen actuele blessures, opstellingen, live odds of geavanceerde spelersdata.
- Het model wordt opnieuw getraind wanneer de app opnieuw wordt gestart nadat de data is ververst. Het leert dus niet automatisch direct na het laatste fluitsignaal; de databron moet de uitslag eerst publiceren.
- Een voorspelling is onzeker en is niet bedoeld als advies om geld in te zetten.

## Lokaal starten
1. Installeer Python 3.10 of nieuwer.
2. Open een terminal in deze map.
3. Maak een virtuele omgeving (optioneel maar aanbevolen):
   `python -m venv .venv`
4. Activeer de omgeving:
   - macOS/Linux: `source .venv/bin/activate`
   - Windows: `.venv\\Scripts\\activate`
5. Installeer pakketten:
   `pip install -r requirements.txt`
6. Start:
   `streamlit run app.py`

## Gratis online zetten
1. Maak een gratis account op GitHub.
2. Maak een nieuwe repository, bijvoorbeeld `football-predictor-ai`.
3. Upload `app.py`, `requirements.txt` en `README.md`.
4. Maak een gratis account op Streamlit Community Cloud.
5. Kies **Create app**, selecteer je GitHub-repository en `app.py`.
6. Deploy. De eerste start kan even duren omdat historische CSV-bestanden worden opgehaald en het model wordt getraind.

## Databron
Football-Data.co.uk: https://www.football-data.co.uk/
