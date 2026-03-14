# NamastAI — État du projet

## Stack actuelle
MediaPipe Pose + Qwen2.5-0.5B LLM local + FastAPI + WebSocket + gTTS/pyttsx3

**Point d'entrée** : `python server.py` → `http://localhost:8000`
**Branche de travail** : `claude/review-claude-md-ejrxq` — NE PAS commit sur `main`

---

## Décisions produit actées

| Décision | Valeur |
|---|---|
| Cible | Débutant complet |
| Modèle économique | Freemium (poses libres vs poses premium) |
| Traitement vidéo | 100% local — les frames ne quittent jamais le device |
| Nom | NamastAI |
| Usage prioritaire | Desktop d'abord, puis optimisation mobile |

---

## Ce qui a été fait ✅

### Architecture
- **Migration Gradio → FastAPI + vanilla JS** (`server.py`, `static/`)
  - `/ws/video` : browser envoie JPEG → serveur annote → renvoie JPEG
  - `/ws/feedback` : push JSON à 10 Hz (vs polling 3s avec Gradio) → latence visuelle ~100ms
  - TTS en thread daemon (non-bloquant)
  - LLM chargé une seule fois au démarrage, partagé entre toutes les poses
  - `switch_reference()` pour changer de pose sans recharger le modèle

### Bugs corrigés
- Double conversion BGR→RGB supprimée
- Thread safety sur `accuracy_score` (`_accuracy_lock`)
- Typo `distance_bestween_feet` → `distance_between_feet`
- Mismatch clé `right_hand_height` → `right_hands_height`
- Threads non-daemon → `daemon=True`
- `overlay_skeletons()` appelée 2× avec données identiques → suppression du call redondant
- LLM thread guard : ne spawne plus de threads concurrents
- Imports MediaPipe robustes (fonctionne sur mp 0.9 / 0.10 / 0.10.32+)

### Features implémentées
- `workout_engine.py` — state machine IDLE → CALIBRATING → COUNTDOWN → ACTIVE_POSE → REST → COMPLETE
- `pose_config.py` — 4 poses, 3 routines, seuils de précision
- `session_tracker.py` — sauvegarde JSON des sessions, historique dans l'UI
- `tts_utils.py` — fallback offline pyttsx3 si gTTS échoue (pas de réseau)
- Répétition auto des corrections si l'accuracy stagne (2 cycles sans amélioration)
- Fallback rule-based si le LLM échoue
- UI responsive (grande accuracy visible à distance, mobile-ready CSS)
- `static/index.html` + `static/app.js` + `static/style.css` — frontend complet

### Optimisations performance
- Frozensets `_ANGLE_KEYS` / `_DISTANCE_KEYS` pour normalisation O(1)
- `process_every_n_frames = 2` (analyse complète tous les 2 frames)
- Pre-cache `_VALID_POSES` (évite os.path.exists à chaque tick)

---

## Ce qui reste à faire 🔲

### Sprint 1 — Bugs critiques (à faire avant tout)

#### 1. Images de référence manquantes
`warrior2_handsup` et `squat` utilisent `target2.png` (image Warrior II) → accuracy fausse
- Trouver/créer des vraies images de référence pour ces 2 poses
- Mettre à jour `pose_config.py`

#### 2. Valider l'accuracy empiriquement
**NE PAS modifier la formule avant d'avoir testé en vrai.**
- Lancer `python server.py`, faire T-Pose + Warrior II devant la webcam
- Noter les scores → si cohérents (>50% pour bonne pose) : ne rien changer
- Si systématiquement 0-5% : appliquer pondération par pose (clé `weights` dans `pose_config.py`)

#### 3. Freemium gating
- Ajouter clé `tier: "free" | "premium"` dans `pose_config.py`
- Free : T-Pose + Warrior II
- Premium : toutes les poses + routines avancées
- Frontend : poses premium avec cadenas, clic → message upgrade

### Sprint 2 — Expérience débutant

#### 4. Onboarding 3 écrans
- Écran 1 : comment positionner le laptop/téléphone
- Écran 2 : explication squelette bleu (référence) vs vert (utilisateur)
- Écran 3 : sélection du mode
- `localStorage` pour ne pas le remontrer

#### 5. Calibration visuelle
- Pendant état `CALIBRATING` : silhouette humaine en overlay sur canvas
- Passe au vert quand le corps est détecté dans le cadre

#### 6. Nouvelles poses (4 prioritaires)
- Mountain Pose, Tree Pose, Chair Pose, Child's Pose
- Pour chaque : image de référence + pictogramme + tier (free/premium)

#### 7. Feedback langage débutant
- Réécrire le prompt dans `inference_new.py:generate_feedback()`
- Ton bienveillant, simple, encourageant (pas de jargon technique)
- Ajouter encouragements si accuracy s'améliore

### Sprint 3 — Rétention

#### 8. Progression par pose
- Graphe accuracy sur 7 dernières sessions par pose
- `session_tracker.py` étendu + Chart.js frontend

#### 9. Streaks + partage
- Compteur jours consécutifs dans `session_tracker.py`
- Bouton "Partager" → PNG avec résultats

---

## Fichiers clés

| Fichier | Rôle |
|---|---|
| `server.py` | Point d'entrée principal (FastAPI) |
| `inference_new.py` | Détection pose + LLM feedback |
| `calculate_difference.py` | 25 mesures de pose |
| `tts_utils.py` | Text-to-speech (gTTS + pyttsx3 fallback) |
| `workout_engine.py` | State machine guided flow |
| `pose_config.py` | Config poses et routines |
| `session_tracker.py` | Persistance sessions JSON |
| `static/index.html` | Frontend HTML |
| `static/app.js` | Frontend JS (WebSocket, caméra, UI) |
| `static/style.css` | Styles UI |
| `app_llm.py` | Ancienne UI Gradio (conservée pour référence) |

---

## Lancer le projet

```bash
git checkout claude/review-claude-md-ejrxq
pip install -r requirements.txt
python server.py
# → http://localhost:8000
```

**Note MediaPipe** : testé sur 0.10.32 (macOS, Python 3.12). Les imports sont robustes multi-versions.
**Note LLM** : Qwen2.5-0.5B se charge au démarrage (~20-30s). Premier clic sur une pose est instantané ensuite.
