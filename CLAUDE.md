# NamastAI — Plan d'amélioration validé

## Contexte
Application de correction de posture yoga en temps réel (NamastAI).
Stack : MediaPipe Pose + Qwen2.5-0.5B LLM local + Gradio + WebRTC + gTTS.

## Priorités validées (toutes complétées ✅)

### Priorité 1 — Corrections de bugs critiques ✅
- **Double conversion couleur BGR→RGB** : `app_llm.py` convertissait avant d'appeler `run()`, et `run()` reconvertissait. Corrigé en supprimant la conversion interne.
- **Thread safety sur `accuracy_score`** : ajout de `_accuracy_lock = threading.Lock()` dans `inference_new.py`, lecture protégée dans `app_llm.py`.
- **Typo `distance_bestween_feet`** : corrigé → `distance_between_feet` dans `calculate_difference.py` et `inference_new.py`.
- **Mismatch de clé** : `right_hand_height` → `right_hands_height` dans `calculate_difference.py` (pour correspondre à `inference_new.py`).
- **Threads non-daemon** : les threads de feedback LLM pouvaient bloquer l'arrêt de l'app. Corrigé avec `daemon=True`.

### Priorité 2 — Intégration overlay squelette ✅
- `overlay_skeletons()` était une méthode morte jamais appelée.
- Désormais appelée dans `run()` quand `accuracy_score < threshold` : affiche le squelette de référence en bleu par-dessus le squelette utilisateur.

### Priorité 3 — Highlighting des pictogrammes ✅
- Les classes CSS `.highlighted` et `.completed` existaient mais n'étaient jamais appliquées dynamiquement.
- Ajout de `_pictograms_html(active_pose_key, completed_pose_keys)` dans `app_llm.py` : génère du HTML dynamique avec bordure verte (actif), grise (complété), transparente (en attente).
- Composant `gr.Image` statique remplacé par `gr.HTML` mis à jour par le timer.
- Le timer retourne désormais 5 valeurs : `[feedback, audio, accuracy, status, pictograms_html]`.

### Priorité 4 — Épinglage des dépendances ✅
- `requirements.txt` mis à jour avec des plages de versions compatibles :
  - `opencv-python>=4.8.0,<5.0`
  - `mediapipe>=0.10.9,<0.11`
  - `gradio>=5.6.0,<6.0`
  - `gradio-webrtc>=0.0.18`
  - `gtts>=2.5.0`
  - `transformers>=4.36.0,<5.0`
  - `torch>=2.1.0`
  - `numpy>=1.24.0,<2.0`

### Priorité 5 — Optimisation performance ✅
- Dans `calculate_difference.py` : `check_body_position()`, `check_engagement()`, `check_stance_width()` étaient chacune appelées 2× par frame. Résultats mis en cache et réutilisés.

## Fichiers clés modifiés
| Fichier | Changements |
|---|---|
| `inference_new.py` | Thread safety, overlay squelette, correction typo, suppression double conversion |
| `app_llm.py` | Pictogrammes dynamiques HTML, thread safety lecture accuracy, timer 5 outputs |
| `calculate_difference.py` | Typos, mismatch clés, optimisation calculs |
| `requirements.txt` | Versions épinglées |

## Nouveaux fichiers ajoutés (session précédente)
- `workout_engine.py` — State machine guided flow : IDLE → CALIBRATING → COUNTDOWN → ACTIVE_POSE → REST → COMPLETE
- `pose_config.py` — 4 poses (tpose, warrior2, warrior2_handsup, squat), 3 routines, seuils de précision
- `css_style.py` — Styles CSS extraits dans un module dédié
- `images_front_end/` — Pictogrammes PNG pour les 4 poses

## Pistes futures possibles
- [ ] Ajouter des images de référence dédiées pour `warrior2_handsup` et `squat` (TODO dans `pose_config.py`)
- [ ] Tests end-to-end avec webcam réelle
- [ ] Ajouter de nouvelles poses
- [ ] Améliorer le prompt LLM pour des retours plus précis par zone corporelle
- [ ] Mode multi-utilisateur ou historique des sessions

## Branche de travail
`claude/exciting-roentgen` — NE PAS commit directement sur `main`.
