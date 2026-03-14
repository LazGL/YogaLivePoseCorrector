# NamastAI — Plan d'amélioration validé

## Contexte
Application de correction de posture yoga en temps réel (NamastAI).
Stack : MediaPipe Pose + Qwen2.5-0.5B LLM local + Gradio + WebRTC + gTTS.

**Pattern d'usage clé** : Le téléphone est posé au sol, caméra pointant vers le haut. L'utilisateur est à distance et ne peut pas interagir pendant la session. Donc :
- La voix/audio est le canal de feedback principal
- L'UI doit être lisible à distance (overlays épais, grands éléments)
- Fonctionnement mains-libres après "Start"
- L'angle caméra (vue du bas) peut affecter la détection — à valider

**Deux modes** : Guided flow autonome + mode pratique libre.

---

## Travail déjà complété (Session 1) ✅

### Priorité 1 — Corrections de bugs critiques ✅
- **Double conversion couleur BGR→RGB** : corrigé en supprimant la conversion interne dans `run()`.
- **Thread safety sur `accuracy_score`** : ajout de `_accuracy_lock = threading.Lock()` dans `inference_new.py`, lecture protégée dans `app_llm.py`.
- **Typo `distance_bestween_feet`** : corrigé → `distance_between_feet` dans `calculate_difference.py` et `inference_new.py`.
- **Mismatch de clé** : `right_hand_height` → `right_hands_height` dans `calculate_difference.py`.
- **Threads non-daemon** : corrigé avec `daemon=True`.

### Priorité 2 — Intégration overlay squelette ✅
- `overlay_skeletons()` était morte, désormais appelée dans `run()` quand `accuracy_score < threshold`.

### Priorité 3 — Highlighting des pictogrammes ✅
- Ajout de `_pictograms_html(active_pose_key, completed_pose_keys)` dans `app_llm.py`.
- Timer retourne 5 valeurs : `[feedback, audio, accuracy, status, pictograms_html]`.

### Priorité 4 — Épinglage des dépendances ✅
- `requirements.txt` mis à jour avec plages de versions compatibles.

### Priorité 5 — Optimisation performance ✅
- `check_body_position()`, `check_engagement()`, `check_stance_width()` mis en cache (étaient appelés 2× par frame).

### Nouveaux fichiers ajoutés ✅
- `workout_engine.py` — State machine : IDLE → CALIBRATING → COUNTDOWN → ACTIVE_POSE → REST → COMPLETE
- `pose_config.py` — 4 poses (tpose, warrior2, warrior2_handsup, squat), 3 routines, seuils de précision
- `css_style.py` — Styles CSS extraits dans un module dédié
- `images_front_end/` — Pictogrammes PNG pour les 4 poses

---

## Plan d'implémentation — Phases 0 à 3

### Phase 0 : Foundation Fixes (à faire en premier)

#### 0A. Code Cleanup
Effort : Faible | Fichiers : `inference_new.py`, anciens fichiers app
- [ ] Supprimer ~90 lignes de code commenté/debug dans `inference_new.py` (lignes 181-270)
- [ ] Supprimer les debug prints (`print("yesss")`, `print("1")` → `print("14")`)
- [ ] Supprimer le commentaire debug ligne 263
- [ ] Ajouter le module Python `logging` à la place des `print`
- [ ] Archiver ou supprimer les versions obsolètes (`app.py`, `app2.py`, `apptts.py`) — l'entrée principale est `app_llm.py`

#### 0B. Fix Accuracy Scoring
Effort : Moyen | Fichiers : `inference_new.py`
- [ ] `calculate_accuracy()` ligne 149 est hardcodée à `return 75.00` — tout en dépend
- [ ] Utiliser les différences normalisées de `normalize_and_calculate_adjustments()`
- [ ] Calculer : `100 - (weighted_average_of_absolute_differences * scale_factor)`
- [ ] Pondérer les mesures selon la pose active
- [ ] Clamp entre 0 et 100
- [ ] L'infrastructure existe déjà — c'est du câblage, pas de nouvelle logique

#### 0C. Camera Angle Validation
Effort : Faible | Fichiers : aucun (tests uniquement)
- [ ] Tester la détection MediaPipe avec angle caméra au sol (vue du bas)
- [ ] Vérifier la fiabilité des landmarks (chevilles, pieds surtout)
- [ ] Documenter les ajustements nécessaires
- [ ] Si détection mauvaise : explorer tweaks MediaPipe ou pré-traitement (correction perspective)

---

### Phase 1 : Core Hands-Free Experience

#### 1A. Voice-First Feedback Enhancement
Effort : Moyen | Fichiers : `tts_utils.py`, `inference_new.py`, `app_llm.py`
- [ ] Évaluer TTS alternatives (pyttsx3 offline vs gTTS)
- [ ] Annoncer les noms de poses, countdowns, transitions
- [ ] Chimes/sons pour événements : pose réussie, changement, session terminée
- [ ] Répéter les corrections si accuracy ne s'améliore pas après 2 cycles

#### 1B. Dynamic Pose Selection (Pre-Session)
Effort : Moyen | Fichiers : `app_llm.py`, `inference_new.py`
- [ ] UI de sélection de poses (tap pour sélectionner/déselectionner, réordonner)
- [ ] Routines preset : "Beginner Flow", "Warrior Series", etc.
- [ ] Chaque pose → chemin image de référence + reference_tag + durée de maintien
- [ ] Stocker les définitions de poses dans une config (JSON/dict) au lieu de chemins hardcodés

#### 1C. Guided Auto-Flow Mode
Effort : Moyen-Élevé | Fichiers : `app_llm.py`, `workout_engine.py`
- [ ] Voix annonce : "First pose: Warrior 2. Get ready..."
- [ ] Countdown 5s avec ticks audio
- [ ] Boucle détection + feedback vocal pendant la durée de maintien
- [ ] Quand accuracy ≥ seuil pendant durée requise → "Great! Next pose..."
- [ ] Période de repos avec countdown entre les poses
- [ ] Résumé vocal de performance en fin de session
- [ ] State machine : SETUP → COUNTDOWN → ACTIVE_POSE → REST → ... → COMPLETE

#### 1D. Manual Practice Mode
Effort : Faible | Fichiers : `app_llm.py`
- [ ] Conserver le comportement actuel comme mode "Free Practice"
- [ ] Sélection d'une seule pose, feedback continu, pas de timer ni d'auto-avance

---

### Phase 2 : Visual & UX Improvements

#### 2A. Distance-Visible UI
Effort : Moyen | Fichiers : `css_style.py`, `app_llm.py`
- [ ] Grand chiffre d'accuracy (énorme police, color-codé : rouge → jaune → vert)
- [ ] Overlay squelette épais (lignes plus larges, cercles de joints plus grands, haut contraste)
- [ ] Parties du corps colorées : vert = correct, rouge/orange = à corriger
- [ ] Flèches directionnelles sur les joints à corriger (visibles à distance)
- [ ] Texte minimal à l'écran — la voix porte les détails

#### 2B. Calibration Screen
Effort : Moyen | Fichiers : `app_llm.py`, nouvelle logique de calibration
- [ ] Afficher une silhouette corporelle outline à l'écran
- [ ] Voix : "Place your phone on the ground and step back until your full body is visible"
- [ ] Auto-détection quand tout le corps est dans le cadre (landmarks clés > seuil de confiance)
- [ ] Voix : "Perfect! Starting in 3... 2... 1..."

#### 2C. Error Handling for Users
Effort : Faible | Fichiers : `app_llm.py`, `inference_new.py`
- [ ] "I can't see you clearly — please adjust your position"
- [ ] "Camera connection lost — please check your device"
- [ ] "Loading the AI model, please wait..."
- [ ] Fallback vers feedback rule-based si le LLM échoue

---

### Phase 3 : Engagement & Polish

#### 3A. Session Summary & Progress
Effort : Moyen | Fichiers : nouveau `session_tracker.py`, `app_llm.py`
- [ ] Voix lit le résumé post-workout ("You completed 4 poses, average accuracy 82%")
- [ ] Sauvegarder les données de session en JSON local
- [ ] Afficher l'historique quand l'utilisateur reprend le téléphone
- [ ] Suivi des streaks pour la motivation

#### 3B. Pose Library Expansion
Effort : Moyen | Fichiers : images de référence, config de poses
- [ ] Ajouter de nouvelles poses avec images de référence
- [ ] Activer les poses au sol (filtres de mesures déjà dans `get_pose_type_landmarks()`)
- [ ] Poses taguées par difficulté (débutant/intermédiaire/avancé)

#### 3C. Responsive Mobile Layout
Effort : Faible-Moyen | Fichiers : `css_style.py`, `app_llm.py`
- [ ] Optimiser le layout Gradio pour écrans de téléphone (layout vertical)
- [ ] Grandes zones tactiles pour le setup pré-session
- [ ] Gestion auto-rotate (paysage pour session, portrait pour setup)

---

## Plan de vérification (après chaque phase)
1. Test manuel sur téléphone — poser au sol, lancer une session complète
2. Test angle caméra — vérifier la détection de pose en vue du bas
3. Test audio — confirmer que le feedback vocal est audible à 2-3m
4. Test des deux modes — guided flow complet end-to-end, mode libre indépendant
5. Test des états d'erreur — sortir du cadre, bloquer la caméra, timeout LLM
6. Test multi-appareils — téléphone, tablette, laptop

---

## Fichiers clés
| Fichier | Rôle |
|---|---|
| `app_llm.py` | Point d'entrée principal, UI Gradio |
| `inference_new.py` | Moteur de détection de pose + LLM |
| `calculate_difference.py` | 25 mesures de pose |
| `tts_utils.py` | Text-to-speech |
| `css_style.py` | Styles UI |
| `workout_engine.py` | State machine guided flow |
| `pose_config.py` | Config poses et routines |
| `images_front_end/` | Pictogrammes des poses |

## Branche de travail
`claude/exciting-roentgen` — NE PAS commit directement sur `main`.
