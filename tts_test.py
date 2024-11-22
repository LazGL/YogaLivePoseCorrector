import os
import shutil
import platform

class Dimits:
    def __init__(self, parent_destn):
        self.parent_destn = parent_destn

    def _get_os(self):
        """
        Cette fonction vérifie le système d'exploitation et renvoie:
        - 'Linux' pour les systèmes basés sur Linux
        - 'Windows' pour les systèmes Windows
        - 'macOS' pour macOS
        - 'unsupported_machine' pour les autres
        """
        os_name = platform.system()
        if os_name == 'Linux':
            return 'Linux'
        elif os_name == 'Windows':
            return 'Windows'
        elif os_name == 'Darwin':  # macOS est identifié par 'Darwin'
            return 'macOS'
        else:
            return 'unsupported_machine'

    def text_2_speech(self, text: str, engine: str = None, **kwargs: str) -> None:
        """
        Convertit le texte fourni en audio et le joue instantanément.

        Args:
            text (str): Le texte à convertir en audio.
            engine (str): Le moteur TTS à utiliser pour jouer le fichier audio.
                         Par défaut, 'aplay' sous Linux, 'System.Media.SoundPlayer' sous Windows et 'say' sous macOS.
                         
        Returns:
            None
        """
        
        # Vérifie si la machine est supportée
        if self._get_os() == 'unsupported_machine':
            print("Cette plateforme n'est pas supportée pour la conversion texte-to-speech.")
            return
        
        # Définit le répertoire de cache
        cache_dir = os.path.join(self.parent_destn, 'cache')
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)  # Supprime le cache existant
            os.mkdir(cache_dir)  # Recréé le dossier cache

        # Génère le fichier audio à partir du texte
        filepath = self.text_2_audio_file(text, 'main', directory=cache_dir)
        
        # Traitement selon le système d'exploitation
        if platform.system() == 'Linux':
            # Si aucun moteur n'est spécifié, utiliser 'aplay' par défaut
            if engine is None:
                engine = 'aplay'
            
            # Si 'aplay' est choisi, on l'exécute
            if engine == 'aplay':
                os.system(f'{engine} {filepath}')
            else:
                print(f"Le moteur '{engine}' n'est pas reconnu pour Linux.")
        
        elif platform.system() == 'Windows':
            # Le moteur Windows par défaut est 'System.Media.SoundPlayer'
            if engine is None:
                engine = 'System.Media.SoundPlayer'
            
            # Si le moteur est 'System.Media.SoundPlayer', utilise PowerShell pour jouer le fichier
            if engine == 'System.Media.SoundPlayer':
                cmd = f"""powershell (New-Object {engine} {filepath}).PlaySync()"""
                os.system(cmd)
            else:
                print(f"Le moteur '{engine}' n'est pas reconnu pour Windows.")
        
        elif platform.system() == 'Darwin':  # macOS
            # Utiliser le moteur 'say' pour macOS
            if engine is None:
                engine = 'say'
            
            # Si le moteur est 'say', utilise le terminal pour le TTS
            if engine == 'say':
                os.system(f"say {text}")
            else:
                print(f"Le moteur '{engine}' n'est pas reconnu pour macOS.")
        
        else:
            print(f"La plateforme '{platform.system()}' n'est pas supportée pour la conversion texte-to-speech.")
    
    def text_2_audio_file(self, text, model, directory):
        """
        Fonction simulée pour générer un fichier audio à partir du texte.
        Renvoie le chemin du fichier audio généré.
        """
        audio_file = os.path.join(directory, 'output_audio.wav')
        # Normalement, ici vous utiliseriez un modèle TTS pour générer le fichier audio à partir du texte
        # Pour l'instant, on suppose que le fichier audio est créé correctement
        print(f"Fichier audio généré à {audio_file}")
        return audio_file

# # Point d'entrée du programme
# if __name__ == '__main__':
#     parent_destn = "/chemin/vers/le/dossier"  # Remplacez par le chemin où vous souhaitez stocker le cache
#     dt = Dimits(parent_destn)

#     # Test de la fonction text-to-speech
#     dt.text_2_speech("Joshua is the king", "say")  # Utilise le moteur 'say' pour macOS