# Importation des bibliothèques nécessaires
import sounddevice as sd  # Pour l'enregistrement audio
import soundfile as sf  # Pour la sauvegarde de l'audio
import numpy as np  # Pour les calculs numériques
import warnings  # Pour ignorer les avertissements
warnings.filterwarnings('ignore', category=UserWarning, module='whisper.*')
import whisper  # Pour la conversion de la parole en texte
import time  # Pour la gestion du temps
import ollama  # Pour le traitement du texte
import pyttsx3  # Pour la conversion du texte en parole
import subprocess  # Pour l'exécution de commandes système



# Définition des paramètres d'enregistrement
taux_echantillonnage = 44100  # Taux d'échantillonnage de l'audio
seuil_demarrage = 0.01  # Seuil de volume pour démarrer l'enregistrement
seuil_arret = 0.001  # Seuil de volume pour arrêter l'enregistrement

# Définition du nom du fichier de sortie
FICHIER_SORTIE = "audio_enregistre.wav"

def calculer_volume(audio):
    """Calculer le volume RMS de l'audio."""
    return np.sqrt(np.mean(np.square(audio)))  

def enregistrer_audio():
    """Enregistrer l'audio lorsque le volume dépasse un certain seuil."""
    audio_enregistre = []  # Liste pour stocker l'audio enregistré
    enregistrement = False  # Indicateur pour savoir si l'enregistrement est en cours
    dernier_son = None  # Temps du dernier son enregistré
    en_cours = True  # Indicateur pour savoir si l'enregistrement est toujours en cours

    def callback(indata, frames, time_info, status):
        """Fonction de rappel pour traiter l'audio en temps réel."""
        nonlocal enregistrement, dernier_son, en_cours
        volume = calculer_volume(indata)  # Calcul du volume de l'audio
        if volume > seuil_demarrage:  # Si le volume dépasse le seuil de démarrage
            if not enregistrement:  # Si l'enregistrement n'est pas encore commencé
                enregistrement = True  # Commencer l'enregistrement
                print("Enregistrement démarré...")
            dernier_son = time.time()  # Mettre à jour le temps du dernier son
            audio_enregistre.append(indata.copy())  # Ajouter l'audio à la liste
        elif enregistrement and dernier_son is not None and time.time() - dernier_son > 1:  # Si le volume est en dessous du seuil d'arrêt et que 1 seconde s'est écoulée depuis le dernier son
            enregistrement = False  # Arrêter l'enregistrement
            print("Enregistrement arrêté.")
            en_cours = False  # Indiquer que l'enregistrement n'est plus en cours
            raise sd.CallbackStop  # Arrêter la fonction de rappel

    with sd.InputStream(callback=callback, channels=1, samplerate=taux_echantillonnage):  # Créer un flux d'entrée audio avec la fonction de rappel
        while en_cours:  # Tant que l'enregistrement est en cours
            pass  # Ne rien faire

    if not audio_enregistre:  # Si aucun audio n'a été enregistré
        print("Aucun audio n'a été enregistré.")
        return np.array([])  # Retourner un tableau vide

    return np.concatenate(audio_enregistre, axis=0)  # Concaténer tous les audios enregistrés et les retourner

def sauvegarder_audio(donnees_audio, taux_echantillonnage, fichier_sortie):
    """Sauvegarder l'audio enregistré dans un fichier."""
    sf.write(fichier_sortie, donnees_audio, taux_echantillonnage)  # Écrire l'audio dans un fichier
    

def speech_to_text(fichier_entree):
    """Convertir l'audio en texte."""
    model = whisper.load_model("large")  # Charger le modèle de conversion de la parole en texte
    result = model.transcribe(fichier_entree)  # Transcrire l'audio en texte

    return result["text"]  # Retourner le texte

def traiter_texte_avec_ollama(texte):
    """Traiter le texte avec le modèle ollama."""
    stream = ollama.chat(  
        model='llama2',
        messages=[{'role': 'user', 'content': 'répond en francais a la requete suivante :' +  texte}],
        stream=True,
    )

    res = []  # Liste pour stocker les réponses
    for chunk in stream:  # Pour chaque réponse du modèle
        print(chunk['message']['content'], end='', flush=True)  # Imprimer la réponse
        res.append(chunk['message']['content'])  # Ajouter la réponse à la liste

    return res  # Retourner la liste des réponses

def text_to_speech(texte, fichier_sortie="out.wav"):
    """Convertir le texte en parole."""
    pyttsx3.speak(texte)  # Convertir le texte en parole

def main():
    """Fonction principale du script."""
    try:
        FICHIER_SORTIE_TTS = "out.wav"  
        donnees_audio = enregistrer_audio()  
        sauvegarder_audio(donnees_audio, taux_echantillonnage, FICHIER_SORTIE)  # Sauvegarder l'audio
        text = speech_to_text(FICHIER_SORTIE)  # Convertir l'audio en texte
        text = ''.join(e for e in text if e.isalnum() or e.isspace())  # Supprimer toute la ponctuation du texte
        print(text)
        mots = text.split()

        # Vérifiez si les deux premiers mots sont "ouvre" et "chrome"
        if len(mots) >= 2 and mots[0].lower() == "ouvre" and mots[1].lower() == "chrome":
            subprocess.run(['start', 'chrome'], shell=True, check=True)  # Exécutez Chrome
        else:
            reponse = traiter_texte_avec_ollama(text)  # Traiter le texte avec le modèle ollama

            reponse = ''.join(reponse)  
            text_to_speech(reponse, FICHIER_SORTIE_TTS)  # Convertir la réponse en parole
    except KeyboardInterrupt: 
        print("\nEnregistrement arrêté par l'utilisateur.")  

if __name__ == "__main__": 
    main()  