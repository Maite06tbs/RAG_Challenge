[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![Built with LangChain](https://img.shields.io/badge/LangChain-OpenAI-blue.svg)](https://github.com/langchain-ai/langchain)  
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)](https://streamlit.io/)  

# 🖼️ Image2Answer  

**Transform visual content into insights — posez, analysez et obtenez des réponses précises en quelques secondes.**

Ce projet a été réalisé à l’issue d’un workshop sur le workshop **🎉 Retrieval-Augmented Generation (RAG)**. L’objectif : implémenter notre propre système basé sur RAG.

**Image2Answer** est une solution qui permet à l’utilisateur de charger une image contenant du texte (comme un document scanné, une photo de notes ou une image médicale), et de poser des questions dessus pour obtenir des réponses intelligentes basées sur le contenu lu. 

---
**🎨 Exemples d’utilisation**

📚 Étudiant
Prendre une photo de ses notes manuscrites et demander :
“Résume-moi cette partie en 3 points.”

🧾 Administration
Télécharger une copie d’un justificatif administratif et demander :
“Quelle est la date limite de validité ?”

---
## 🚀 Fonctionnalités clés

1. **Upload Multi‑format**  
   - PDF, PNG, JPG/JPEG  

2. **OCR et Extraction de Texte**  
   - Traitement d’images avec Tesseract OCR (`pytesseract`, `Pillow`).  
   - Découpage et vectorisation de documents PDF avec LangChain.

3. **Base Vectorielle FAISS**  
   - Création automatique d’un index FAISS.  
   - Suppression et régénération de l’index à chaque nouvelle analyse.

4. **Chaîne RAG avec LangChain & OpenAI**  
   - Récupération contextuelle des passages les plus pertinents.  
   - Génération de réponses naturelles via l’API ChatOpenAI.  

5. **Interface Streamlit**  
   - Sidebar ergonomique pour uploader et lancer le traitement.  
   - Zone de chat en streaming pour poser des questions et afficher les réponses.  

---

## 📦 Installation

### Prérequis

- **Python ≥ 3.10**  
- **Tesseract OCR** (moteur natif, à installer via votre gestionnaire de paquets)  

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install tesseract-ocr -y

# Fedora
sudo dnf install tesseract -y

### Gestionnaire de paquets UV

UV est un installateur et un résolveur de paquets Python rapide et fiable. Installez-le avec :

```shell
# Windows (PowerShell)
curl -sSf https://install.python-uv.org/install.ps1 | powershell

# macOS et Linux
curl -sSf https://install.python-uv.org/install.sh | bash
```

Vérifier l'installation :
```shell
uv --version
```

## Configuration de l'environnement

Cloner le dépôt et installer les dépendances :

```shell

# Créer et activer un environnement virtuel
python -m venv venv

# Sous Windows
venv\Scripts\activate

# Sous macOS/Linux
source venv/bin/activate

# Installer les dépendances avec UV
uv pip install -r requirements.txt
```

### Configuration des clés API

1. Créer un fichier `.env` à la racine du projet
2. Ajouter vos clés API :
```
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE=https://api.openai.com/v1 # Point de terminaison OpenAI par défaut
MODÈLE_NOM=gpt-3.5-turbo # ou gpt-4
```


**Image2Answer est un assistant IA capable de lire et comprendre n'importe quelle image contenant du texte. Posez-lui des questions, il vous répond en s'appuyant sur ce qu'il a "lu" dans l'image.** *Amusez-vous!*