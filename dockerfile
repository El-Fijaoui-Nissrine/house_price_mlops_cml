# Dockerfile
FROM python:3.11-slim

# Installer git et dépendances système si nécessaire
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Créer le dossier app
WORKDIR /app

# Copier le fichier requirements et installer
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install "cml>=0.0.1"

# Copier tout le code
COPY . .

# Commande par défaut (utile pour debug)
CMD ["python", "-m", "src.train"]
