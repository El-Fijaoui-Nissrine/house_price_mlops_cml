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


# Copier tout le code
COPY . .

# Exposer le port pour l'API
EXPOSE 8000

# Commande par défaut : lancer l'API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
