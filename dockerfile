# Dockerfile
FROM iterativeai/cml:0.17.3

# Créer le dossier app
WORKDIR /app

# Copier le fichier requirements et installer
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


# Copier tout le code
COPY . .

# Commande par défaut (utile pour debug)
CMD ["python", "-m", "src.train"]
