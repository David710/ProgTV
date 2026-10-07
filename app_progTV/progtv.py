# import libraries
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import pickle
import argparse
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Construire le modèle
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class TVProgram():
    def __init__(self):
        self.downloading_url = "https://daga123-tv-api.onrender.com/getPrograms"
        self.download_folder = Path(__file__).resolve().parent / "program_download"
        self.download_folder.mkdir(parents=True, exist_ok=True)
        self.train_folder = Path(__file__).resolve().parent / "train"
        self.channels = ['TF1', 'France 2', 'France 3', 'Canal+', 'France 5', 'M6', 'Arte',
       'C8', 'W9', 'TMC', 'TFX', 'NRJ12', 'LCP', 'France 4', 'Gulli', 'TF1 Séries-Films',
       'La chaine l’Équipe', '6ter', 'RMC STORY', 'RMC Découverte',
       'Chérie 25', 'Paris Première']
        self.input_dim = 770

    def get_programs(self, url):
        """Récupérer les données des programmes TV"""
        try:
            # Envoyer une requête GET
            response = requests.get(url, timeout=30)
            
            # Vérifier si la requête a réussi (code 200)
            response.raise_for_status()
            
            # Charger le contenu JSON
            data = response.json()
            # Charger en DataFrame
            df = pd.DataFrame(data["data"])
            df.to_pickle(f"{self.download_folder}/progtv_{datetime.now(ZoneInfo('Europe/Paris')).strftime('%Y-%m-%d')}.pkl")
            return df
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la récupération des données : {e}")
            return None
    
    def read_programs(self, file):
        """Lire les données des programmes TV"""
        try:
            # Charger le fichier Excel
            df = pd.read_pickle(file)
            return df
        except FileNotFoundError:
            print(f"Le fichier {file} est introuvable.")
            return None
        
    def format_programs(self, programs):
        """Formater les programmes TV
        - transformer la colonne 'programs' en DataFrame
        - Convertir les dates de début et de fin en datetime
        """
        df_programs = pd.DataFrame(programs)
        df_programs.start = pd.to_datetime(df_programs.start, unit="s", utc=True).dt.tz_convert("Europe/Paris")
        df_programs.end = pd.to_datetime(df_programs.end, unit="s", utc=True).dt.tz_convert("Europe/Paris")
        return df_programs
    
    def filter_programs(self, df, channels):
        """Filtrer les programmes TV par chaîne
        - Filtrer les données par colonne 'name'
        - Appliquer la fonction format_programs
        """
        try:
            # Filtrer les données
            filtered_df = df[df["name"].isin(channels)]
            filtered_df.loc[:, "programs"] = filtered_df["programs"].apply(self.format_programs)
            return filtered_df
        except KeyError:
            print("Le DataFrame ne contient pas de colonne 'name'.")
            return None
        
    def generate_embeddings(self, df, model_name, file_name):
        df = df.copy()
        if model_name == "camembert":
            from transformers import CamembertTokenizer, CamembertModel
            # Charger le tokenizer et le modèle
            tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
            model = CamembertModel.from_pretrained('camembert-base')
            model.eval()

            # Fonction pour générer des embeddings
            def generate_embeddings(text):
                inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1)
                return ((outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)).numpy()
        elif model_name == "llama3":
            from langchain_community.embeddings import OllamaEmbeddings
            # Charger le modèle d'embeddings
            embed_model = OllamaEmbeddings(model="llama3:latest", show_progress=True)

            # Fonction pour générer des embeddings
            def generate_embeddings(text):
                return np.array(embed_model.embed_query(text))
        else:
            raise ValueError(f"Type d’embedding inconnu : {model_name}")

        def flow_through_programs(program):
            program[f'embeddings_{model_name}'] = program["desc"].apply(generate_embeddings)
            return program
        
        # Appliquer la fonction à la colonne "desc"
        df["programs"] = df["programs"].apply(flow_through_programs)
        df.to_pickle(file_name)
        return df
    
    def train_model(self, file_name):
        # Préparer les données
        df = pd.read_pickle(self.train_folder / file_name)

        # Les mappings et le normaliseur sont appris uniquement sur le train.
        X = np.vstack(df["embeddings"])
        y = df['note'].values

        # Diviser les données en ensembles d'entraînement et de test
        train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
        self.category_maps = {
            column: {value: i for i, value in enumerate(sorted(df.iloc[train_idx][column].fillna("").astype(str).unique()))}
            for column in ("rating", "cat")
        }
        encoded = self.encode_categories(df)
        X = np.hstack((encoded, X))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Normaliser les caractéristiques
        scaler = StandardScaler()
        self.scaler = scaler
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convertir les données en tenseurs PyTorch
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        # Créer des DataLoader pour l'entraînement et le test
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        input_dim = X_train.shape[1]
        self.input_dim = input_dim
        model = NeuralNetwork(input_dim)

        # Définir la fonction de perte et l'optimiseur
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Entraîner le modèle
        num_epochs = 50
        for epoch in range(num_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        model.eval()
        with torch.no_grad():
            test_mse = criterion(model(X_test_tensor), y_test_tensor).item()
        print(f"Test MSE: {test_mse:.4f}")

        # Sauvegarder le modèle et son prétraitement
        model_file_path = f"{self.train_folder}/trained_model.pth"
        self.save_model(model, model_file_path)
        return model

    def save_model(self, model, file_path):
        torch.save({"state_dict": model.state_dict(), "input_dim": model.fc1.in_features}, file_path)
        with open(Path(file_path).with_suffix(".preprocessing.pkl"), "wb") as handle:
            pickle.dump({"category_maps": self.category_maps, "scaler": self.scaler}, handle)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path, input_dim):
        """Charger le modèle sauvegardé"""
        preprocessing_path = Path(file_path).with_suffix(".preprocessing.pkl")
        if not preprocessing_path.exists():
            raise ValueError("Modèle historique sans prétraitement : réentraînez le modèle avant de prédire.")
        with open(preprocessing_path, "rb") as handle:
            preprocessing = pickle.load(handle)
        self.category_maps = preprocessing["category_maps"]
        self.scaler = preprocessing["scaler"]
        checkpoint = torch.load(file_path, map_location="cpu", weights_only=True)
        model = NeuralNetwork(checkpoint["input_dim"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        print(f"Model loaded from {file_path}")
        return model

    def rate_programs(self, model, progs_filtered, embedding_type):
        progs_filtered = progs_filtered.copy()
        for i, row in progs_filtered.iterrows():
            df = row['programs'].copy()
            if df.empty:
                df['note_pred'] = pd.Series(dtype=float)
                progs_filtered.at[i, 'programs'] = df
                continue

            embedding_column = f'embeddings_{embedding_type}'
            X_new = np.hstack((self.encode_categories(df), np.vstack(df[embedding_column])))
            X_new = self.scaler.transform(X_new)

            X_new_tensor = torch.tensor(X_new, dtype=torch.float32)
            model.eval()
            with torch.no_grad():
                y_pred_new = model(X_new_tensor)
                y_pred_new = y_pred_new.numpy()
                df['note_pred'] = y_pred_new.ravel()
            progs_filtered.at[i, 'programs'] = df
        rated_programs_file_path = f"{self.download_folder}/progtv_rated_{datetime.now(ZoneInfo('Europe/Paris')).strftime('%Y-%m-%d')}.pkl"
        temporary_path = Path(rated_programs_file_path).with_suffix(".tmp")
        progs_filtered.to_pickle(temporary_path)
        temporary_path.replace(rated_programs_file_path)
        return progs_filtered
    
    def encode_categories(self, df):
        return np.column_stack([
            df[column].fillna("").astype(str).map(self.category_maps[column]).fillna(-1).to_numpy()
            for column in ("rating", "cat")
        ])

    def flatten_programs(self, rated_progs):
        frames = []
        for _, row in rated_progs.iterrows():
            programs = row["programs"].copy()
            if programs.empty:
                continue
            required = {"name", "start", "end", "note_pred"}
            if not required.issubset(programs.columns):
                raise ValueError("Cache TV incomplet : colonnes obligatoires absentes.")
            # Les caches historiques contiennent des dates UTC sans fuseau.
            for column in ("start", "end"):
                programs[column] = pd.to_datetime(programs[column], utc=True).dt.tz_convert("Europe/Paris")
            programs["duration"] = (programs["end"] - programs["start"]).dt.total_seconds() / 60
            programs["channel_name"] = row["name"]
            programs["channel_icon"] = row.get("icon", "")
            programs["id"] = [hashlib.sha256(f"{row['name']}|{start.isoformat()}|{name}".encode()).hexdigest()[:24]
                              for start, name in zip(programs["start"], programs["name"])]
            frames.append(programs)
        columns = ["id", "name", "start", "end", "icon", "rating", "cat", "desc", "note_pred", "duration", "channel_name", "channel_icon"]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)

    def get_prime_programs(self, rated_progs, day=None):
        programs = self.flatten_programs(rated_progs)
        day = day or datetime.now(ZoneInfo("Europe/Paris")).date()
        target = pd.Timestamp(f"{day} 21:00", tz="Europe/Paris")
        # Inclure les émissions déjà commencées au moment du prime time.
        candidates = programs[(programs["start"] <= target) & (programs["end"] > target)]
        return candidates.sort_values("start", ascending=False).drop_duplicates("channel_name")

    def get_best_programs(self, rated_progs, n=5, whitelist=None, now=None):
        if n <= 0:
            return self.flatten_programs(rated_progs).iloc[:0]
        programs = self.flatten_programs(rated_progs)
        now = now or datetime.now(ZoneInfo("Europe/Paris"))
        programs = programs[programs["start"] > now]
        if whitelist:
            programs = programs[programs["channel_name"].isin(whitelist)]
        programs = programs.sort_values(["note_pred", "start", "id"], ascending=[False, True, True])
        return programs.drop_duplicates("id").head(n)

    def get_ollama_comment(self, program_desc):
        import ollama
        response = ollama.Client(timeout=60).chat(
            # model="gemma3:12b",
            model="gemma3:12b-it-qat",
            messages=[
                {
                    "role": "user",
                    "content": f"j'aime les films d'action et les polars, j'aime également les émissions de cuisine, est ce que je vais aimer ce programme ?: {program_desc}, répond en français, fait un texte assez court de quelques lignes.", 
                },
            ],
        )
        return response["message"]["content"]
    
    def add_ollama_comment_to_dataset(self, df):
        df = df.copy()
        df["ollama_comment"] = df["desc"].apply(self.get_ollama_comment)
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Préparer les programmes et leur classement")
    parser.add_argument("--train", metavar="FILE", help="Jeu annoté situé dans le dossier train")
    args = parser.parse_args()
    tv_program = TVProgram()
    if args.train:
        tv_program.train_model(args.train)
    else:
        # Valider le modèle avant de télécharger et de calculer les embeddings.
        model = tv_program.load_model(tv_program.train_folder / "trained_model.pth", tv_program.input_dim)
        progs = tv_program.get_programs(tv_program.downloading_url)
        if progs is None:
            raise SystemExit("Téléchargement impossible ; les derniers programmes sont conservés.")
        progs_filtered = tv_program.filter_programs(progs, tv_program.channels)
        file_name = tv_program.download_folder / f"progtv_{datetime.now(ZoneInfo('Europe/Paris')).date()}.pkl"
        progs_filtered = tv_program.generate_embeddings(progs_filtered, "camembert", file_name)
        tv_program.rate_programs(model, progs_filtered, embedding_type="camembert")
