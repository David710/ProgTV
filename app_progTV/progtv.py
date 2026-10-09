# import libraries
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import pickle
import argparse
import unicodedata
from functools import lru_cache
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Correction demandée pour les horaires actuellement fournis par la source TV.
SOURCE_TIME_OFFSET = pd.Timedelta(hours=-2)
TIME_CORRECTION_ATTRIBUTE = 'progtv_time_offset_seconds'

# Ordre éditorial du guide : grandes chaînes nationales, puis TNT.
CHANNEL_ORDER = (
    'TF1', 'France 2', 'France 3', 'Canal+', 'France 5', 'M6', 'Arte',
    'C8', 'W9', 'TMC', 'TFX', 'NRJ12', 'LCP', 'France 4', 'BFMTV',
    'CNews', 'LCI', 'franceinfo', 'Gulli', 'TF1 Séries Films',
    'La chaîne L’Équipe', '6ter', 'RMC Story', 'RMC Découverte',
    'Chérie 25', 'Paris Première',
)


def channel_sort_key(name):
    """Accepter les variantes de casse, accents, espaces et ponctuation."""
    def normalize(value):
        text = unicodedata.normalize('NFKD', str(value).casefold())
        return ''.join(char for char in text
                       if char.isalnum() and not unicodedata.combining(char))

    normalized = normalize(name)
    aliases = {'lequipe': 'lachainelequipe', 'lcpassembleenationale': 'lcp',
               'lcppublicsenat': 'lcp', 'publicsenat': 'lcp'}
    normalized = aliases.get(normalized, normalized)
    order = {normalize(channel): index
             for index, channel in enumerate(CHANNEL_ORDER)}
    return order.get(normalized, len(CHANNEL_ORDER)), normalized


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
            return self.correct_cached_times(df)
        except FileNotFoundError:
            print(f"Le fichier {file} est introuvable.")
            return None
        
    @staticmethod
    def correct_cached_times(data):
        """Corriger les caches historiques une seule fois, sans toucher aux scores."""
        if data.attrs.get(TIME_CORRECTION_ATTRIBUTE) == -7200:
            return data
        data = data.copy()
        if 'programs' not in data:
            return data
        for index, row in data.iterrows():
            programs = row['programs']
            # Un cache brut de l’API sera corrigé par format_programs.
            if not isinstance(programs, pd.DataFrame):
                continue
            programs = programs.copy()
            if programs.attrs.get(TIME_CORRECTION_ATTRIBUTE) != -7200:
                for column in ('start', 'end'):
                    if column in programs:
                        programs[column] = (
                            pd.to_datetime(programs[column], utc=True)
                            + SOURCE_TIME_OFFSET
                        ).dt.tz_convert('Europe/Paris')
                programs.attrs[TIME_CORRECTION_ATTRIBUTE] = -7200
            data.at[index, 'programs'] = programs
        return data

    def format_programs(self, programs):
        """Formater les programmes TV
        - transformer la colonne 'programs' en DataFrame
        - Convertir les dates de début et de fin en datetime
        """
        df_programs = pd.DataFrame(programs)
        for column in ('start', 'end'):
            df_programs[column] = (
                pd.to_datetime(df_programs[column], unit='s', utc=True)
                + SOURCE_TIME_OFFSET
            ).dt.tz_convert('Europe/Paris')
        df_programs.attrs[TIME_CORRECTION_ATTRIBUTE] = -7200
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
            filtered_df.attrs[TIME_CORRECTION_ATTRIBUTE] = -7200
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

        generate_embeddings = lru_cache(maxsize=8192)(generate_embeddings)

        def flow_through_programs(program):
            print(f"Calcul des embeddings : {len(program)} programmes…", flush=True)
            program[f'embeddings_{model_name}'] = program["desc"].fillna("").astype(str).apply(generate_embeddings)
            return program
        
        # Appliquer la fonction à la colonne "desc"
        df["programs"] = df["programs"].apply(flow_through_programs)
        df.to_pickle(file_name)
        return df
    
    def training_file(self, file_name="df_programs_tf1_note.pkl"):
        path = Path(file_name)
        if path.is_absolute():
            candidates = [path]
        else:
            candidates = [self.train_folder / path, Path(__file__).resolve().parents[1] / path]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"Jeu annoté introuvable : {file_name}. Placez-le dans le dossier train ou à la racine de ProgTV."
        )

    def ensure_model(self):
        """Préparer un modèle compatible au premier lancement sans écraser l'historique."""
        model_path = self.train_folder / "trained_model_v2.pth"
        if not model_path.is_file() or not model_path.with_suffix(".preprocessing.pkl").is_file():
            print("Premier lancement : entraînement du modèle compatible à partir du jeu annoté…")
            self.train_model("df_programs_tf1_note.pkl")
        return self.load_model(model_path, self.input_dim)

    def train_model(self, file_name):
        df = pd.read_pickle(self.training_file(file_name))
        required = {"cat", "rating", "embeddings", "note"}
        if not required.issubset(df.columns) or len(df) < 5:
            raise ValueError("Jeu annoté invalide : au moins 5 lignes et colonnes cat, rating, embeddings, note requises.")

        # Les mappings et le normaliseur sont appris uniquement sur le train.
        X = np.vstack(df["embeddings"])
        y = pd.to_numeric(df['note'], errors="raise").to_numpy(dtype=float)
        if X.ndim != 2 or X.shape[0] != len(df) or not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Jeu annoté invalide : embeddings et notes numériques finis requis.")
        torch.manual_seed(42)

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
        model_file_path = self.train_folder / "trained_model_v2.pth"
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
        return self.select_prime_time(programs, day)

    @staticmethod
    def select_prime_time(programs, day):
        """Estimer le prime time par heure de début et durée, par chaîne."""
        early = pd.Timestamp(f'{day} 20:30', tz='Europe/Paris')
        prime = pd.Timestamp(f'{day} 21:00', tz='Europe/Paris')
        latest = pd.Timestamp(f'{day} 21:30', tz='Europe/Paris')
        duration = programs['duration']
        preferred = (
            programs['start'].between(prime, latest) & (duration >= 40)
        )
        fallback = (
            (programs['start'] >= early) & (programs['start'] < prime)
            & (duration >= 60) & (programs['end'] >= latest)
        )
        candidates = programs[preferred | fallback].copy()
        candidates['_prime_priority'] = preferred.loc[candidates.index].map(
            {True: 0, False: 1}
        )
        candidates = candidates.sort_values(
            ['_prime_priority', 'start', 'id']
        ).drop_duplicates('channel_name')
        candidates['_channel_order'] = candidates['channel_name'].map(channel_sort_key)
        return candidates.sort_values('_channel_order').drop(
            columns=['_prime_priority', '_channel_order']
        )

    @staticmethod
    def search_text(value):
        text = unicodedata.normalize("NFKD", str(value or "").casefold())
        return "".join(char for char in text if not unicodedata.combining(char))

    def select_programs(self, rated_progs, view="tonight", now=None, query="",
                        channel="", category="", max_duration=None, limit=5):
        """Retourner les résultats filtrés et les choix disponibles dans cette vue."""
        if view not in {"now", "tonight", "tomorrow", "suggestions"}:
            raise ValueError("Vue inconnue.")
        now = pd.Timestamp(now or datetime.now(ZoneInfo("Europe/Paris")))
        if now.tzinfo is None:
            raise ValueError("L'heure de référence doit contenir un fuseau.")
        now = now.tz_convert("Europe/Paris")
        programs = self.flatten_programs(rated_progs)
        selected_date = now.date()
        if view == "now":
            programs = programs[(programs["start"] <= now) & (programs["end"] > now)]
        elif view == "tonight":
            programs = self.select_prime_time(programs, selected_date)
        elif view == "tomorrow":
            selected_date += timedelta(days=1)
            start = pd.Timestamp(selected_date, tz="Europe/Paris")
            end = pd.Timestamp(selected_date + timedelta(days=1), tz="Europe/Paris")
            programs = programs[(programs["start"] >= start) & (programs["start"] < end)]
        else:
            programs = programs[programs["start"] > now]
        # Les choix restent disponibles même lorsque les filtres ne donnent rien.
        choices = {
            key: sorted(programs[column].dropna().astype(str).loc[lambda values: values != ""].unique())
            if column in programs else []
            for key, column in (("channels", "channel_name"), ("categories", "cat"))
        }
        choices['channels'].sort(key=channel_sort_key)
        if channel:
            programs = programs[programs["channel_name"] == channel]
        if category:
            programs = programs[programs.get("cat", pd.Series("", index=programs.index)) == category]
        if max_duration is not None:
            programs = programs[(programs["duration"] >= 0) & (programs["duration"] <= max_duration)]
        query = self.search_text(query.strip())
        if query:
            searchable = pd.Series("", index=programs.index)
            for column in ("name", "desc", "channel_name", "cat"):
                if column in programs:
                    searchable += " " + programs[column].fillna("").astype(str)
            programs = programs[searchable.map(self.search_text).str.contains(query, regex=False)]
        if view == "suggestions":
            programs = programs.sort_values(["note_pred", "start", "id"], ascending=[False, True, True])
            programs = programs.drop_duplicates("id").head(max(0, limit))
        else:
            programs = programs.assign(
                _channel_order=programs['channel_name'].map(channel_sort_key)
            ).sort_values(['_channel_order', 'start', 'id']).drop_duplicates('id')
            programs = programs.drop(columns='_channel_order')
        return programs, choices, selected_date

    def get_best_programs(self, rated_progs, n=5, whitelist=None, now=None):
        programs = self.flatten_programs(rated_progs)
        now = now or datetime.now(ZoneInfo("Europe/Paris"))
        programs = programs[programs["start"] > now]
        if whitelist:
            programs = programs[programs["channel_name"].isin(whitelist)]
        programs = programs.sort_values(["note_pred", "start", "id"], ascending=[False, True, True])
        return programs.drop_duplicates("id").head(max(0, n))

    def get_ollama_comment(self, program_desc, preferences=None,
                           program=None, reasons=None):
        from explanations import generate_explanation
        program = program or {'desc': program_desc}
        return generate_explanation(program, preferences=preferences, reasons=reasons)

    def add_ollama_comment_to_dataset(self, df):
        df = df.copy()
        df["ollama_comment"] = df["desc"].apply(self.get_ollama_comment)
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Préparer les programmes et leur classement")
    parser.add_argument("--train", metavar="FILE", help="Jeu annoté situé dans le dossier train")
    args = parser.parse_args()
    # Éviter la surallocation de threads sur les machines à nombreux cœurs.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    tv_program = TVProgram()
    if args.train:
        tv_program.train_model(args.train)
    else:
        # Valider le modèle avant de télécharger et de calculer les embeddings.
        model = tv_program.ensure_model()
        progs = tv_program.get_programs(tv_program.downloading_url)
        if progs is None:
            raise SystemExit("Téléchargement impossible ; les derniers programmes sont conservés.")
        progs_filtered = tv_program.filter_programs(progs, tv_program.channels)
        file_name = tv_program.download_folder / f"progtv_{datetime.now(ZoneInfo('Europe/Paris')).date()}.pkl"
        progs_filtered = tv_program.generate_embeddings(progs_filtered, "camembert", file_name)
        tv_program.rate_programs(model, progs_filtered, embedding_type="camembert")
