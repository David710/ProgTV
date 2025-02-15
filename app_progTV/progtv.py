# import libraries
import requests
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from datetime import datetime
import json
from transformers import CamembertTokenizer, CamembertModel
import torch
from langchain_community.embeddings import OllamaEmbeddings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
        self.download_folder = "program_download"
        self.train_folder = "train"
        self.channels = ['TF1', 'France 2', 'France 3', 'Canal+', 'France 5', 'M6', 'Arte',
       'C8', 'W9', 'TMC', 'TFX', 'NRJ12', 'LCP', 'France 4', 'Gulli', 'TF1 Séries-Films',
       'La chaine l’Équipe', '6ter', 'RMC STORY', 'RMC Découverte',
       'Chérie 25', 'Paris Première']
        self.input_dim = 770

    def get_programs(self, url):
        """Récupérer les données des programmes TV"""
        try:
            # Envoyer une requête GET
            response = requests.get(url)
            
            # Vérifier si la requête a réussi (code 200)
            response.raise_for_status()
            
            # Charger le contenu JSON
            data = response.json()
            # Charger en DataFrame
            df = pd.DataFrame(data["data"])
            df.to_pickle(f"{self.download_folder}/progtv_{datetime.now().today().strftime('%Y-%m-%d')}.pkl")
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
        df_programs.start = pd.to_datetime(df_programs.start, unit="s")
        df_programs.end = pd.to_datetime(df_programs.end, unit="s")
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
            # Charger le tokenizer et le modèle
            tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
            model = CamembertModel.from_pretrained('camembert-base')

            # Fonction pour générer des embeddings
            def generate_embeddings(text):
                inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
                outputs = model(**inputs)
                return outputs.last_hidden_state.mean(dim=1).detach().numpy()
        elif model_name == "llama3":
            # Charger le modèle d'embeddings
            embed_model = OllamaEmbeddings(model="llama3:latest", show_progress=True)

            # Fonction pour générer des embeddings
            def generate_embeddings(text):
                return np.array(embed_model.embed_query(text))

        def flow_through_programs(program):
            program[f'embeddings_{model_name}'] = program["desc"].apply(generate_embeddings)
            return program
        
        # Appliquer la fonction à la colonne "desc"
        df["programs"] = df["programs"].apply(flow_through_programs)
        df.to_pickle(file_name)
        return df
    
    def train_model(self, file_name):
        # Préparer les données
        df = pd.read_pickle(self.train_folder + "/" + file_name)

        # Encoder la colonne "cat"
        label_encoder_cat = LabelEncoder()
        label_encoder_rating = LabelEncoder()
        df['cat_encoded'] = label_encoder_cat.fit_transform(df['cat'])
        df['rating_encoded'] = label_encoder_rating.fit_transform(df['rating'])

        # Séparer les caractéristiques (features) et la cible (target)
        X = np.hstack((df[['rating_encoded', 'cat_encoded']].values, np.vstack(df['embeddings'])))
        y = df['note'].values

        # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normaliser les caractéristiques
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convertir les données en tenseurs PyTorch
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        # Créer des DataLoader pour l'entraînement et le test
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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
        
        # Sauvegarder le modèle
        model_file_path = f"{self.train_folder}/trained_model.pth"
        self.save_model(model, model_file_path)
        return model

    def save_model(self, model, file_path):
        torch.save(model.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path, input_dim):
        """Charger le modèle sauvegardé"""
        model = NeuralNetwork(input_dim)
        model.load_state_dict(torch.load(file_path))
        model.eval()
        print(f"Model loaded from {file_path}")
        return model

    def rate_programs(self, model, progs_filtered, embedding_type):
        progs_filtered = progs_filtered.copy()
        for i, row in progs_filtered.iterrows():
            df = row['programs']

            # Encoder la colonne "cat"
            label_encoder_cat = LabelEncoder()
            label_encoder_rating = LabelEncoder()
            df['cat_encoded'] = label_encoder_cat.fit_transform(df['cat'])
            df['rating_encoded'] = label_encoder_rating.fit_transform(df['rating'])

            # Séparer les caractéristiques (features) et la cible (target)
            embedding_column = f'embeddings_{embedding_type}'
            X_new = np.hstack((df[['rating_encoded', 'cat_encoded']].values, np.vstack(df[embedding_column])))

            X_new_tensor = torch.tensor(X_new, dtype=torch.float32)
            model.eval()
            with torch.no_grad():
                y_pred_new = model(X_new_tensor)
                y_pred_new = y_pred_new.numpy()
                df['note_pred'] = y_pred_new
            progs_filtered.at[i, 'programs'] = df
        rated_programs_file_path = f"{tv_program.download_folder}/progtv_rated_{datetime.now().today().strftime('%Y-%m-%d')}.pkl"
        progs_filtered.to_pickle(rated_programs_file_path)
        return progs_filtered
    
    def get_prime_programs(self, rated_progs):
        today = datetime.now().today().strftime('%Y-%m-%d')
        prime_time_hour_min = datetime.strptime(f"{today} 20:50", '%Y-%m-%d %H:%M')
        prime_time_hour_max = datetime.strptime(f"{today} 21:30", '%Y-%m-%d %H:%M')
        prime_programs = pd.DataFrame()
        for i, row in rated_progs.iterrows():
            # Filter programs within the prime time range
            prime_time_progs = row.programs.loc[
                (row.programs["start"] > prime_time_hour_min) & 
                (row.programs["start"] < prime_time_hour_max)
            ]

            # Calculate the duration of each program
            prime_time_progs.loc[:, "duration"] = (prime_time_progs["end"] - prime_time_progs["start"]).dt.total_seconds() / 60

            # Find the program with the maximum duration
            prime_program = prime_time_progs.loc[prime_time_progs["duration"] == prime_time_progs["duration"].max()]
            prime_program.loc[:, "channel_name"] = row["name"]
            prime_program.loc[:, "channel_icon"] = row["icon"]
            prime_programs = pd.concat([prime_programs, prime_program])
        return prime_programs

if __name__ == "__main__":
    # Créer une instance de la classe TVProgram
    tv_program = TVProgram()
    # Récupérer les données
    # progs = tv_program.get_programs(tv_program.downloading_url)
    # file_name = f"{tv_program.download_folder}/progtv_{datetime.now().today().strftime('%Y-%m-%d')}.pkl"
    # progs = tv_program.read_programs(file_name)
    # progs_filtered = tv_program.filter_programs(progs, tv_program.channels)
    # progs_filtered = tv_program.generate_embeddings(progs_filtered, "camembert", file_name)
    # progs_filtered = tv_program.read_programs(file_name)
    # tv_program.train_model("df_programs_tf1_note.pkl")
    # model = tv_program.load_model("train/trained_model.pth", tv_program.input_dim)
    # rated_progs = tv_program.rate_programs(model, progs_filtered, embedding_type="camembert")
    file_name_rated = f"{tv_program.download_folder}/progtv_rated_{datetime.now().today().strftime('%Y-%m-%d')}.pkl"
    rated_progs = tv_program.read_programs(file_name_rated)
    # print(rated_progs.iloc[0]['programs'].head())
    # print(rated_progs.iloc[0]['programs'][rated_progs.iloc[1]['programs']["note_pred"] == rated_progs.iloc[1]['programs']["note_pred"].max()])
    # print(rated_progs.iloc[0]['programs'].iloc[68][["name", "cat", "note_pred", "desc"]])

    print(rated_progs.iloc[4  ]['programs'][['name', 'note_pred', 'desc']].sort_values(by='note_pred', ascending=False).head(20))