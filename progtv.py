# import libraries
import requests
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from datetime import datetime

class TVProgram():
    def __init__(self):
        self.downloading_url = "https://daga123-tv-api.onrender.com/getPrograms"
        self.download_folder = "program_download"
        self.channels = ['TF1', 'France 2', 'France 3', 'Canal+', 'France 5', 'M6', 'Arte',
       'C8', 'W9', 'TMC', 'TFX', 'NRJ12', 'LCP', 'France 4', 'Gulli', 'TF1 Séries-Films',
       'La chaine l’Équipe', '6ter', 'RMC STORY', 'RMC Découverte',
       'Chérie 25', 'Paris Première']

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
            df.to_excel(f"{self.download_folder}/progtv_{datetime.now().today().strftime('%Y-%m-%d')}.xlsx", index=False)
            return df
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la récupération des données : {e}")
            return None
    
    def read_programs(self, file):
        """Lire les données des programmes TV"""
        try:
            # Charger le fichier Excel
            df = pd.read_excel(file)
            return df
        except FileNotFoundError:
            print(f"Le fichier {file} est introuvable.")
            return None
        
    def filter_programs(self, df, channels):
        """Filtrer les programmes TV par chaîne"""
        try:
            # Filtrer les données
            filtered_df = df[df["name"].isin(channels)]
            return filtered_df
        except KeyError:
            print("Le DataFrame ne contient pas de colonne 'name'.")
            return None

if __name__ == "__main__":
    # Créer une instance de la classe TVProgram
    tv_program = TVProgram()
    # Récupérer les données
    # progs = tv_program.get_programs(tv_program.downloading_url)
    file_name = f"{tv_program.download_folder}/progtv_{datetime.now().today().strftime('%Y-%m-%d')}.xlsx"
    progs = tv_program.read_programs(file_name)
    progs_filtered = tv_program.filter_programs(progs, tv_program.channels)
    print(progs_filtered["name"].unique())
