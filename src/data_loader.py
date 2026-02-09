import pandas as pd
import zipfile
import glob
import os

def load_data():
    """
    Charge tous les CSV contenus dans les fichiers ZIP de data/raw,
    ignore les fichiers __MACOSX et les fichiers commençant par ._
    """
    # Dossier data/raw
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, "data", "raw")

    # Cherche tous les ZIP
    zip_files = glob.glob(os.path.join(data_path, "*.zip"))
    if not zip_files:
        raise ValueError(f"Aucun fichier ZIP trouvé dans {data_path}")

    df_list = []

    for zip_file in zip_files:
        print(f"Ouverture du ZIP : {zip_file}")
        with zipfile.ZipFile(zip_file, 'r') as z:
            for file_name in z.namelist():
                # On ignore __MACOSX et les fichiers commençant par ._
                if "__MACOSX" in file_name or os.path.basename(file_name).startswith("._"):
                    continue
                if file_name.endswith(".csv"):
                    print(f"  Lecture du CSV : {file_name}")
                    with z.open(file_name) as f:
                        df = pd.read_csv(f)
                        df_list.append(df)

    if not df_list:
        raise ValueError("Aucun CSV valide trouvé dans les ZIP")

    df = pd.concat(df_list, ignore_index=True)

    # Convertir la date
    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
    df["date"] = df["started_at"].dt.date
    df["hour"] = df["started_at"].dt.hour

    return df


def aggregate_hourly(df):
    hourly = (
        df.groupby(["date", "hour"])
        .size()
        .reset_index(name="demand")
    )
    hourly["datetime"] = pd.to_datetime(hourly["date"]) + pd.to_timedelta(hourly["hour"], unit="h")
    hourly = hourly.sort_values("datetime")
    return hourly
