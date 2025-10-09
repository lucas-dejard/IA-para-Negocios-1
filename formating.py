import pandas as pd

# Caminho dos arquivos
input_path = "GMAT3_STK_PRICE_fixed.xlsx"
output_path = "GMAT3_STK_PRICE_formatted.csv"

# Lê o arquivo original
df = pd.read_excel(input_path)

# Renomeia colunas para o padrão do MicrosoftStock.csv
rename_map = {
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Price": "close",
    "Vol": "volume"
}

df.rename(columns=rename_map, inplace=True)

# Garante que apenas as colunas relevantes existam
cols = ["date", "open", "high", "low", "close", "volume"]
df = df[cols]

# Converte a data para o mesmo formato ISO (YYYY-MM-DD)
df["date"] = pd.to_datetime(df["date"], errors="coerce", infer_datetime_format=True)
df.dropna(subset=["date"], inplace=True)
df["date"] = df["date"].dt.strftime("%Y-%m-%d")

# Cria colunas adicionais para compatibilidade
df.insert(0, "index", range(len(df)))
df["Name"] = "GMAT3"

# Salva o arquivo no novo formato
df.to_csv(output_path, index=False)

print("✅ Arquivo formatado salvo em:", output_path)
print(df.head())
