# Importando pacotes necessários
import streamlit as st
import numpy as np
from streamlit_cropper import st_cropper
from PIL import Image
import requests
from io import BytesIO
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Função para carregar dados do GitHub
def load_data_from_github(url):
    response = requests.get(url)
    response.raise_for_status()  # Verifica se a requisição foi bem-sucedida
    return np.load(BytesIO(response.content))

# Treinar e testar o modelo com diferentes componentes PLS
def treinar_e_testar_pls(n_components, X_train, y_train, X_test, y_test):
    pls = PLSRegression(n_components=n_components)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    pls.fit(X_train_scaled, y_train)
    y_pred_train = pls.predict(X_train_scaled)
    y_pred_test = pls.predict(X_test_scaled)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    return mse_train, mse_test, y_pred_train, y_pred_test

# Processar imagem para gerar histogramas
def process_image(image):
    if image is None:
        st.error("Erro ao carregar a imagem.")
        return None

    # Mostrar a imagem original
    st.image(image, caption='Imagem Original', use_column_width=True)

    # Usar o streamlit-cropper para permitir ao usuário selecionar a área de interesse
    cropped_image = st_cropper(image, realtime_update=True, box_color='blue', aspect_ratio=None)

    if cropped_image is not None:
        st.image(cropped_image, caption='Imagem Recortada', use_column_width=True)

        # Separar os canais de cores (B, G, R)
        cropped_image = np.array(cropped_image)
        canal_azul = cropped_image[:, :, 0]
        canal_verde = cropped_image[:, :, 1]
        canal_vermelho = cropped_image[:, :, 2]

        # Calcular os histogramas
        hist_azul = np.histogram(canal_azul, bins=256, range=(0, 256))[0]
        hist_verde = np.histogram(canal_verde, bins=256, range=(0, 256))[0]
        hist_vermelho = np.histogram(canal_vermelho, bins=256, range=(0, 256))[0]

        # Concatenar os histogramas em um único vetor
        vetor_concatenado = np.concatenate((hist_azul, hist_verde, hist_vermelho), axis=None)
        return vetor_concatenado

# Subtítulo para a seção de upload
st.subheader('Upload das imagens')
st.write("Este aplicativo usa OpenCV para processar imagens. Você pode carregar as imagens em formato .png, .jpg ou .jpeg.")

# Botão para upload de imagem
uploaded_files = st.file_uploader("Escolha uma imagem...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Lista para armazenar os vetores concatenados
vetores_concatenados = []

# Processar cada imagem carregada
for uploaded_file in uploaded_files:
    # Ler a imagem usando o PIL
    image = Image.open(uploaded_file)

    # Processar a imagem e gerar vetor de histogramas
    vetor_histogramas = process_image(image)
    if vetor_histogramas is not None:
        vetores_concatenados.append(vetor_histogramas)

# Converter a lista em uma matriz numpy
matriz_histogramas = np.array(vetores_concatenados)

# Carregar a matriz de dados
data_file_path = 'https://raw.githubusercontent.com/Pedroprog2/streamlit/eff5b2eba6dee61ad39f42aa8e63182820bdf027/X.npy'  # Substitua pelo caminho do seu arquivo .npy
X = load_data_from_github(data_file_path)
st.write("Dados carregados!")
st.write(X)
