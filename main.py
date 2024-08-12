# Importando pacotes necessários
import streamlit as st
import cv2
import numpy as np
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
    
    # Mostrar a imagem
    st.image(image, caption='Imagem Original', use_column_width=True)

    # Coordenadas do recorte (você pode ajustar essas coordenadas para cada imagem)
    x, y, w, h = 0, 0, 20, 20

    # Realizar o recorte
    cropped_image = image[y:y + h, x:x + w]
    st.image(cropped_image, caption='Imagem Recortada', use_column_width=True)

    # Separar os canais de cores (B, G, R)
    canal_azul = cropped_image[:, :, 0]
    canal_verde = cropped_image[:, :, 1]
    canal_vermelho = cropped_image[:, :, 2]

    # Calcular os histogramas
    hist_azul = cv2.calcHist([canal_azul], [0], None, [256], [0, 256])
    hist_verde = cv2.calcHist([canal_verde], [0], None, [256], [0, 256])
    hist_vermelho = cv2.calcHist([canal_vermelho], [0], None, [256], [0, 256])

    # Concatenar os histogramas em um único vetor
    vetor_concatenado = np.concatenate((hist_azul, hist_verde, hist_vermelho), axis=None)
    return vetor_concatenado

st.title('Análise do pH do solo via imagens')
st.write('Bem-vind@!')
st.write('Abaixo você poderá enviar sua imagem da solução extraída do solo com a adição de extrato de repolho roxo')
# Subtítulo para a seção de upload
st.subheader('Upload das imagens')
st.write("Você pode carregar as imagens em formato .png, .jpg ou .jpeg.")

# Botão para upload de imagem
uploaded_files = st.file_uploader("Escolha uma imagem...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Lista para armazenar os vetores concatenados
vetores_concatenados = []

# Processar cada imagem carregada
for uploaded_file in uploaded_files:
    # Ler a imagem
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

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
