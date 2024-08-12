# Importando pacotes necessários
import streamlit as st
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
#from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
#from sklearn.metrics import mean_squared_error, make_scorer, r2_score
#from sklearn.preprocessing import StandardScaler
#from sklearn.svm import SVR
from scipy.stats import ttest_rel, f_oneway

# Variáveis globais
vetores_concatenados = []  # Lista para armazenar os vetores de histogramas

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

# Processar pasta de imagens
def process_folder(folder_path):
    global vetores_concatenados

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(folder_path, filename)
            st.write(f"Processando: {filename}")
            image = cv2.imread(full_path)
            if image is not None:
                vetor_concatenado = process_image(image)
                if vetor_concatenado is not None:
                    vetores_concatenados.append(vetor_concatenado)
            else:
                st.error(f"Erro ao carregar a imagem: {filename}")

    # Converter a lista em uma matriz numpy
    if vetores_concatenados:
        matriz_histogramas = np.array(vetores_concatenados)
        st.write("Matriz de histogramas:")
        st.write(matriz_histogramas)
        st.write("Formato da matriz:", matriz_histogramas.shape)

        # Processamento estatístico
        # (Continuar o processamento conforme necessário)

    else:
        st.warning("Nenhum histograma para mostrar.")

# Configurar a interface do Streamlit
st.title("Análise de pH do solo via imagens")
folder_path = st.text_input("Caminho para a pasta com as imagens")

if st.button("Processar"):
    if os.path.isdir(folder_path):
        process_folder(folder_path)
    else:
        st.error("Caminho para a pasta inválido.")



matriz_X = https://github.com/Pedroprog2/streamlit/blob/59d192e2be917affe84e9d3ab2b14023d027b551/X.npy
