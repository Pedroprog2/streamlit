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




# Subtítulo para a seção de upload
st.subheader('Upload das imagens')
st.write("Este aplicativo usa OpenCV para processar imagens. Você pode carregar as imagens em formato .png.")

# Botão para upload de imagem
uploaded_files = st.file_uploader("Escolha uma imagem...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Lista para armazenar os vetores concatenados
vetores_concatenados = []

# Verifica se o arquivo foi enviado
if uploaded_files is None:
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





        
            # Criar um gráfico
            # Criação de um gráfico 3D

            #plt.figure()
            #plt.plot(vetor_concatenado)
            #plt.title('Histograma da Imagem')
            #plt.xlabel('Bins')
            #plt.ylabel('Frequência')
            
            # Exibir o gráfico
            #st.pyplot(plt)
            
            #st.write("Histograma da Imagem:", vetor_concatenado.T)
            #st.write("Tamanho do vetor da imagem:", vetor_concatenado.shape)

        # Converter a lista em uma matriz numpy
matriz_histogramas = np.array(vetores_concatenados)

#https://github.com/Pedroprog2/blank-app-template-v1x6l39uxlg/572aa01eb6c4460f3416903e99b5178f0b03968f/TUCUNARE.npy
# Carregar a matriz de dados
data_file_path = 'https://raw.githubusercontent.com/Pedroprog2/streamlit/eff5b2eba6dee61ad39f42aa8e63182820bdf027/X.npy'  # Substitua pelo caminho do seu arquivo .npy
X = load_data_from_github(data_file_path)
st.write("dados carregados!")
#X = np.load(matriz)
st.write(X)



#matriz_X = 'https://raw.githubusercontent.com/Pedroprog2/streamlit/eff5b2eba6dee61ad39f42aa8e63182820bdf027/X.npy'
#matriz_X = 'https://github.com/Pedroprog2/streamlit/blob/59d192e2be917affe84e9d3ab2b14023d027b551/X.npy'
