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

# Configurações iniciais
st.set_page_config(
    page_title="Plataforma de pH do Solo 🌱",
    page_icon="🌱",
    layout="wide",
)




# Define o CSS diretamente em uma string
css = """
<style>
body {
    background-color: #964B00; /* Cor de fundo desejada */    
}
</style>
"""

st.markdown(
    """
    <style>
    html, body {
        background-color: #000000; /* Cor preta */
        color: white; /* Texto branco para melhor contraste */
    }
    .stApp {
        max-width: 80%;
        margin: auto;
        background-color: #000000; /* Também define fundo preto no conteúdo */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# CSS para ajustar a posição da imagem
st.markdown(
    """
    <style>
    .left-aligned-image {
        display: flex;
        justify-content: flex-start;
    }
    .left-aligned-image img {
        max-width: 30%; /* Ajusta a largura para responsividade */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Insere a imagem
st.markdown(
    """
    <div class="left-aligned-image">
        <img src="https://portalpadrao.ufma.br/site/noticias/semic-e-semiti-celebracao-da-iniciacao-cientifica-tecnologica-e-de-inovacao-sera-realizada-de-02-a-06-de-dezembro-na-ufma/2024-11-26-semic-e-semiti-1.jpeg/@@images/0d7016bb-0dcf-4860-ba42-93e6137d2a8d.jpeg" alt="Minha Imagem">
    </div>
    """,
    unsafe_allow_html=True
)


# Função para carregar dados do GitHub
def load_data_from_github(url):
    response = requests.get(url)
    response.raise_for_status()  # Verifica se a requisição foi bem-sucedida
    return np.load(BytesIO(response.content))

# Treinar e testar o modelo com diferentes componentes PLS
def treinar_e_testar_pls(n_components, X_train, y_train, X_test):
    pls = PLSRegression(n_components=n_components)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    pls.fit(X_train_scaled, y_train)
    y_pred_train = pls.predict(X_train_scaled)
    y_pred_test = pls.predict(X_test_scaled)
    #mse_train = mean_squared_error(y_train, y_pred_train)
    #mse_test = mean_squared_error(y_test, y_pred_test)
    return  y_pred_train, y_pred_test

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
        # Separar os canais de cores (B, G, R)
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
st.title('Análise de pH do solo por meio de imagens digitais e repolho roxo')
st.caption("Atualizado em 03/12/2024")
st.info("Bem-vind@ ao SEMIC! Utilize a plataforma para calcular o pH do solo de maneira prática e inovadora.")

# Sidebar para orientações
with st.sidebar:
    st.header("📋 Orientações")
    st.write("Instruções detalhadas para o preparo das amostras e do extrato de repolho roxo.")
    st.subheader("Links Úteis")
    st.write("[Acesse imagens de exemplo](https://drive.google.com/drive/folders/10YKwXpJL8zCH5HXVxd3VaTD6QWNGtB_e?usp=sharing)")
    st.write("Dúvidas: morais.pedro@ufma.br")



#st.subheader('Orientações')
st.header("Orientações", divider="gray")
st.write('- O intuito desta plataforma é determinar o pH do solo utilizando o extrato de repolho roxo. Atente-se as orientações abaixo:')
st.subheader('Preparo da amostra', divider="gray")
st.write('- A amostra de solo deve estar seca, moída e peneirada à 2 mm.')
st.write('- Pese 10 g de solo e adicione 25 mL de solução de CaCl2 0,01 mol/L. Agite utilizando um bastão de vidro.')
st.write('- Aguarde uma hora. Agite novamente e aguarde a decantação do solo.')
st.write('- Após a decantação, colete 5 mL do sobrenadante e transfira para um tubo de vidro.')
st.write('- Guarde os tubos em local adequado enquanto prepara o extrato de repolho roxo.')

st.subheader('Preparo do extrato de repolho roxo', divider="gray")
st.write('- Pese 25 gramas de repolho roxo, previamente lavados. Em seguida, adicione 100 mL de água destilada. Aqueça até a fervura.')
st.write('- Após o resfriamento, filtre. O extrato está pronto!')


st.subheader('Utilizando a plataforma', divider="gray")
st.write('- Adicione 2 mL de extrato de repolho roxo em cada tubo. Agite!')
st.write('- Observe a mudança de cor')
st.write('- Obtenha uma imagem digital utilizando seu celular')
st.write('- Envie a imagem neste servidor')
st.write('- A Plataforma vai carregar sua imagem, selecione a região do tubo que contenha sua amostra')
st.write('- O valor do pH da sua amostra será calculado automaticamente pelo sistema, o resultado será gerado abaixo (após as imagens).')
st.write(' - Como você já sabe, o valor do pH varia na escala de 0 a 14, caso a sua amostra esteja fora desta faixa, utilize um recorte menor da imagem e tente novamente')
st.write('- Neste link você pode acessar algumas imagens de tubos contendo as soluções do solo e o extrato de repolho roxo: https://drive.google.com/drive/folders/10YKwXpJL8zCH5HXVxd3VaTD6QWNGtB_e?usp=sharing')
st.write(' - Em caso de problemas ou dúvidas, escreva para: morais.pedro@ufma.br')

        






# Botão para upload de imagem
st.subheader("📤 Upload de Imagens")
st.write("Você pode carregar as imagens em formato .png, .jpg ou .jpeg.")
uploaded_files = st.file_uploader("Escolha uma imagem...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Lista para armazenar os vetores concatenados
vetores_concatenados = []



# Processar cada imagem carregada
if uploaded_files:
    for uploaded_file in uploaded_files:
    # Ler a imagem usando o PIL
        image = Image.open(uploaded_file)

    # Processar a imagem e gerar vetor de histogramas
        vetor_histogramas = process_image(image)
        if vetor_histogramas is not None:
            vetores_concatenados.append(vetor_histogramas)

# Converter a lista em uma matriz numpy
#matriz_histogramas = np.array(vetores_concatenados)


    # Converter a lista em uma matriz numpy
    matriz_histogramas = np.array(vetores_concatenados)

    # Carregar a matriz de dados
    data_file_path = 'https://raw.githubusercontent.com/Pedroprog2/streamlit/eff5b2eba6dee61ad39f42aa8e63182820bdf027/X.npy'
    X = load_data_from_github(data_file_path)
    X_train = X[0:15,:]
    st.write("Dados carregados!")

    # Definir uma variável y para fins de exemplo (substitua com seus próprios dados)
    y = np.array([7.57, 3.68, 6.51, 7.98, 9.4, 6.49, 3.23, 6.65, 7.66, 7.56, 7.76, 7.73, 7.34, 7.51, 7.00])
    
    # Chamada da função de treinamento e teste do PLS
    try:
        y_pred_train_2, y_pred_test_2 = treinar_e_testar_pls(4, X_train, y, matriz_histogramas)
        #st.write(f"MSE Train: {y_pred_train_2}")
        st.success(f"Resultado do pH da sua amostra: {y_pred_test_2}")

   
    except ValueError as e:
        st.error(f"Erro ao treinar o modelo: {e}")
else:
    st.write("Por favor, carregue uma imagem para processamento.")
