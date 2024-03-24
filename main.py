import streamlit as st
from PIL import Image
import numpy as np

def contar_digitos_iniciais(imagem):
    # Abrir a imagem e converter para escala de cinza
    img = Image.open(imagem).convert('L')
    # Converter para array numpy
    img_array = np.array(img)
    
    # Contar o número de pixels com cada dígito inicial
    contagem_digitos = [1] # Inicializar a lista de contagem com zeros
    for linha in img_array:
        for pixel in linha:
            primeiro_digito = int(str(pixel)[0])  # Extrair o primeiro dígito convertendo o pixel em uma string
            if 1 <= primeiro_digito <= 9:  # Verificar se o primeiro dígito é um valor válido entre 1 e 9
                contagem_digitos[primeiro_digito ] += 1  # Ajustar o índice para começar de 0
    
    return contagem_digitos

# Configurações da página
st.title('Introdução à estatística - CCCh - UFMA')

st.title('Contador de pixels em imagens')

st.write('Faça upload de uma imagem para contar o número de pixels com o dígito inicial entre 1 a 9.')

# Upload da imagem
imagem = st.file_uploader('Escolha uma imagem', type=['jpg', 'png'])

if imagem is not None:
    # Exibir a imagem
    st.image(imagem, caption='Imagem enviada', use_column_width=True)
    
    # Processar a imagem e contar os dígitos iniciais
    contagem_digitos = contar_digitos_iniciais(imagem)
    
    # Preparar os dados para o gráfico de barras
    dados_grafico = {'Dígito': list(range(1, 10)), 'Número de pixels': contagem_digitos}
    
    # Plotar o gráfico de barras
    st.bar_chart(dados_grafico)
