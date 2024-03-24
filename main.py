import streamlit as st
from PIL import Image
import numpy as np

def contar_digitos_finais(imagem):
    # Abrir a imagem e converter para escala de cinza
    img = Image.open(imagem).convert('L')
    # Converter para array numpy
    img_array = np.array(img)
   

    # Contar o número de pixels com cada dígito final
    contagem_digitos = [0] * 10  # Inicializar a lista de contagem com zeros
    for linha in img_array:
        for pixel in linha:
            primeiro_digito = int(str(pixel)[0])  # Extrair o primeiro dígito convertendo o pixel em uma string
            if 0 <= primeiro_digito <= 9:  # Verificar se o primeiro dígito é um valor válido
                contagem_digitos[primeiro_digito] += 1
    
    return contagem_digitos

# Configurações da página
st.title('Introdução à estatística - CCCh - UFMA')

st.title('Contador de pixels em imagens')

st.write('Faça upload de uma imagem para contar o número de pixels com o dígito final entre 0 a 9.')

# Upload da imagem
imagem = st.file_uploader('Escolha uma imagem', type=['jpg', 'png'])

if imagem is not None:
    # Exibir a imagem
    st.image(imagem, caption='Imagem enviada', use_column_width=True)
    
    # Processar a imagem e contar os dígitos finais
    contagem_digitos = contar_digitos_finais(imagem)
    
    # Preparar os dados para o gráfico de barras
    dados_grafico = {'Dígito': list(range(10)), 'Número de pixels': contagem_digitos}
    
    # Plotar o gráfico de barras
    st.bar_chart(dados_grafico)
