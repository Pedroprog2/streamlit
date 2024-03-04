import streamlit as st

def main():
    st.title("Minha Interface com Streamlit")
    st.sidebar.header("Menu")

    # Adicione as seções do menu
    selected_page = st.sidebar.radio("Selecione uma opção", ["Página Inicial", "Sobre"])

    if selected_page == "Página Inicial":
        show_homepage()
    elif selected_page == "Sobre":
        show_about()

def show_homepage():
    st.header("Bem-vindo à Página Inicial")
    st.write("Esta é a página inicial da minha interface.")
    
    # Adicione outros elementos, gráficos, etc.

def show_about():
    st.header("Sobre")
    st.write("Esta é uma breve descrição sobre a minha interface.")
    
    # Adicione outros elementos sobre a página 'Sobre'.

if __name__ == "__main__":
    main()
