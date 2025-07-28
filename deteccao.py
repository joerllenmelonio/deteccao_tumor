import warnings
import cv2
from tensorflow.keras.utils import normalize
import os
import time
import numpy as np
import time

warnings.filterwarnings("ignore")
def diretorio(nome_path: any):
    DIRETORIO_BASE = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(DIRETORIO_BASE, nome_path)
    return path

def identificar_tumor(caminho_imagem, model):
    """
    Carrega uma imagem, a pré-processa e usa o modelo treinado para prever
    se ela contém um tumor.
    """
    try:
        # Carrega a imagem
        img = cv2.imread(caminho_imagem)
        if img is None:
            print(f"Erro: Não foi possível carregar a imagem em: {caminho_imagem}")
            return

        # Exibe a imagem original
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.title("Imagem para Análise")
        # plt.show()

        # normaliza imagem
        largura, altura=224,224
        img_processada = cv2.resize(img, (largura, altura))
        img_processada = cv2.cvtColor(img_processada, cv2.COLOR_BGR2RGB)
        img_processada = normalize(img_processada, axis=1)
        img_processada = img_processada.reshape(1, largura, altura, 3) # Adiciona dimensão de batch

        # Faz a predição
        print("\n--- Iniciando Predição ---")
        predicao = model.predict(img_processada)
        
        # Interpreta o resultado
        print("\n--- Resultado da Análise ---")
        probabilidade = predicao[0][0] * 100
        
        if predicao[0][0] > 0.5:
            print(f"Diagnóstico: Tumor Detectado (Probabilidade: {probabilidade:.2f}%)")
        else:
            print(f"Diagnóstico: Sem Tumor (Probabilidade de ser tumor: {probabilidade:.2f}%)")

    except Exception as e:
        print(f"Ocorreu um erro durante a identificação: {e}")
