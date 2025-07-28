import os
from flask import Flask
from deteccao import identificar_tumor, diretorio
from flask_cors import CORS

UPLOAD_FOLDER = diretorio('uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.secret_key = 'supersecretkey' 

# Verifica se a extensão do arquivo é permitida
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_rm', methods=['POST'])
def upload_rm():

    """ Upload de arquivos para classificação de tumores. """
    
    if 'file' not in request.files:
        flash('Nenhum arquivo enviado')
        return redirect(request.url)
    
    file = request.files['file']

    if file.filename == '':
        flash('Nenhum arquivo selecionado')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Garante que o nome do arquivo é seguro
        filename = secure_filename(file.filename)
        path_imagem_rm = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Salva o arquivo no diretório de uploads
        file.save(path_imagem_rm)
        
        try:
            # Carrega o modelo treinado
            caminho_modelo = diretorio('modelo_treinado/modelo_detector_tumor.h5')
            if not os.path.exists(caminho_modelo):
                return "<p>Erro: Modelo treinado não encontrado!</p>"
            
            modelo = tf.keras.models.load_model(caminho_modelo)
            
            # Chama a função para identificar o tumor
            resultado = identificar_tumor(path_imagem_rm, modelo)
            
            # Remove o arquivo após a análise para limpar o diretório
            os.remove(path_imagem_rm)

            return f"<p>Análise concluída!</p><p><b>{resultado}</b></p><a href='/'>Analisar outra imagem</a>"

        except Exception as e:
            return f"<p>Ocorreu um erro durante a análise: {e}</p>"
    else:
        return "<p>Extensão de arquivo não permitida. Use 'png', 'jpg' ou 'jpeg'.</p>"