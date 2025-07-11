import streamlit as st
import numpy as np
from qiskit_algorithms import QAOA, SamplingVQE
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.circuit.library import RealAmplitudes
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import InequalityToEquality, IntegerToBinary, LinearEqualityToPenalty
from qiskit_algorithms.utils import algorithm_globals
import warnings
import time
import sys
import math
from math import exp
from pyDOE2 import lhs
from sklearn.cluster import KMeans
import io
import matplotlib.pyplot as plt
import base64



parametros_treino=[
    [5.64955258, 5.13768523],
    [3.61058585, 1.50012797],
    [4.010099, 3.52868256],
    [4.11607976, 5.6834854],
    [0.06207073, 5.94693377],
    [2.95672097, 2.50510161],
    [4.5185035, 4.89354295],
    [0.3059588, 4.61665871],
    [1.16213395, 2.69425644],
    [3.2913161, 4.62269263],
    [5.39444087, 0.93767015],
    [1.65486403, 3.92155331],
    [0.9007122, 1.12241408],
    [3.5433401, 0.36532233],
    [5.44229483, 0.14221492],
    [5.59171369, 0.54184375],
    [5.77141418, 1.36856365],
    [4.88822815, 4.4179515],
    [2.21347623, 2.5046945],
    [5.59580687, 1.93161085],
    [5.43202626, 4.43408805],
    [1.74131047, 3.28836299],
    [1.11717397, 1.40617162],
    [5.10379713, 4.82242841],
    [0.94864183, 4.26102119],
    [1.22151151, 4.17421882],
    [2.48937933, 2.39064838],
    [2.43619386, 0.59423984],
    [5.94310436, 1.3699992],
    [4.63213223, 0.40957529],
    [0.75894679, 6.23837798],
    [2.77578539, 1.436039],
    [6.26838495, 1.37941869],
    [0.41929643, 0.24710771],
    [4.72602909, 2.861201],
    [5.40509589, 1.68638764],
    [0.29483925, 0.7874109],
    [2.33328555, 1.79361212],
    [5.97029726, 4.83125872],
    [3.47801, 1.46867375],
    [3.91608824, 0.71458607],
    [0.44421512, 3.37681099],
    [1.94995772, 3.18787309],
    [5.33968064, 5.06136689],
    [2.71236618, 4.98453269],
    [0.66708969, 6.00416504],
    [0.7003309, 0.18990556],
    [5.14133123, 1.89366819],
    [3.84203933, 1.56963872],
    [3.82093591, 4.77167525],
    [1.41782966, 2.12239654],
    [2.20481875, 0.74545343],
    [4.14560754, 3.93178518],
    [1.64510614, 2.99335506],
    [1.48930073, 0.68871199],
    [2.88094723, 4.14656843]
]

def generate_lhs_samples(param_intervals, num_samples):
    lhs_samples = lhs(len(param_intervals), samples=num_samples, criterion='maximin')
    lhs_scaled = np.zeros((num_samples, len(param_intervals)))

    for i in range(len(param_intervals)):
        lhs_scaled[:, i] = param_intervals[i][0] + lhs_samples[:, i] * (param_intervals[i][1] - param_intervals[i][0])

    return lhs_scaled

textos_idioma = {
    "Português": "Idioma",
    "English": "Language"
}


# Textos multilíngues
TEXTOS = {
    "pt": {
        "intro": "Este aplicativo foi criado para incentivar o uso da computação quântica em três áreas distintas, apresentadas a seguir.\nEscolha a área que deseja explorar e descubra as possibilidades oferecidas por essa tecnologia inovadora.",
        "pagina_otimizacao": "Otimização Quântica",
        "pagina_otimizacao2": "Otimização Quântica em Problemas de Alocação de Redundâncias",
        "pagina_inferencia": "Inferência Quântica",
        "pagina_ml": "Aprendizagem de Máquina Quântica",
        "instancia_input": "Digite alguma coisa para testar a instância:",
        "instancia_recebida": "Instância recebida:",
        "idioma": "Escolha o idioma:", 
        "referencias_titulo": "Referências",
        "referencias_intro": "Para conhecer mais sobre nossos trabalhos nas áreas, consulte as referências abaixo:",
        "info_ml": "Seção para descrever as técnicas de Machine Learning Quântico usadas.",
        "info_inf": "Seção para descrever as técnicas de Inferância Quântica usadas.",
        "titulo": "Seja bem-vindo ao <span style='color:#0d4376;'>QXplore</span>!",
        "corpo": (
            "O QXplore é um aplicativo focado em apoiar o estudo e a experimentação com computação quântica aplicada a problemas comuns da engenharia da confiabilidade.\n\n"
            "Ele oferece três áreas principais onde você pode explorar como métodos quânticos podem ser usados para modelar e analisar desafios em sistemas e processos de confiabilidade.\n\n"
            "Embora a tecnologia quântica ainda esteja em desenvolvimento, este aplicativo traz ferramentas e exemplos que ajudam a entender seu funcionamento e seu potencial, mesmo que ainda de forma exploratória, para problemas de engenharia.\n\n"
            "Explore as áreas disponíveis para conhecer melhor essa tecnologia e como ela pode ser aplicada a casos reais."
        ),
        "ini": "Página incial", 
        "pagina_referencias": "Referências"
    },
    "en": {
        "intro": "This application was developed to promote the use of quantum computing in three distinct areas, described below.\nSelect the area you want to explore and discover the possibilities offered by this innovative technology.",
        "pagina_otimizacao": "Quantum Optimization",
        "pagina_otimizacao2": "Quantum Optimization in Redundancy Allocation Problems",
        "pagina_inferencia": "Quantum Inference",
        "pagina_ml": "Quantum Machine Learning",
        "instancia_input": "Type something to test the instance:",
        "instancia_recebida": "Received instance:",
        "idioma": "Choose the language:", 
        "referencias_titulo": "References",
        "referencias_intro": "To learn more about our work in this areas, check the references below:", 
        "info_ml": "Section describing the Quantum Machine Learning techniques used.",
        "info_inf": "Section describing the Quantum Inference techniques used.",
        "titulo": "Welcome to <span style='color:#0d4376;'>QXplore</span>!",
        "corpo": (
            "Welcome to QXplore!\n\n"
            "QXplore is an application focused on supporting the study and experimentation of quantum computing applied to common problems in reliability engineering.\n\n"
            "It offers three main areas where you can explore how quantum methods can be used to model and analyze challenges in system and process reliability.\n\n"
            "Although quantum technology is still under development, this app provides tools and examples to help you understand its operation and potential, even if exploratory, in engineering problems.\n\n"
            "Explore the available areas to better understand this technology and how it can be applied to real cases."
        ),
        "ini": "Homepage",
        "pagina_referencias": "References"
    }
}

TEXTOS_OPT = {
    "pt": {
        "idioma": "Idioma",
        "insira_dados": "Insira os dados do problema a ser analisado:",
        "instancia": "Instância fornecida:",
        "carregar_arquivo": "Carregar arquivo:",
        "minutos": "minutos",
        "minutos_e_segundos": "minutos e {segundos} segundos",
        # Textos da ajuda
        "problema_rap": "Problema de Alocação de Redundâncias (RAP):",
        "descricao_rap": "O RAP refere-se à otimização da alocação de componentes redundantes em um sistema para aumentar sua confiabilidade e disponibilidade.",
        "algoritmos": "Algoritmos quânticos disponíveis:",
        "inicializacoes_titulo": "Métodos de Inicialização",
        "inicializacoes_descricao": (
            "**Clusterização:** parâmetros baseados nos centros dos clusters ótimos.\n\n"
            "**LHS:** amostragem uniforme pelo hipercubo latino.\n\n"
            "**Randômica:** parâmetros iniciados aleatoriamente.\n\n"
            "**Ponto Fixo:** valores iniciais fixos e pré-definidos."),
        "descricao_algoritmos": "Os algoritmos quânticos de otimização são projetados para explorar as propriedades únicas da mecânica quântica, como superposição e entrelaçamento, para resolver problemas de otimização, como o RAP.",
        "qaoa_nome": "QAOA",
        "qaoa_desc": "Quantum Approximate Optimization Algorithm é um algoritmo quântico projetado para resolver problemas de otimização combinatória, como o RAP, aproximando-se das soluções ótimas utilizando uma sequência parametrizada de operações quânticas.",
        "vqe_nome": "VQE",
        "vqe_desc": "Variational Quantum Eigensolver é um algoritmo híbrido quântico-clássico que usa um circuito quântico variacional para encontrar o estado de menor energia de um Hamiltoniano, mas requer mais parâmetros e pode demandar mais tempo computacional em comparação com o QAOA.", 
        "modo_leitura_label": "Selecione o modo de entrada dos dados:",
        "modo_leitura_manual": "Inserção manual (preencher os dados manualmente)",
        "modo_leitura_upload": "Upload de arquivo (arquivo .txt)",
        "ajuda_upload_botao": "Mostrar ajuda para upload",
        "ajuda_upload_texto": """
        <div style="background-color: #f9f9f9; margin: 0; padding: 12px; border-radius: 5px; border: 1px solid #ddd; max-width: 850px; font-size: 14px; line-height: 1.4;">
            <h4 style="color: #333; font-size: 16px; margin: 8px 0;">Instruções para Upload</h4>
            <p style="margin: 4px 0;">O arquivo de entrada deve ser um arquivo de texto (.txt), onde cada linha representa uma instância, com o seguinte formato:</p>
            <p style="background-color: #eee; padding: 6px; border-radius: 2px; font-size: 10px; margin: 2px 0;"><code>[s, nj_max, nj_min, ctj_of, Rjk_of, cjk_of, C_of]</code></p>
            <p style="margin: 4px 0;">Certifique-se de que o arquivo siga exatamente este formato para que os dados sejam lidos corretamente.</p>
            <p style="margin: 4px 0;">Clique no botão abaixo para baixar um arquivo de teste já formatado.</p>
        </div>
        """,
        "botao_mostrar_instancia": "Mostrar instância",
        "selecionar_algoritmo": "Selecione o algoritmo quântico:",
        "tipo_inicializacao": "Selecione o método de inicialização dos parâmetros:",
        "inserir_ponto_fixo": "Insira o ponto fixo:",
        "inserir_camadas": "Insira o número de camadas:",
        "inserir_rodadas": "Insira o número de rodadas:", 
        "executar": "Executar",
        "parametros_iniciais": "Parâmetros iniciais",
        "rodada": "Rodada",
        "camada": "Camada",
        "executando_qaoa": "Executando QAOA, por favor, aguarde...",
        "resultados": "Resultados",
        "energia_otima": "Energia Ótima",
        "confiabilidade_otima": "Confiabilidade Ótima",
        "componentes_solucao": "Componentes da Solução",
        "custo_total": "Custo Total da Solução",
        "medidas_energia": "Medidas Descritivas das Energias",
        "media_energia": "Média das Energias",
        "desvio_padrao_energia": "Desvio Padrão das Energias",
        "conteudo_pagina_ml": "Dantas",
        "conteudo_pagina_inferencia": "Lavínia",
        "tipo_inicializacao": "Tipo de inicialização",
        "inserir_ponto_fixo": "Insira o valor do ponto fixo",
        "tipos_inicializacao_vqe": ['LHS', 'Randômica', 'Ponto Fixo'],
        "tipos_inicializacao_qaoa": ['Clusterização', 'LHS', 'Randômica', 'Ponto Fixo'],
        "executando_vqe": "Executando VQE, por favor, aguarde...",
        "de": "de",
        "pagina_otimizacao": "Otimização Quântica",
        "s": "Número de subsistema",
        "nj_max": "Valor máximo dos componentes por subsistema",
        "nj_min": "Valor mínimo dos componentes por subsistema",
        "ctj_of": "Quantidade de tipos de componentes disponíveis",
        "lista_componentes": "Informe a confiabilidade e o custo de cada componente:",
        "confiabilidade": "Confiabilidade do componente",
        "custo": "Custo do componente",
        "custo_total_limite": "Limite máximo de custo",
        "selecionar_tipo_circuito": "Selecione o tipo de circuito VQE:",
        "real_amplitudes": "Real Amplitudes",
        "two_local": "Two Local",
        "opcoes_rotacao": ["rx", "ry", "rz"],
        "selecionar_rotacao": "Selecione as portas de rotação:",
        "opcoes_emaranhamento": ["cx", "cz", "iswap"],
        "selecionar_emaranhamento": "Selecione as portas de emaranhamento:",
        "tipo_inicializacao": "Selecione o método de inicialização:",
        "selecionar_otimizador": "Selecione o otimizador clássico:",
        "opcoes_otimizadores": ["SPSA", "COBYLA"],
        "inserir_shots": "Insira o número de shots:",
        "area_de_aplicacao": "Áreas de Aplicação:",
        "circuito_quantico": "Circuito Quântico",
        "Baixar": "Baixar arquivo",
        "download_text": "Caso deseje, faça o download do arquivo de teste exemplificado para usar ou visualizar.", 
        "rap_descricao": (
            "#### Problema de Alocação de Redundâncias (RAP)\n\n"
            "O Problema de Alocação de Redundâncias é um problema clássico da engenharia de confiabilidade que consiste em determinar "
            "quantos componentes redundantes alocar em cada subsistema para maximizar a confiabilidade do sistema total, respeitando restrições de custo.\n\n"

            "##### Formulação Matemática\n\n"
            "Seja um sistema com \\( s \\) subsistemas, o objetivo é maximizar a confiabilidade total \\( R(x) \\):\n\n"
            "$$ R(x) = \\prod_{i=1}^{s} \\left(1 - R_i\\right)^{x_i} $$\n\n"

            "##### Restrições:\n\n"
            "$$ \\sum_{i=1}^{s} c_i x_i \\leq C $$\n\n"
            "$$ n_{\\text{min}} \\leq x_i \\leq n_{\\text{max}}, \\quad \\forall i = 1, 2, \\ldots, s $$\n\n"
            "$$ x_i \\in \\{0, 1, 2, \\ldots, n_{\\text{max}}\\} $$\n\n"
            
            "##### Termos Utilizados\n\n"
            "- $x_i$: número de componentes redundantes no subsistema $i$  \n"
            "- $R_i$: confiabilidade de um componente do subsistema $i$  \n"
            "- $R(x)$: confiabilidade global do sistema  \n"
            "- $c_i$: custo de adicionar um componente no subsistema $i$  \n"
            "- $C$: orçamento máximo  \n"
            "- $n_{\\mathrm{min}}, n_{\\mathrm{max}}$: limites inferior e superior para redundâncias  \n"

        ),
        "aplicacao": "Aplicação",
    },
    "en": {
        "idioma": "Language",
        "carregar_arquivo": "Upload file:",
        "modo_leitura_label": "Select the data input mode:",
        "modo_leitura_manual": "Manual input (manually fill the data)",
        "modo_leitura_upload": "File upload (.txt file)",
        "minutos": "minutes",
        "minutos_e_segundos": "minutes and {segundos} seconds",
        "insira_dados": "Enter the problem data to be analyzed:",

        # Help section
        "problema_rap": "Redundancy Allocation Problem (RAP):",
        "descricao_rap": "RAP refers to the optimization of allocating redundant components in a system to increase its reliability and availability.",

        "algoritmos": "Available quantum algorithms:",
        "descricao_algoritmos": "Quantum optimization algorithms are designed to leverage the unique properties of quantum mechanics, such as superposition and entanglement, to solve optimization problems like RAP.",

        "qaoa_nome": "QAOA",
        "qaoa_desc": "Quantum Approximate Optimization Algorithm is a quantum algorithm designed to solve combinatorial optimization problems, such as RAP, by approximating optimal solutions using a parameterized sequence of quantum operations.",

        "vqe_nome": "VQE",
        "vqe_desc": "Variational Quantum Eigensolver is a hybrid quantum-classical algorithm that uses a variational quantum circuit to find the lowest energy state of a Hamiltonian, but it requires more parameters and may take longer computational time compared to QAOA.",
        "ajuda_upload_botao": "Show upload help",
        "ajuda_upload_texto": """
        <div style="background-color: #f9f9f9; margin: 0; padding: 12px; border-radius: 5px; border: 1px solid #ddd; max-width: 850px; font-size: 14px; line-height: 1.4;">
            <h4 style="color: #333; font-size: 16px; margin: 8px 0;">Upload Instructions</h4>
            <p style="margin: 4px 0;">The input file must be a plain text file (.txt), where each line represents one instance in the following format:</p>
            <p style="background-color: #eee; padding: 6px; border-radius: 2px; font-size: 10px; margin: 2px 0;"><code>[s, nj_max, nj_min, ctj_of, Rjk_of, cjk_of, C_of]</code></p>
            <p style="margin: 4px 0;">Ensure that the file strictly follows this format so that the data can be read correctly.</p>
            <p style="margin: 4px 0;">Click the button below to download a pre-formatted test file.</p>
        </div>
        """,
        "botao_mostrar_instancia": "Show instance",
        "selecionar_algoritmo": "Select the quantum algorithm:",
        "tipo_inicializacao": "Select the parameter initialization method:",
        "inserir_ponto_fixo": "Enter the fixed point:",
        "inserir_camadas": "Enter the number of layers:",
        "inserir_rodadas": "Enter the number of iterations:",
        "executar": "Execute",
        "modo_leitura_upload": "Upload",
        "parametros_iniciais": "Initial parameters",
        "rodada": "Round",
        "camada": "Layer",
        "executando_qaoa": "Running QAOA, please wait...",
        "resultados": "Results",
        "energia_otima": "Optimal Energy",
        "confiabilidade_otima": "Optimal Reliability",
        "circuito_quantico": "Quantum Circuit",
        "componentes_solucao": "Solution Components",
        "custo_total": "Total Cost of the Solution",
        "medidas_energia": "Descriptive Measures of Energy",
        "media_energia": "Average Energy",
        "desvio_padrao_energia": "Standard Deviation of Energy",
        "conteudo_pagina_ml": "Dantas",
        "conteudo_pagina_inferencia": "Lavínia",
        "tipo_inicializacao": "Initialization type",
        "inserir_ponto_fixo": "Enter the fixed point value",
        "tipos_inicializacao_vqe": ['LHS', 'Random', 'Fixed Point'],
        "tipos_inicializacao_qaoa": ['Clustering', 'LHS', 'Random', 'Fixed Point'],
        "executando_vqe": "Running VQE, please wait...",
        "de": "of",
        "pagina_otimizacao": "Quantum Optimization",
        "s": "Number of subsystems",
        "nj_max": "Maximum number of components per subsystem",
        "nj_min": "Minimum number of components per subsystem",
        "ctj_of": "Number of available component types",
        "lista_componentes": "Enter the reliability and cost for each component type:",
        "confiabilidade": "Reliability of component ",
        "custo": "Cost of component",
        "custo_total_limite": "Maximum total cost limit",
        "inicializacoes_titulo": "Initialization Methods",
        "inicializacoes_descricao": (
            "**Clustering:** parameters based on centers of optimal clusters.\n\n"
            "**LHS:** uniform sampling via Latin Hypercube.\n\n"
            "**Random:** parameters initialized randomly.\n\n"
            "**Fixed Point:** fixed, predefined initial values."),
        "selecionar_tipo_circuito": "Select the type of VQE circuit:",
        "real_amplitudes": "Real Amplitudes",
        "two_local": "Two Local",

        "opcoes_rotacao": ["rx", "ry", "rz"],
        "selecionar_rotacao": "Select rotation gates:",

        "opcoes_emaranhamento": ["cx", "cz", "iswap"],
        "selecionar_emaranhamento": "Select entanglement gates:",

        "tipo_inicializacao": "Select the initialization method:",
        "selecionar_otimizador": "Select the classical optimizer:",
        "opcoes_otimizadores": ["SPSA", "COBYLA"], 
        "inserir_shots": "Enter the number of shots:",
        "area_de_aplicacao": "Areas of Application:",
        "modo_leitura_label": "Select the data input mode:",
        "modo_leitura_manual": "Manual input (enter the data manually)",
        "modo_leitura_upload": "File upload (.txt file)",
        "Baixar": "Download file",
        "download_text": "If you wish, download the sample test file to use or visualize.",
        "rap_descricao": (
        "#### Redundancy Allocation Problem (RAP)\n\n"
        "The Redundancy Allocation Problem is a classic issue in reliability engineering. "
        "It involves determining how many redundant components to allocate to each subsystem in order to maximize the overall system reliability, "
        "while respecting cost constraints.\n\n"

        "##### Mathematical Formulation\n\n"
        "Consider a system with \\( s \\) subsystems. The objective is to maximize the total system reliability \\( R(x) \\):\n\n"
        "$$ R(x) = \\prod_{i=1}^{s} \\left(1 - R_i\\right)^{x_i} $$\n\n"

        "##### Constraints:\n\n"
        "$$ \\sum_{i=1}^{s} c_i x_i \\leq C $$\n\n"
        "$$ n_{\\text{min}} \\leq x_i \\leq n_{\\text{max}}, \\quad \\forall i = 1, 2, \\ldots, s $$\n\n"
        "$$ x_i \\in \\{0, 1, 2, \\ldots, n_{\\text{max}}\\} $$\n\n"

        "##### Terms Used\n\n"
        "- $x_i$: number of redundant components in subsystem $i$  \n"
        "- $R_i$: reliability of a component in subsystem $i$  \n"
        "- $R(x)$: overall system reliability  \n"
        "- $c_i$: cost of adding a component to subsystem $i$  \n"
        "- $C$: total budget available  \n"
        "- $n_{\\mathrm{min}}, n_{\\mathrm{max}}$: lower and upper bounds for redundancy allocation  \n"
    ),
        "aplicacao": "Application",

    }
}


TEXTOS_ML = {
    "pt": {
        "pagina_ml": "Aprendizagem de Máquina Quântica",
        "idioma_label": "Idioma / Language",
        "dataset_opcao": "Escolha entre dados já existentes de vibração (rolamentos):",
        "selecione_base": "Selecione a base",
        "upload_dados": "Importe dados próprios:",
        "upload_label": "Faça upload da sua base de dados",
        "upload_info": "Por favor, envie um arquivo CSV, Excel ou Parquet.",
        "upload_sucesso": "Base de dados carregada com sucesso!",
        "preview": "Visualização da base de dados:",
        "selecione_features": "Selecione as features a serem extraídas da base (caso deseje)",
        "label_features": "Selecione as features:",
        "encoding_title": "Escolha a codificação quântica.",
        "encoding_label": "Escolha um método de codificação",
        "euler_title": "PQC: escolha a quantidade de rotações de Euler:",
        "euler_label": "Selecione a quantidade",
        "euler_eixo1": "Escolha o eixo da rotação",
        "euler_eixo_n": "Escolha o eixo da {n}ª rotação",
        "entanglement_title": "PQC: escolha a porta de emaranhamento",
        "paciencia": "Insira o valor da paciência:",
        "epocas": "Insira o número de épocas:",
        "erro_1": "Por favor, selecione um dataset.",
        "erro_2": "Por favor, selecione ao menos uma característica.",
        "erro_3": "Por favor, selecione um método de codificação.",
        "erro_4": "Por favor, selecione os eixos das rotações.",
        "erro_5": "Erro ao carregar o dataset.",
        "exec_1": "Execução iniciada!",
        "acc": "Acurácia do modelo:",
        "exec_2": "Executar modelo",
        "metodos": "Métodos de codificação quântica disponíveis:",
        "angle": "Angle encoding",
        "desc_angle": "Codifica os dados clássicos como ângulos de rotação aplicados a portas quânticas (como RX, RY, RZ). Cada valor de uma feature é mapeado diretamente para uma rotação em um qubit.",
        "ampli": "Amplitude encoding",
        "desc_ampli": "Codifica os dados nos amplitudes do estado quântico, normalizando o vetor de entrada para representar diretamente o estado do sistema.",
        "pauli": "Pauli Feature Map",
        "desc_pauli": "Codificações que usam portas baseadas nos operadores de Pauli para mapear dados em circuitos quânticos (X, Y, Z e ZZFeaturemap)",
        "help_1": "Escolha uma base de dados pré-carregada com sinais de vibração.",
        "help_2": "Nessa opção você pode realizar o upload de dados próprios em formato csv, xlsx ou parquet.",
        "help_3": "Nessa opção você pode selecionar a quantidade de características que quiser extrair do dataset original, selecionado acima.",
        "help_4": "Selecione o método que realizará a codificação dos dados clássicos em quânticos.",
        "help_5": "Selecione quantas rotações de euler ocorrerão no seu circuito quântico.",
        "help_6": "Selecione as rotações correspondentes a quantidade que você escolheu.",
        "help_7": "Nessa etapa você vai selecionar o tipo de emaranhamento quântico que gostaria de adicionar no circuito."
    },
    "en": {
        "idioma_label": "Idioma / Language",
        "dataset_opcao": "Choose from existing vibration data (bearings):",
        "selecione_base": "Select the dataset",
        "upload_dados": "Upload your own data:",
        "upload_label": "Upload your dataset",
        "pagina_ml": "Quantum Machine Learning",
        "upload_info": "Please upload a CSV, Excel, or Parquet file.",
        "upload_sucesso": "Dataset loaded successfully!",
        "preview": "Dataset preview:",
        "selecione_features": "Select the features to extract from the dataset (optional)",
        "label_features": "Select features:",
        "encoding_title": "Choose the quantum encoding.",
        "encoding_label": "Choose an encoding method",
        "euler_title": "PQC: choose the number of Euler rotations:",
        "euler_label": "Select quantity",
        "euler_eixo1": "Choose the rotation axis",
        "euler_eixo_n": "Choose the {n}st rotation axis",
        "entanglement_title": "PQC: choose the entanglement gate",
        "paciencia": "Enter the patience value:",
        "epocas": "Enter the number of epochs:",
        "erro_1": "Please select a dataset.",
        "erro_2": "Please select at least one feature.",
        "erro_3": "Please select an encoding method.",
        "erro_4": "Please select the rotation axes.",
        "erro_5": "Error loading the dataset.",
        "exec_1": "Execution started!",
        "acc": "Model accuracy:",
        "exec_2": "Run model",
        "metodos": "Available quantum encoding methods:",
        "angle": "Angle encoding",
        "desc_angle": "Encodes classical data as rotation angles applied to quantum gates (such as RX, RY, RZ). Each feature value is directly mapped to a qubit rotation.",
        "ampli": "Amplitude encoding",
        "desc_ampli": "Encodes data into the amplitudes of the quantum state, normalizing the input vector to directly represent the system state.",
        "pauli": "Pauli Feature Map",
        "desc_pauli": "Encodings that use gates based on Pauli operators to map data into quantum circuits (X, Y, Z, and ZZFeatureMap).",
        "help_1": "Choose a pre-loaded database with vibration signals.",
        "help_2": "In this option, you can upload your own data in csv, xlsx, or parquet format.",
        "help_3": "In this option, you can select the number of characteristics you want to extract from the original dataset selected above.",
        "help_4": "Select the method that will perform the encoding of classical data into quantum.",
        "help_5": "Select how many Euler rotations will occur in your quantum circuit.",
        "help_6": "Select the rotations corresponding to the quantity you chose.",
        "help_7": "At this stage, you will select the type of quantum entanglement you would like to add to the circuit."
    }
}

def aplicar_css_botoes():
    st.markdown(
        """
        <style>
        /* Aplica estilo aos botões de forma global */
        div.stButton > button {
            background-color: #0d4376 !important;
            color: white !important;
            width: 150px !important;
            height: 80px !important;
            border-radius: 8px !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            transition: background-color 0.1s ease !important;
            margin-top: 10px !important;
        }
        div.stButton > button:hover {
            background-color: #07294a !important;
        }
        
        /* Aplica o mesmo estilo ao botão de download */
        div.stDownloadButton > button {
            background-color: #0d4376 !important;
            color: white !important;
            width: 150px !important;
            height: 80px !important;
            border-radius: 8px !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            transition: background-color 0.1s ease !important;
            margin-top: 10px !important;
        }
        div.stDownloadButton > button:hover {
            background-color: #07294a !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
def mostrar_introducao_e_titulo(textos):
    titulo = textos['titulo']
    corpo = textos['corpo']

    st.markdown(
        f"""
        <div style="text-align: center; max-width: 700px; margin: auto;">
            <h1 style="font-size: 32px; margin-bottom: 2px;">{titulo}</h1>
            <p style="font-size: 16px; line-height: 1.5; margin-top: 0;">
                {corpo}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
def mostrar_referencias(textos, textos_otim):
    st.title(textos.get("pagina_referencias_titulo", "Referências"))

    st.header(textos_otim["pagina_otimizacao"])
    st.markdown("""
    - **Araújo, L. M. M., Lins, I., Aichele, D., Maior, C., Moura, M., & Droguett, E. (2022).**  
      *Review of Quantum(-Inspired) Optimization Methods for System Reliability Problems.*  
      16th International Probabilistic Safety Assessment and Management Conference - PSAM 16.
    
    - **Araújo, L. M. M., Lins, I., Maior, C., Aichele, D., & Droguett, E. (2022).**  
      *A Quantum Optimization Modeling for Redundancy Allocation Problems.*  
      32nd European Safety and Reliability (ESREL) Conference.
    
    - **Araújo, L. M. M., Lins, I., Maior, C. S., Moura, M., & Droguett, E. (2023b).**  
      *A Linearization Proposal for the Redundancy Allocation Problem.*  
      INFORMS Annual Meeting.
    
    - **Araújo, L. M. M., Raupp, L., Lins, I., & Moura, M. (2024).**  
      *Quantum Approaches for Reliability Estimation: A Systematic Literature Review.*  
      34th European Safety and Reliability (ESREL) Conference.
    
    - **Bezerra, V., Araújo, L., Lins, I., Maior, C., & Moura, M. (2024a).**  
      *Exploring initialization strategies for quantum optimization algorithms to solve the redundancy allocation problem.*  
      34th European Safety and Reliability (ESREL) Conference.
    
    - **Bezerra, V., Araújo, L., Lins, I., Maior, C., & Moura, M. (2024b).**  
      *Quantum optimization applied to the allocation of redundancies in systems in the Oil & Gas industry.*  
      Anais do LVI Simpósio Brasileiro de Pesquisa Operacional.
    
    - **Bezerra, V. M. A., Araújo, L. M. M., Lins, I. D., Maior, C. B. S., & Moura, M. J. D. C. (2024).**  
      *Optimization of system reliability based on quantum algorithms considering the redundancy allocation problem.*  
      [DOI: 10.48072/2525-7579.roge.2024.3481](https://doi.org/10.48072/2525-7579.roge.2024.3481)
    
    - **Lins, I., Araújo, L., Maior, C., Teixeira, E., Bezerra, P., Moura, M., & Droguett, E. (2023).**  
      *Quantum Optimization for Redundancy Allocation Problem Considering Various Subsystems.*  
      33th European Safety and Reliability (ESREL) Conference.
        """)
        
def mostrar_cartoes_de_area(textos):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.write("")
    with col2:
        st.image("opt3.png", width=150)
        if st.button(textos["pagina_otimizacao"], key="otimizacao_btn"):
            st.session_state['pagina'] = 'otimizacao'
    with col3:
        st.image("ml3.png", width=150)
        if st.button(textos["pagina_ml"], key="ml_btn"):
            st.session_state['pagina'] = 'ml'

        if st.button(textos["pagina_referencias"], key="referencias_btn"):
            st.session_state['pagina'] = 'referencias'
            
    with col4:
        st.image("infer3.png", width=150)
        if st.button(textos["pagina_inferencia"], key="inferencia_btn"):
            st.session_state['pagina'] = 'inferencia'
    with col5:
        st.write("")



def ler_manualmente(textos):
    valor = st.text_input(textos["instancia_input"])
    if valor:
        return {"valor": valor}
    return None

def mostrar_instancia(instancia, textos):
    st.write(textos["instancia_recebida"])
    st.json(instancia)

def mostrar_logo_topo():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("qxplore.png", width=600)
        
#Otimização

def ler_manualmente(textos_otim):
    st.write(textos_otim["insira_dados"])

    # Organizar inputs principais em 2 colunas
    col1, col2 = st.columns(2)
    with col1:
        s = st.number_input(f"{textos_otim['s']}:", step=1, min_value=1)
        nj_min = st.number_input(f"{textos_otim['nj_min']}:", step=1, min_value=0)
    with col2:
        nj_max = st.number_input(f"{textos_otim['nj_max']}:", step=1, min_value=1)
        ctj_of = st.number_input(f"{textos_otim['ctj_of']}:", step=1, min_value=1)

    st.markdown(f"**{textos_otim['lista_componentes']}**")

    Rjk_of = []
    cjk_of = []

    for i in range(int(ctj_of)):
        col_r, col_c = st.columns(2)
        with col_r:
            Rjk_of.append(
                st.number_input(f"{textos_otim['confiabilidade']} [{i+1}]:", 
                                key=f'Rjk_of_{i}', 
                                step=0.001, 
                                min_value=0.000, 
                                max_value=1.0, 
                                format="%.8f")
            )
        with col_c:
            cjk_of.append(
                st.number_input(f"{textos_otim['custo']} [{i+1}]:", 
                                key=f'cjk_of_{i}', 
                                step=1, 
                                min_value=0)
            )

    # Input final em destaque
    C_of = st.number_input(f"{textos_otim['custo_total_limite']}:", step=1, min_value=1)

    dados = [[s, nj_max, nj_min, ctj_of, Rjk_of, cjk_of, C_of]]
    return dados
    
def mostrar_instancia(instancia, textos_otim):
    st.subheader(textos_otim["instancia"])
    
    s, nj_max, nj_min, ctj_of = instancia[0][0], instancia[0][1], instancia[0][2], instancia[0][3]
    Rjk_of = instancia[0][4]
    cjk_of = instancia[0][5]
    C_of = instancia[0][6]
    
    # Dados gerais em uma linha de colunas
        
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("s", s)
    col2.metric("nj_max", nj_max)
    col3.metric("nj_min", nj_min)
    col4.metric("ctj_of", ctj_of)
    col5.metric("C_of", C_of)
    
    # Mostrar Rjk_of e cjk_of lado a lado em uma tabela organizada
    st.write("#### Valores de Rjk_of e cjk_of")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Rjk_of**")
        for i, val in enumerate(Rjk_of, 1):
            st.write(f"{i}: {val:.8f}")
            
    with col2:
        st.write("**cjk_of**")
        for i, val in enumerate(cjk_of, 1):
            st.write(f"{i}: {val}")
    
    st.markdown("---")


def ler_do_drive(textos_otim):
    arquivo = st.file_uploader(textos_otim["carregar_arquivo"], type=['txt'])
    if arquivo is not None:
        dados = arquivo.readlines()
        return [eval(linha.strip()) for linha in dados]
    return []

def formatar_tempo(segundos, textos_otim):
    minutos = math.floor(segundos / 60)
    segundos_restantes = math.ceil(segundos % 60)
    if segundos_restantes == 60:
        segundos_restantes = 0
        minutos += 1
    return (
        f"{minutos} {textos_otim['minutos']}"
        if segundos_restantes == 0 else
        f"{minutos} {textos_otim['minutos_e_segundos'].format(segundos=segundos_restantes)}"
    )

def mostrar_otim(textos_otim):
    with st.sidebar:
        st.markdown(f"#### {textos_otim['area_de_aplicacao']}")
        with st.sidebar.expander(textos_otim["pagina_otimizacao"]):        
            st.markdown(f"#### {textos_otim['problema_rap']}")
            st.markdown(f"{textos_otim['descricao_rap']}")

            st.markdown(f"#### {textos_otim['algoritmos']}")

            st.markdown(f"**QAOA**: {textos_otim['qaoa_desc']}")
            st.markdown(f"**VQE**: {textos_otim['vqe_desc']}")

            st.markdown(f"#### {textos_otim['inicializacoes_titulo']}")
            st.markdown(textos_otim['inicializacoes_descricao'])
        
def mostrar_ml(textos_ml):
    with st.sidebar.expander(textos_ml['pagina_ml']):
        st.markdown(f"#### {textos_ml['metodos']}")
        st.markdown(f"**Angle Encoding:** {textos_ml['desc_angle']}")
        st.markdown(f"**Amplitude Encoding:** {textos_ml['desc_ampli']}")
        st.markdown(f"**Pauli Feature Map:** {textos_ml['desc_pauli']}")
        
def mostrar_inf(textos):
    with st.sidebar.expander(textos["pagina_inferencia"]):
        st.markdown(f"#### {'Coisas de Lavínia'}")
        st.markdown(f"{textos['info_inf']}")

def main():
    st.set_page_config(
    page_title="QXplore",
    page_icon="pesq.png",
    layout="wide"
)

    aplicar_css_botoes()

    # 1 - imagem no topo da sidebar
    st.sidebar.image("CM.png", use_container_width=True)

    # 2 - escolha de idioma logo abaixo da imagem
    if 'lang' not in st.session_state:
        st.session_state.lang = None
    
    # Modal para escolha do idioma na primeira visita
    if st.session_state.lang is None:
        # Centraliza tudo usando markdown com CSS
        st.markdown(
            """
            <style>
                .centered {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: start;
                }
                .stButton > button {
                    width: 200px;
                    height: 50px;
                    font-size: 18px;
                }
            </style>
            """,
            unsafe_allow_html=True
        )
    
        st.markdown('<div class="centered">', unsafe_allow_html=True)
    
        mostrar_logo_topo()  # Sua função para exibir a logo
    
        st.markdown(
            """
            <div style="text-align: center;">
                <p style="font-size:36px; margin-bottom: 5px; font-weight: bold;">
                    Explore Quantum Computing with <span style="color:#0d4376;">QXplore!</span>
                </p>
                <p style="font-size:30px; margin-top: 0px; margin-bottom: 5px; font-weight: bold;">
                    Explore a Computação Quântica com <span style="color:#0d4376;">QXplore!</span>
                </p>
                <p style="font-size:18px; margin-top: 5px;">
                    Select your language to get started / Selecione seu idioma para começar:
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
        # Botões centralizados horizontalmente
        col1, col2, col3, col4= st.columns([1.65, 1, 1, 1.5])
        with col1:
            st.write("")
        with col2:
            if st.button("English"):
                st.session_state.lang = "en"
        
        with col3:
            if st.button("Português"):
                st.session_state.lang = "pt"


        st.markdown("</div>", unsafe_allow_html=True)
    
        # Caixa de informação sobre idioma
        st.info(
            "ℹ️ For a better experience, you can change the language anytime during navigation.\n\n"
            "ℹ️ Para uma melhor experiência, você pode alterar o idioma a qualquer momento durante a navegação."
        )

    
        st.stop()
    
    # 3 - Após escolha do idioma, sincroniza a seleção do sidebar com o idioma atual
    idioma_atual = "Português" if st.session_state.lang == "pt" else "English"
    idioma_selecionado = st.sidebar.selectbox(
        "Language / Idioma:",
        ("🇺🇸 English (US)", "🇧🇷 Português (BR)"),
        index=0 if idioma_atual == "English"  else 1
    )

    # Atualiza o idioma no estado se o usuário mudar pelo selectbox
    if idioma_selecionado == "🇧🇷 Português (BR)" and st.session_state.lang != "pt":
        st.session_state.lang = "pt"
    elif idioma_selecionado == "🇺🇸 English (US)" and st.session_state.lang != "en":
        st.session_state.lang = "en"

    lang = st.session_state.lang
    textos = TEXTOS[lang]
    textos_otim = TEXTOS_OPT[lang]
    textos_ml = TEXTOS_ML[lang]

    # 4 - referências em expander

    mostrar_logo_topo()
    
        
    # Ajuda
    mostrar_otim(textos_otim)
    mostrar_ml(textos_ml)
    mostrar_inf(textos)

    if 'pagina' not in st.session_state:
        st.session_state['pagina'] = 'inicio'
    
    if st.session_state['pagina'] == 'inicio':
        mostrar_introducao_e_titulo(textos)
        mostrar_cartoes_de_area(textos)
        
    elif st.session_state['pagina'] == 'otimizacao':
        st.subheader(textos["pagina_otimizacao2"])

        st.markdown(textos_otim["rap_descricao"])
        st.divider()

        st.subheader(textos_otim["aplicacao"])
        
        # Função para mudar de página
        def ir_para_explicacao():
            st.session_state['pagina'] = 'explicacao_otimizacao'
        
        # Botão fixo visível no topo direito
        st.markdown("""
            <style>
                .botao-fixo-topo {
                    position: fixed;
                    top: 10px;
                    right: 10px;
                    z-index: 9999;
                    background-color: #0e1117;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 10px;
                    font-size: 14px;
                    cursor: pointer;
                    box-shadow: 2px 2px 8px rgba(0,0,0,0.3);
                }
                .botao-fixo-topo:hover {
                    background-color: #262730;
                }
            </style>
        """, unsafe_allow_html=True)
        
        # Este botão usa o método Streamlit padrão, mas colocado em um container HTML fixo
        st.markdown("""
            <div class="botao-fixo-topo">
                <form action="" method="post">
                    <input type="submit" name="mudar_pagina" value="Ir para Explicação" style="all: unset; color: white; cursor: pointer;" />
                </form>
            </div>
        """, unsafe_allow_html=True)
        
        # Detecta clique e atualiza a sessão
        if st.session_state.get('pagina') != 'explicacao_otimizacao' and st.experimental_get_query_params().get('mudar_pagina'):
            st.session_state['pagina'] = 'explicacao_otimizacao'

        
        # Aplica estilos personalizados
        st.markdown("""
            <style>
            div[role="radiogroup"] > label > div:first-child {
                background-color: #03518C;
                border-radius: 50%;
                width: 20px;
                height: 20px;
            }
            body {
                color: #000000;
                background-color: #B6D0E4;
            }
            .centered-box {
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #ffffff;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            .loading-gif {
                width: 100px;
            }
            .loading-text {
                margin-top: 10px;
                font-size: 18px;
            }
            .stButton > button {
                color: #ffffff !important;
                background-color: #03518C !important;
                border-color: #03518C !important;
            }
            .stButton > button:active {
                background-color: #02416B !important;
            }
            .st-Radio > div > label {
                color: #03518C !important;
            }
            .st-Radio > div > div {
                border-color: #03518C !important;
            }
            a {
                color: #03518C !important;
            }
            </style>
        """, unsafe_allow_html=True)
    
        # Leitura de dados
       
        modo_leitura = st.radio(
            textos_otim["modo_leitura_label"],
            (textos_otim["modo_leitura_manual"], textos_otim["modo_leitura_upload"]),
            key=f"modo_leitura_{lang}", help= "OI"
        )
        
        dados = []
        with open("testeapp.txt", "r", encoding="utf-8") as f:
            conteudo_arquivo = f.read()
        if modo_leitura == textos_otim["modo_leitura_manual"]:
            dados = ler_manualmente(textos_otim)
        elif modo_leitura == textos_otim["modo_leitura_upload"]:
            if st.button(textos_otim["ajuda_upload_botao"]):
                st.markdown(textos_otim["ajuda_upload_texto"], unsafe_allow_html=True)

                st.download_button(
                    label=textos_otim["Baixar"],
                    data=conteudo_arquivo,
                    file_name="testeapp.txt",
                    mime="text/plain"
                )
            dados = ler_do_drive(textos_otim)
        
        # Verifica se os dados estão válidos
        if (modo_leitura == textos_otim["modo_leitura_manual"] and len(dados[0]) == 7) or \
           (modo_leitura == textos_otim["modo_leitura_upload"] and dados):
        
            if st.button(textos_otim["botao_mostrar_instancia"]):
                mostrar_instancia(dados, textos_otim)
        
            if len(dados[0]) != 1:
                col_alg, col_param = st.columns(2)
        
                with col_alg:
                    modo_algoritmo = st.radio(textos_otim["selecionar_algoritmo"], ('QAOA', 'VQE'))
        
                    if modo_algoritmo == 'VQE':
                        tipo_circuito = st.radio(
                            textos_otim["selecionar_tipo_circuito"], 
                            (textos_otim["real_amplitudes"], textos_otim["two_local"])
                        )
        
                        if tipo_circuito == textos_otim["two_local"]:
                            col_rot, col_ent = st.columns(2)
                            with col_rot:
                                rotacao_escolhida = st.multiselect(
                                    textos_otim["selecionar_rotacao"],
                                    textos_otim["opcoes_rotacao"]
                                )
                            with col_ent:
                                entanglement_escolhido = st.multiselect(
                                    textos_otim["selecionar_emaranhamento"],
                                    textos_otim["opcoes_emaranhamento"]
                                )
        
                        tipo_inicializacao = st.radio(
                            textos_otim["tipo_inicializacao"],
                            textos_otim["tipos_inicializacao_vqe"]
                        )
        
                        if tipo_inicializacao in ['Ponto Fixo', 'Fixed Point']:
                            numero_ponto_fixo = st.number_input(
                                textos_otim["inserir_ponto_fixo"], step=0.1
                            )
        
                    elif modo_algoritmo == 'QAOA':
                        tipo_inicializacao = st.radio(
                            textos_otim["tipo_inicializacao"],
                            textos_otim["tipos_inicializacao_qaoa"]
                        )
        
                        if tipo_inicializacao in ['Ponto Fixo', 'Fixed Point']:
                            numero_ponto_fixo = st.number_input(
                                textos_otim["inserir_ponto_fixo"], step=0.1
                            )
        
                with col_param:
                    otimizador = st.radio(
                        textos_otim["selecionar_otimizador"],
                        textos_otim["opcoes_otimizadores"]
                    )
                    camadas = st.number_input(
                        textos_otim["inserir_camadas"], min_value=1, max_value=3, value=1
                    )
                    rodadas = st.number_input(
                        textos_otim["inserir_rodadas"], min_value=1, value=1
                    )
                    shots = st.number_input(
                        textos_otim["inserir_shots"], min_value=100, value=1000
                    )
                
        if st.button(textos_otim['executar']):

            # Verifica o modo leitura escolhido (upload/manual)
            if modo_leitura == textos_otim['modo_leitura_upload']:
                instancia = dados[0]  # Dados do upload
            else:
                instancia = dados     # Dados da entrada manual
        
            # Extrai variáveis da instância
            s = instancia[0]
            nj_max = instancia[1]
            nj_min = instancia[2]
            ctj_of = instancia[3]
            Rjk_of = instancia[4]
            cjk_of = instancia[5]
            C_of = instancia[6]

            x = nj_max
            nmax = x

            i = 0
            b = []

            while x != 0:
                b.append(x % 2)
                x = np.floor(x / 2)
                i = i + 1
            nb = i

            ct = ctj_of
            Rjk = Rjk_of
            cjk = cjk_of
            C = C_of

            v = len(Rjk)

            qp = QuadraticProgram()

            for j in range(1, v + 1):
                for k in range(i):
                    var_name = f"b{k}{j}"
                    qp.binary_var(name=var_name)
            
            num_vars = len(qp.variables)

            linear_terms = {}

            for j in range(1, v + 1):
                for k in range(i):
                    linear_terms[f'b{k}{j}'] = np.log(1 - Rjk[j - 1]) * (2 ** (i - k - 1))

            qp.minimize(linear=linear_terms)

            constraint_terms = {}
            for j in range(1, v + 1):
                for k in range(i):
                    constraint_terms[f'b{k}{j}'] = cjk[j - 1] * (2 ** (i - k - 1))

            qp.linear_constraint(linear=constraint_terms, sense='<=', rhs=C, name='constraint_1')

            constraint_terms2 = {}
            for j in range(1, v + 1):
                for k in range(i):
                    constraint_terms2[f'b{k}{j}'] = (2 ** (i - k - 1))

            qp.linear_constraint(linear=constraint_terms2, sense='>=', rhs=1, name='constraint_2')

            constraint_terms3 = {}
            for j in range(1, v + 1):
                for k in range(i):
                    constraint_terms3[f'b{k}{j}'] = (2 ** (i - k - 1))

            qp.linear_constraint(linear=constraint_terms3, sense='<=', rhs=nmax, name='constraint_3')

            ineq2eq = InequalityToEquality()
            qp_eq = ineq2eq.convert(qp)

            int2bin = IntegerToBinary()
            qp_eq_bin = int2bin.convert(qp_eq)

            lineq2penalty = LinearEqualityToPenalty()
            qubo = lineq2penalty.convert(qp_eq_bin)

            qubits = np.array(qubo.variables)
            qubits = qubits.shape[0]

            op, offset = qubo.to_ising()
            
            if modo_algoritmo == 'QAOA':

                time_qaoa = 0
                energias = []
                parametros = []
                tempos_execucao = []
                componentes_otimos = [] 

                for i in range(rodadas):
                    for j in range(camadas):
                        if tipo_inicializacao == textos_otim["tipos_inicializacao_qaoa"][1]:  # LHS
                            param_intervals = [(0, 2*np.pi)] * 2 
                            lhs_samples = generate_lhs_samples(param_intervals, rodadas+1)
                            params = lhs_samples[i]
                        elif tipo_inicializacao == textos_otim["tipos_inicializacao_qaoa"][2]:
                            params = np.random.uniform(0, 2 * np.pi, 2)
                        elif tipo_inicializacao == textos_otim["tipos_inicializacao_qaoa"][3]:  # Ponto Fixo / Fixed Point
                            params = np.full(2, numero_ponto_fixo)
                        elif tipo_inicializacao == textos_otim["tipos_inicializacao_qaoa"][0]:
                            K = 2
                            Q = 56  

                            kmeans = KMeans(n_clusters=K)
                            cluster_labels = kmeans.fit_predict(parametros_treino)

                            random_cluster = np.random.randint(0, K)
                            cluster_indices = np.where(cluster_labels == random_cluster)[0]
                            closest_point_index = np.random.choice(cluster_indices)
                            params = parametros_treino[closest_point_index]
                                        
                        st.write("---")
                        st.write(f"{textos_otim['parametros_iniciais']} - {textos_otim['rodada']} {i+1} : {textos_otim['camada']} {j+1} = {', '.join(map(str, params))}")
                        loading_placeholder = st.empty()
                        
                        loading_placeholder.markdown(
                            f"""
                            <div style='display: flex; flex-direction: column; align-items: center; justify-content: center;'>
                                <div class='loading-gif'>
                                    <img src='https://th.bing.com/th/id/R.4e7379292ef4b8d1945b1c3bc628d00d?rik=1iNOSJvqT0k%2bww&riu=http%3a%2f%2fbookrosabv.com.br%2fimagens%2floader.gif&ehk=OOTFpItH%2fvfYkf4YThgEExBU9BILk0f4c629HC36vTI%3d&risl=&pid=ImgRaw&r=0' 
                                    alt='Carregando...' width='100'>
                                </div>
                                <div class='loading-text' style='margin-top: 10px; font-size:18px;'>
                                    {textos_otim['executando_qaoa']}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                        st.markdown(f"<div class='counter'>{textos_otim['rodada']} {i + 1} / {rodadas}</div>", unsafe_allow_html=True)
                        algorithm_globals.random_seed = 10598

                        if otimizador == textos_otim["opcoes_otimizadores"][0]:  # SPSA
                            otimizador_instanciado = SPSA()
                        elif otimizador == textos_otim["opcoes_otimizadores"][1]:  # COBYLA
                            otimizador_instanciado = COBYLA()

                        sampler = Sampler(options={"shots": shots})
                        mes = QAOA(sampler=Sampler(), optimizer= otimizador_instanciado, initial_point=params)
                        meo = MinimumEigenOptimizer(min_eigen_solver=mes)

                        start = time.time()
                        qaoa_result = meo.solve(qubo)
                        end = time.time()

                        energias.append(qaoa_result.fval)
                        tempos_execucao.append(end - start)
                        componentes_otimos.append(qaoa_result.x)
                        st.write(qaoa_result)
                        

                energia_otimizada = min(energias)
                confiabilidade = 1 - math.exp(energia_otimizada)
                media_energia = np.mean(energias)
                desvio_padrao_energia = np.std(energias)

                indice_min_energia = energias.index(energia_otimizada)
                componente_otimo = componentes_otimos[indice_min_energia]

                #st.write("Configuração ótima dos componentes:")
                #st.write(componente_otimo)
                componentes_variaveis = []

                f= ct
                d= nb
                
                var_index = 0
                for m in range(1, f + 1):
                    componente = 0
                    for k in range(d):
                        var_value = componente_otimo[var_index]
                        componente += var_value * (2 ** (m - k - 1))
                        var_index += 1 
                    componentes_variaveis.append(componente)
                
                pesos = cjk
                custo_total = sum(c * p for c, p in zip(componentes_variaveis, pesos))
                componentes_formatados = [int(v) for v in componentes_variaveis]
                                                
                loading_placeholder.empty()  # Remove the loading GIF
                st.subheader(textos_otim['resultados'])
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(label=textos_otim['energia_otima'], value=round(energia_otimizada, 4))
                    st.metric(label=textos_otim['confiabilidade_otima'], value=round(confiabilidade, 4))

                with col2:
                    st.metric(label=textos_otim['custo_total'], value=custo_total)
                    st.markdown(
                        f"""
                        <div>
                            <span style="font-size: 16px; font-weight: normal;">{textos_otim['componentes_solucao']}</span><br>
                            <span style="font-size: 32px; font-weight: normal; margin-left: 0;">{componentes_formatados}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                st.subheader(textos_otim['medidas_energia'])
                st.markdown(f"**{textos_otim['media_energia']}:** {round(media_energia, 4)}")
                st.markdown(f"**{textos_otim['desvio_padrao_energia']}:** {round(desvio_padrao_energia, 4)}")

            elif modo_algoritmo == 'VQE':
                time_vqe = 0
                energias = []
                parametros = []
                tempos_execucao = []

                for i in range(rodadas):
                    for j in range(camadas):
                        if tipo_circuito == textos_otim["real_amplitudes"]:
                            num_parametros = qubits * 2 * camadas  
                        elif tipo_circuito == textos_otim["two_local"]:
                            num_parametros = (len(rotacao_escolhida)*2) * camadas * qubits 

                        if tipo_inicializacao == textos_otim["tipos_inicializacao_vqe"][0]:  # LHS
                            param_intervals = [(0, 2 * np.pi)] * num_parametros  # Intervalo para cada parâmetro
                            lhs_samples = generate_lhs_samples(param_intervals, rodadas + 1)  # Gerando amostras LHS
                            params = lhs_samples[i]  # Selecionando a amostra correspondente à rodada
                        elif tipo_inicializacao == textos_otim["tipos_inicializacao_vqe"][1]:  # Randômica / Random
                            params = np.random.uniform(0, 2 * np.pi, num_parametros)  # Inicialização randômica
            
                        elif tipo_inicializacao == textos_otim["tipos_inicializacao_vqe"][2]:  # Ponto Fixo / Fixed Point
                            params = np.full(num_parametros, numero_ponto_fixo)  # Inicialização com valor fixo
            
                        st.write("---")
                        st.write(f"{textos_otim['parametros_iniciais']} - {textos_otim['rodada']} {i+1} : {textos_otim['camada']} {j+1} = {', '.join(map(str, params))}")
            
                        loading_placeholder = st.empty()
                        loading_placeholder.markdown(f"""
                        <div style='display: flex; flex-direction: column; align-items: center; justify-content: center;'>
                            <div class='loading-gif'>
                                <img src='https://th.bing.com/th/id/R.4e7379292ef4b8d1945b1c3bc628d00d?rik=1iNOSJvqT0k%2bww&riu=http%3a%2f%2fbookrosabv.com.br%2fimagens%2floader.gif&ehk=OOTFpItH%2fvfYkf4YThgEExBU9BILk0f4c629HC36vTI%3d&risl=&pid=ImgRaw&r=0' 
                                alt='Carregando...' width='100'>
                            </div>
                            <div class='loading-text' style='margin-top: 10px; font-size:18px;'>
                                {textos_otim['executando_vqe']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
                        st.markdown(f"<div class='counter'>{textos_otim['rodada']} {i + 1} / {rodadas}</div>", unsafe_allow_html=True)
            
                        algorithm_globals.random_seed = 10598

                        if tipo_circuito ==textos_otim["real_amplitudes"]:
                            variational_circuit = RealAmplitudes(qubits, reps=camadas)
                        elif tipo_circuito == textos_otim["two_local"]: 
                            variational_circuit = TwoLocal(qubits, rotacao_escolhida, entanglement_escolhido, reps=camadas)

                        if otimizador == textos_otim["opcoes_otimizadores"][0]:  # SPSA
                            otimizador_instanciado = SPSA()
                        elif otimizador == textos_otim["opcoes_otimizadores"][1]:  # COBYLA
                            otimizador_instanciado = COBYLA()

                        sampler = Sampler(options={"shots": shots})
                        mes = SamplingVQE(sampler=Sampler(), ansatz=variational_circuit, optimizer=otimizador_instanciado, initial_point=params)
                        meo = MinimumEigenOptimizer(min_eigen_solver=mes)
            
                        start = time.time()
                        vqe_result = meo.solve(qubo)
                        end = time.time()
            
                        energias.append(vqe_result.fval)
                        tempos_execucao.append(end - start)
                        componentes_otimos.append(vqe_result.x)

                        if i == (rodadas-1): 
                            st.subheader(textos_otim["circuito_quantico"])
                            vqe_circuit = mes.ansatz
                            fig = plt.figure(figsize=(6, 8)) 
                            ax = fig.add_subplot(111)
                            vqe_circuit.draw(output='mpl', ax=ax)
                            buf = io.BytesIO()
                            plt.savefig(buf, format='png', bbox_inches='tight')
                            buf.seek(0)
                            st.image(buf, caption="VQE",  use_container_width=False)
                            plt.close(fig)
            
                energia_otimizada = min(energias)
                confiabilidade = 1 - math.exp(energia_otimizada)
                media_energia = np.mean(energias)
                desvio_padrao_energia = np.std(energias)
            
                indice_min_energia = energias.index(energia_otimizada)
                componente_otimo = componentes_otimos[indice_min_energia]
            
                componentes_variaveis = []
                f = ct
                d = nb
            
                var_index = 0
                for m in range(1, f + 1):
                    componente = 0
                    for k in range(d):
                        var_value = componente_otimo[var_index]
                        componente += var_value * (2 ** (m - k - 1))
                        var_index += 1 
                    componentes_variaveis.append(componente)
            
                pesos = cjk
                custo_total = sum(c * p for c, p in zip(componentes_variaveis, pesos))
                componentes_formatados = [int(v) for v in componentes_variaveis]
            
                loading_placeholder.empty()
                st.subheader(textos_otim['resultados'])
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(label=textos_otim['energia_otima'], value=round(energia_otimizada, 4))
                    st.metric(label=textos_otim['confiabilidade_otima'], value=round(confiabilidade, 4))

                with col2:
                    st.metric(label=textos_otim['custo_total'], value=custo_total)
                    st.markdown(
                        f"""
                        <div>
                            <span>{textos_otim['componentes_solucao']}</span><br>
                            <span style="font-size: 32px; font-weight: normal; margin-left: 0;">{componentes_formatados}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )       
                st.subheader(textos_otim['medidas_energia'])
                st.markdown(f"**{textos_otim['media_energia']}:** {round(media_energia, 4)}")
                st.markdown(f"**{textos_otim['desvio_padrao_energia']}:** {round(desvio_padrao_energia, 4)}")
        
        with st.sidebar:
            if st.button(textos["ini"]):
                st.session_state['pagina'] = 'inicio'
                    

    elif st.session_state['pagina'] == 'explicacao_otimizacao':
        st.title("📘 Explicação sobre Otimização")
        st.markdown("""
        **O que é otimização?**  
        Otimização é o processo de ajustar variáveis para encontrar a melhor solução possível dentro de um conjunto de restrições...
    
        ### Exemplos de métodos:
        - Programação Linear
        - Algoritmos Genéticos
        - QUBO / Otimização Quântica
        - etc.
        """)
        if st.button("⬅️ Voltar para Otimização"):
            st.session_state['pagina'] = 'otimizacao'

    elif st.session_state['pagina'] == 'ml':
        st.subheader(textos["pagina_ml"])
        
        # Agora usa os textos com a função
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(textos_ml["dataset_opcao"])
            dataset_opcao = st.selectbox(textos_ml["selecione_base"], [" - ", "CWRU", "JNU"], help=textos_ml["help_1"])
        
        with col2:
            st.markdown(textos_ml["upload_dados"])
            uploaded_file = st.file_uploader(textos_ml["upload_label"], type=["csv", "xlsx", "parquet"], help=textos_ml["help_2"])
            
        
            if uploaded_file is not None:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.parquet'):
                    df = pd.read_parquet(uploaded_file)
                else:
                    st.error("Unsupported file format.")
                    df = None
        
                if df is not None:
                    st.success(textos_ml["upload_sucesso"])
                    st.write(textos_ml["preview"])
                    st.dataframe(df.head())
            
        st.divider()
        
        # === FEATURES ===
        st.markdown(textos_ml["selecione_features"])
        features = [
            " - ", "Média", "Variância", "Desvio-padrão", "RMS", "Kurtosis",
            "Peak to peak", "Max Amplitude", "Min Amplitude", "Skewness",
            "CrestFactor", "Mediana", "Energia", "Entropia"
        ]
        selected_features = st.multiselect(textos_ml["label_features"], options=features, help=textos_ml["help_3"])
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(textos_ml["encoding_title"])
            encoding_method = st.selectbox(textos_ml["encoding_label"], [
                " - ", "Angle encoding", "Amplitude encoding",
                "ZFeaturemap", "XFeaturemap", "YFeaturemap", "ZZFeaturemap"
            ], help=textos_ml["help_4"])
        
            st.markdown(textos_ml["euler_title"])
            rot = st.selectbox(textos_ml["euler_label"], [" - ", "1", "2", "3"], help=textos_ml["help_5"])
        
            if rot == "1":
                eixos = [st.selectbox(f"**{textos_ml['euler_eixo1']}**", [" - ", "X", "Y", "Z"], help=textos_ml["help_6"])]
            elif rot == "2":
                eixos = [
                    st.selectbox(f"**{textos_ml['euler_eixo_n'].format(n=1)}**", [" - ", "X", "Y", "Z"], help=textos_ml["help_6"]),
                    st.selectbox(f"**{textos_ml['euler_eixo_n'].format(n=2)}**", [" - ", "X", "Y", "Z"], help=textos_ml["help_6"])
                ]
            elif rot == "3":
                eixos = [
                    st.selectbox(f"**{textos_ml['euler_eixo_n'].format(n=1)}**", [" - ", "X", "Y", "Z"], help=textos_ml["help_6"]),
                    st.selectbox(f"**{textos_ml['euler_eixo_n'].format(n=2)}**", [" - ", "X", "Y", "Z"], help=textos_ml["help_6"]),
                    st.selectbox(f"**{textos_ml['euler_eixo_n'].format(n=3)}**", [" - ", "X", "Y", "Z"], help=textos_ml["help_6"])
                ]
            else:
                eixos = []
        
        with col2:
            if encoding_method.strip() != " - ":
                st.markdown(textos_ml['entanglement_title'])
                entanglement_method = st.selectbox(" ", [" - ", "CZ", "iSWAP", "Real Amplitudes", "QCNN"], help=textos_ml["help_7"])
        
            st.number_input(textos_ml["paciencia"], min_value=0, max_value=400, value=0, step=1)
            st.number_input(textos_ml["epocas"], min_value=1, max_value=500, value=1, step=1)
        
        st.divider()

        import pennylane as qml
        from pennylane import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score
        from sklearn.decomposition import PCA
        from scipy.stats import kurtosis, skew
        import scipy
        import os
        
        # ==== FUNÇÕES DE EXTRAÇÃO DE FEATURES ====
        def extrair_features_amostra(amostra):
            features_calculadas = {}
            features_calculadas["Média"] = np.mean(amostra)
            features_calculadas["Variância"] = np.var(amostra)
            features_calculadas["Desvio-padrão"] = np.std(amostra)
            features_calculadas["RMS"] = np.sqrt(np.mean(np.square(amostra)))
            features_calculadas["Kurtosis"] = kurtosis(amostra)
            features_calculadas["Peak to peak"] = np.ptp(amostra)
            features_calculadas["Max Amplitude"] = np.max(amostra)
            features_calculadas["Min Amplitude"] = np.min(amostra)
            features_calculadas["Skewness"] = skew(amostra)
            features_calculadas["CrestFactor"] = np.max(np.abs(amostra)) / (np.sqrt(np.mean(np.square(amostra))) + 1e-10)
            features_calculadas["Mediana"] = np.median(amostra)
            features_calculadas["Energia"] = np.sum(amostra ** 2)
            prob, _ = np.histogram(amostra, bins=30, density=True)
            prob = prob[prob > 0]
            features_calculadas["Entropia"] = -np.sum(prob * np.log(prob))
            return features_calculadas
        
        def extrair_features_dataset(dataset_bruto, selected_features):
            lista_dicts = []
            for amostra in dataset_bruto:
                f = extrair_features_amostra(amostra)
                f_sel = {key: f[key] for key in selected_features}
                lista_dicts.append(f_sel)
            return pd.DataFrame(lista_dicts)
        
        
        
        # ==== FUNÇÃO PARA CARREGAR DADOS BRUTOS ====
        def carregar_dados_brutos(nome):
            # Para usar bases reais, substitua essa parte pelo carregamento real.
            if nome == "CWRU":
                df = pd.DataFrame(columns=['DE_data', 'fault'])
        
                for root, dirs, files in os.walk(r"C:\\Arthur\\load_12K", topdown=False):
                    for file_name in files:
                        path = os.path.join(root, file_name)
        
                        mat = scipy.io.loadmat(path)
        
                        key_name = list(mat.keys())[3]
                        DE_data = mat.get(key_name)
                        fault = np.full((len(DE_data), 1), file_name[:-4])
        
                        df_temp = pd.DataFrame(
                            {'DE_data': np.ravel(DE_data), 'fault': np.ravel(fault)})
        
                        df = pd.concat([df, df_temp], axis=0)
                        # print(df['fault'].unique())
        
        
                df.to_csv(r'todas_faltas.csv', index=False)
        
                df = pd.read_csv('todas_faltas.csv')
        
                win_len = 1000
                stride = 900
        
                x = []
                y = []
        
        
                for k in df['fault'].unique():
        
                    df_temp_2 = df[df['fault'] == k]
        
                    for i in np.arange(0, len(df_temp_2)-(win_len), stride):
                        temp = df_temp_2.iloc[i:i+win_len, :-1].values
                        temp = temp.reshape((1, -1))
                        x.append(temp)
                        y.append(df_temp_2.iloc[i+win_len, -1])
        
                x = np.array(x)
                x = x.reshape((x.shape[0], win_len))
                y = np.array(y)
        
                return x, y
            
            elif nome == "JNU":
                dataset = np.load(r"C:\\Arthur\\JNU_quantum_8.npz")
                X = dataset['data']
                y = dataset['label']
        
                return X, y
            else:
                return None, None
        
        def selecionar_features(X, features, selecionadas):
            indices = [features.index(f) for f in selecionadas]
            return X[:, indices]
        
        
        
        # --- CIRCUITOS DE ENCODING ---
        def angle_encoding(x, wires, eixos):
            # Aplica rotações baseadas nos eixos fornecidos
            for i, wire in enumerate(wires):
                for eixo in eixos:
                    if eixo == "X":
                        qml.RX(x[i], wires=wire)
                    elif eixo == "Y":
                        qml.RY(x[i], wires=wire)
                    elif eixo == "Z":
                        qml.RZ(x[i], wires=wire)
        
        def amplitude_encoding(x, wires):
            qml.AmplitudeEmbedding(features=x, wires=wires, normalize=True)
        
        def z_featuremap(x, wires):
            for i, wire in enumerate(wires):
                qml.RZ(x[i], wires=wire)
        
        def x_featuremap(x, wires):
            for i, wire in enumerate(wires):
                qml.RX(x[i], wires=wire)
        
        def y_featuremap(x, wires):
            for i, wire in enumerate(wires):
                qml.RY(x[i], wires=wire)
        
        def zz_featuremap(x, wires):
            for i, wire in enumerate(wires):
                qml.RZ(x[i], wires=wire)
            for i in range(len(wires) - 1):
                qml.CNOT(wires=[wires[i], wires[i+1]])
        
        
        
        # --- CRIAÇÃO DO CIRCUITO PQC ---
        def criar_circuito(encoding_method, eixos, entanglement_gate, n_qubits):
            dev = qml.device("default.qubit", wires=n_qubits)
        
            @qml.qnode(dev)
            def circuit(x, weights):
                # Encoding
                if encoding_method == "Angle encoding":
                    angle_encoding(x, wires=range(n_qubits), eixos=eixos)
                elif encoding_method == "Amplitude encoding":
                    amplitude_encoding(x, wires=range(n_qubits))
                elif encoding_method == "ZFeaturemap":
                    z_featuremap(x, wires=range(n_qubits))
                elif encoding_method == "XFeaturemap":
                    x_featuremap(x, wires=range(n_qubits))
                elif encoding_method == "YFeaturemap":
                    y_featuremap(x, wires=range(n_qubits))
                elif encoding_method == "ZZFeaturemap":
                    zz_featuremap(x, wires=range(n_qubits))
                else:
                    pass  # Nenhuma codificação
                
                # Camada parametrizada - camada simples com RX, RY, RZ com pesos
                for i in range(n_qubits):
                    qml.RX(weights[i, 0], wires=i)
                    qml.RY(weights[i, 1], wires=i)
                    qml.RZ(weights[i, 2], wires=i)
        
                # Emaranhamento (exemplo simples)
                if entanglement_gate == "CZ":
                    for i in range(n_qubits - 1):
                        qml.CZ(wires=[i, i+1])
                elif entanglement_gate == "iSWAP":
                    for i in range(n_qubits - 1):
                        qml.ISWAP(wires=[i, i+1])
                elif entanglement_gate == "Real Amplitudes":
                    qml.templates.layers.RealAmplitudes(weights, wires=range(n_qubits))
                elif entanglement_gate == "QCNN":
                    # Coloque aqui o seu template QCNN se quiser
                    pass
                
                # Medição
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
            return circuit

        # --- EXECUÇÃO DO MODELO ---
        def executar_teste():
        
            if dataset_opcao == " - " and uploaded_file is None:
                st.error(textos_ml['erro_1'])
                return
            if len(selected_features) == 0:
                st.error(textos_ml['erro_2'])
                return
            if encoding_method == " - ":
                st.error(textos_ml['erro_3'])
                return
            if len(eixos) == 0:
                st.error(textos_ml['erro_4'])
                return
        
            # Carrega dados
            if dataset_opcao != " - ":
                # Ajeitar o local dos dados
                # X, y = carregar_dados_brutos(dataset_opcao)
                if X is None or y is None:
                    st.error(textos_ml['erro_5'])
                    return
            
            else:
                X = df.drop(columns='label').values
                y = df['label'].values
        
            st.success(textos_ml['exec_1'])
            # Seleciona features
            X_sel = extrair_features_dataset(X, selected_features)
            
            x_train, x_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.2, random_state=1)
        
            # Pré-processa (normalização)
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
            x_train = np.array([x / np.linalg.norm(x) for x in x_train])
            x_test = np.array([x / np.linalg.norm(x) for x in x_test])
        
            # Define número de qubits como o número de features selecionadas
            n_qubits = x_train.shape[1]
            
            # Pesos aleatórios para camada parametrizada
            weights = np.random.uniform(low=0, high=2 * np.pi, size=(n_qubits, 3), requires_grad=True)
        
            # Porta de emaranhamento escolhida (default CZ)
            entanglement_gate = entanglement_method
        
            # Cria circuito
            circuit = criar_circuito(encoding_method, eixos, entanglement_gate, n_qubits)
        
            # Executa circuito para todo dataset (exemplo: só execução direta, sem treino)
            resultados = np.array([circuit(x, weights) for x in x_train])
        
            # Como exemplo simples, usa saída quântica para treino SVM
            X_features = resultados
            X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
        
        
            clf = SVC()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
        
            st.write(textos_ml['acc'])
    
        
        if st.button(textos_ml['exec_2']):
            executar_teste()


        
        with st.sidebar:
            if st.button(textos["ini"]):
                st.session_state['pagina'] = 'inicio'

    elif st.session_state['pagina'] == 'inferencia':
        st.subheader(textos["pagina_inferencia"])
        st.write("Lavínia")

        with st.sidebar:
            if st.button(textos["ini"]):
                st.session_state['pagina'] = 'inicio'

    elif st.session_state['pagina'] == 'referencias':
        mostrar_referencias(textos, textos_otim)

        with st.sidebar:
            if st.button(textos["ini"]):
                st.session_state['pagina'] = 'inicio'

if __name__ == "__main__":
    main()
