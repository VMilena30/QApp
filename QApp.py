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
import streamlit.components.v1 as components
import re
import sqlite3
from datetime import datetime
import os
from datetime import datetime
import csv
#Adicionado por Lavínia - comentário para controle
import itertools
import json
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd

st.set_page_config(
    page_title="qPrism",
    page_icon="pesq.png",
    layout="wide"
)

ADMIN_EMAILS = [
    "vmilena3010@gmail.com",
    "lav@ufpe.br",
]

TEXTOS_LOGIN = {
  "en": {
    "welcome_msg": "Welcome!",
    "select_lang": "Select a language",
    "tagline": "Quantum Platform for Reliability: Inference, Systems modeling, and Machine learning",
    "access_title": "Access qPrism",
    "access_sub": "Please fill in the form to continue.",
    "name_opt": "Name (optional)",
    "email_req": "Email *",
    "country_req": "Country *",
    "company_req": "Company / Institution *",
    "role_opt": "Role / Position (optional)",
    "continue": "Continue",
    "err_required": "Please provide email, country, and company.",
    "err_email": "Invalid email address.",
    "err_send": "Could not send verification email. Check SMTP secrets."
  },
  "pt": {
    "welcome_msg": "Bem-vindo(a)!",
    "select_lang": "Selecione um idioma",
    "tagline": "Quantum Platform for Reliability: Inference, Systems modeling, and Machine learning",
    "access_title": "Acessar o qPrism",
    "access_sub": "Preencha o formulário para continuar.",
    "name_opt": "Nome (opcional)",
    "email_req": "E-mail *",
    "country_req": "País *",
    "company_req": "Empresa / Instituição *",
    "role_opt": "Cargo / Função (opcional)",
    "continue": "Continuar",
    "err_required": "Informe e-mail, país e instituição.",
    "err_email": "E-mail inválido.",
    "err_send": "Não foi possível enviar o e-mail. Verifique as secrets SMTP."
  }
}


def load_logo_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = load_logo_base64("qpb.png")

BAR_COLOR = "#0d4376"
BAR_HEIGHT = 64

st.markdown(
    f"""
    <style>
        header[data-testid="stHeader"] {{
            background: transparent;
        }}

        .qx-topbar {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: {BAR_HEIGHT}px;
            background-color: {BAR_COLOR};
            display: flex;
            align-items: center;
            padding: 0 28px;
            z-index: 1000;
            box-sizing: border-box;
        }}

        .qx-topbar img {{
            height: 36px;
            margin-right: 12px;
        }}

        .qx-title {{
            color: white;
            font-size: 32px;
            font-weight: 700;
            font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
        }}

        section[data-testid="stMain"] {{
            padding-top: {BAR_HEIGHT - 15}px;
        }}

        aside[data-testid="stSidebar"] {{
            top: {BAR_HEIGHT}px;
            height: calc(100vh - {BAR_HEIGHT}px);
            z-index: 1001;
        }}

        aside[data-testid="stSidebar"] > div {{
            padding-top: 0px;
        }}
    </style>

    <div class="qx-topbar">
        <img src="data:image/png;base64,{logo_base64}">
        <div class="qx-title">qPrism</div>
    </div>
    """,
    unsafe_allow_html=True
)



LOG_DIR = "registros"
os.makedirs(LOG_DIR, exist_ok=True)

DB_PATH = os.path.join(LOG_DIR, "qxplore_users.db")
CSV_PATH = os.path.join(LOG_DIR, "acessos.csv")

import smtplib
from email.message import EmailMessage

def send_otp_email(to_email, otp_code):
    msg = EmailMessage()
    msg["Subject"] = "qPrism – Verification Code"
    msg["From"] = st.secrets["SMTP_FROM"]
    msg["To"] = to_email

    msg.set_content(f"""
Hello,

Your verification code to access qPrism is:

{otp_code}

This code is valid for a short time.
If you did not request this, please ignore this email.

— qPrism Team
""")

    with smtplib.SMTP(st.secrets["SMTP_HOST"], int(st.secrets["SMTP_PORT"])) as server:
        server.starttls()
        server.login(
            st.secrets["SMTP_USER"],
            st.secrets["SMTP_PASS"]
        )
        server.send_message(msg)

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS registrations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT NOT NULL,
        company TEXT NOT NULL,
        role TEXT,
        created_at TEXT NOT NULL
    )
    """)
    con.commit()
    con.close()

def is_valid_email(email: str) -> bool:
    return re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email or "") is not None

def save_registration(name, email, company, role, created_at):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO registrations (name, email, company, role, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (name, email, company, role, created_at))
    con.commit()
    con.close()


def append_csv_log(name, email, company, role, created_at):
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["created_at", "name", "email", "company", "role"])
        w.writerow([created_at, name, email, company, role])

init_db()

import random

def generate_otp():
    return str(random.randint(100000, 999999))
        
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
        "pagina_ml2": "Aprendizagem de Máquina Quântica em problemas de classificação e predição de falhas",
        "instancia_input": "Digite alguma coisa para testar a instância:",
        "instancia_recebida": "Instância recebida:",
        "idioma": "Escolha o idioma:", 
        "referencias_titulo": "Referências",
        "referencias_intro": "Para conhecer mais sobre nossos trabalhos nas áreas, consulte as referências abaixo:",
        "info_ml": "Seção para descrever as técnicas de Machine Learning Quântico usadas.",
        "info_inf": "Seção para descrever as técnicas de Inferância Quântica usadas.",
        "titulo": "Seja bem-vindo ao <span style='color:#0d4376;'>qPrism</span>!",
        "corpo": (
            "O qPrism é um aplicativo focado em apoiar o estudo e a experimentação com computação quântica aplicada a problemas comuns da engenharia da confiabilidade.\n\n"
            "Ele oferece três áreas principais onde você pode explorar como métodos quânticos podem ser usados para modelar e analisar desafios em sistemas e processos de confiabilidade.\n\n"
            "Embora a tecnologia quântica ainda esteja em desenvolvimento, este aplicativo traz ferramentas e exemplos que ajudam a entender seu funcionamento e seu potencial, mesmo que ainda de forma exploratória, para problemas de engenharia.\n\n"
            "Explore as áreas disponíveis para conhecer melhor essa tecnologia e como ela pode ser aplicada a casos reais."
        ),
        "ini": "Página inicial", 
        "pagina_referencias": "Referências",
        "pagina_info": "Ajuda",
        "inf_ref": "Ajuda e Referências",
        "ref": "Escolha um módulo para acessar as referências e materiais relacionados.",
        "pagina_info2": "Informação sobre conceitos nas três áreas",
        "inf1": "Estimação de Confiabilidade com QBN:",
        "inf2": "Módulo voltado para análise de confiabilidade utilizando Quantum Bayesian Networks (QBNs). Permite integrar variáveis discretas e contínuas em um mesmo modelo, representando eventos e suas dependências probabilísticas de forma unificada.",
        "inf3": "Método:",
        "apoio": "Apoio:",
        "apo": "Agradecimentos às seguintes agências brasileiras de fomento à pesquisa pelo apoio financeiro: Conselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq) (processos 409701/2022-0, 305198/2025-4, 310892/2022-8, 402761/2023-5 e 201271/2024-9); Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES) (Código de Financiamento 001); Programa de Recursos Humanos 38 (PRH 38), intitulado “Gestão de Riscos, Confiabilidade e Sustentabilidade na Exploração e Produção de Petróleo, Gás Natural e Energias Renováveis”, financiado pela Agência Nacional do Petróleo, Gás Natural e Biocombustíveis (ANP) e gerido pela Fundação de Amparo à Pesquisa do Estado de São Paulo (FAPESP) (processo 2025/03497-0); Fundação de Amparo à Ciência e Tecnologia do Estado de Pernambuco (FACEPE) (processo APQ-0504-3.08/25); e a Financiadora de Estudos e Projetos (FINEP).",
        "inf4": "As relações probabilísticas entre nós são implementadas com portas quânticas controladas. A probabilidade de falha é estimada a partir de medições de circuitos quânticos e amplificação de amplitudes, podendo ser comparada com métodos clássicos, como a simulação de Monte Carlo.",
    },
    "en": {
        "intro": "This application was developed to promote the use of quantum computing in three distinct areas, described below.\nSelect the area you want to explore and discover the possibilities offered by this innovative technology.",
        "pagina_otimizacao": "Quantum Optimization",
        "apoio": "Support:",
        "apo": "Acknowledgments are due to the following Brazilian research funding agencies for their financial support: Conselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq) (grants 409701/2022-0, 305198/2025-4, 310892/2022-8, 402761/2023-5, and 201271/2024-9); Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES) (Finance Code 001); Human Resources Program 38 (PRH 38), entitled “Risk Management, Reliability and Sustainability in Oil, Natural Gas and Renewable Energy Exploration and Production”, financed by Agência Nacional do Petróleo, Gás Natural e Biocombustíveis (ANP) and managed by Fundação de Amparo à Pesquisa do Estado de São Paulo (FAPESP) (grant 2025/03497-0); Fundação de Amparo à Ciência e Tecnologia do Estado de Pernambuco (FACEPE) (grant APQ-0504-3.08/25); and Financiadora de Estudos e Projetos (FINEP).",
        "pagina_otimizacao2": "Quantum Optimization in Redundancy Allocation Problems",
        "pagina_inferencia": "Quantum Inference",
        "pagina_ml": "Quantum Machine Learning",
        "pagina_ml2": "Quantum Machine Learning in classification problems and failure prediction",
        "instancia_input": "Type something to test the instance:",
        "instancia_recebida": "Received instance:",
        "idioma": "Choose the language:", 
        "referencias_titulo": "References",
        "inf_ref": "Help and References",
        "ref": "Choose a module to access its references and related materials.",
        "referencias_intro": "To learn more about our work in this areas, check the references below:", 
        "info_ml": "Section describing the Quantum Machine Learning techniques used.",
        "info_inf": "Section describing the Quantum Inference techniques used.",
        "titulo": "Welcome to <span style='color:#0d4376;'>qPrism</span>!",
        "corpo": (
            "qPrism is an application focused on supporting the study and experimentation of quantum computing applied to common problems in reliability engineering.\n\n"
            "It offers three main areas where you can explore how quantum methods can be used to model and analyze challenges in system and process reliability.\n\n"
            "Although quantum technology is still under development, this app provides tools and examples to help you understand its operation and potential, even if exploratory, in engineering problems.\n\n"
            "Explore the available areas to better understand this technology and how it can be applied to real cases."
        ),
        "ini": "Homepage",
        "pagina_referencias": "References",
        "ref": "Escolha um módulo para acessar as referências e materiais relacionados.",
        "pagina_info": "Help",
        "pagina_info2": "Information about concepts in the three areas",
        "inf1": "Reliability Assessment with QBN:",
        "inf2": "Module designed for reliability assessment using Quantum Bayesian Networks (QBNs). It allows the integration of both discrete and continuous variables into a single model, representing events and their probabilistic dependencies in a unified way.",
        "inf3": "Method:",
        "ref": "Choose a module to access its references and related materials.",
        "inf4": "Probabilistic dependencies between nodes are implemented with controlled quantum gates. Failure probability is estimated using Quantum Amplitude Estimation (QAE) and can be compared with classical methods, such as Monte Carlo simulation.",
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
        "descricao_rap": "Módulo dedicado à resolução de problemas combinatórios por meio de algoritmos quânticos de otimização, utilizando formulações baseadas em QUBO (Quadratic Unconstrained Binary Optimization). Essa abordagem permite representar funções-objetivo e restrições na forma de operadores quânticos, explorando a superposição, interferência e o paralelismo quântico para buscar soluções ótimas.",
        "algoritmos": "Problema abordado:",
        "inicializacoes_titulo": "- RAP (Reliability Allocation Problem)",
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
        "s": "Número de subsistemas",
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
            "Seja um sistema com $(s)$ subsistemas, o objetivo é maximizar a confiabilidade total $(R(x))$:\n\n"
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
        "aplicacao": "\n\n Aplicação",
        "info1_titulo": "Guia de Uso – Otimização Quântica para Alocação de Redundâncias",
        "info1": (
            "Este guia tem como objetivo orientar você a preencher corretamente todos os campos da plataforma de otimização quântica, "
            "aplicada ao problema de alocação de redundâncias em sistemas com múltiplos subsistemas, respeitando restrições de custo e confiabilidade. "
            "A plataforma transforma seu problema em uma formulação do tipo QUBO (Quadratic Unconstrained Binary Optimization), "
            "e resolve essa formulação usando algoritmos quânticos variacionais, que dependem de circuitos parametrizados e otimizadores clássicos."
        ),
    
        "info2_titulo": "1. Modo de Entrada dos Dados",
        "info2": (
            "### 1.1 Inserção Manual\n\n"
            "Você pode preencher todos os dados do problema diretamente na tela. Informe:\n"
            "- Número de subsistemas: quantidade de partes no sistema onde você pode alocar redundância.\n"
            "- Valor mínimo e máximo de componentes por subsistema.\n"
            "- Quantidade de tipos de componentes disponíveis: diferentes modelos com custos e confiabilidades distintas.\n"
            "- Confiabilidade e custo de cada componente.\n"
            "- Limite máximo de custo: valor total disponível para uso na alocação."
        ),
        "info21": (
            "### **1.2 Upload de Arquivo (.txt)**\n\n"
            "Você pode importar os dados por meio de um arquivo .txt estruturado conforme o formato exigido pela plataforma.\n\n"
        ),
    
        "info3_titulo": "2. Algoritmos Quânticos de Otimização",
        "info3": (
            "Você pode escolher entre dois algoritmos quânticos variacionais:\n\n"
            "### **2.1 QAOA (Quantum Approximate Optimization Algorithm):**\n\n"
            "Algoritmo ideal para problemas combinatórios formulados como QUBO. Utiliza camadas parametrizadas compostas por operações que codificam o problema e outras que exploram o espaço de soluções. "
            "Essas camadas são ajustadas por parâmetros numéricos otimizados por algoritmos clássicos.\n\n"
            "### **2.2 VQE (Variational Quantum Eigensolver):**\n\n"
            "Inicialmente usado na química quântica, também pode ser aplicado a problemas de otimização. Exige a definição de um circuito ansatz — uma estrutura de portas quânticas que representa o espaço de soluções. "
            "O VQE ajusta os parâmetros desse circuito para minimizar o valor esperado da função objetivo.\n\n"
            "Ao utilizar o VQE, você deverá configurar:\n"
            "- Tipo de circuito (ansatz):\n"
            "  - Real Amplitudes: utiliza apenas rotações Ry. Simples, eficiente e ideal para casos com poucos qubits.\n"
            "  - Two Local: mais complexo e expressivo. Permite maior controle, mas requer mais tempo e recursos.\n"
            "- Portas de rotação disponíveis: Rx, Ry, Rz — definem as transformações unárias dos qubits.\n"
            "- Portas de emaranhamento disponíveis: CX (CNOT), CZ, CRX, CRY, CRZ — definem como os qubits interagem entre si."
        ),
    
        "info4_titulo": "3. Parâmetros Personalizáveis",
        "info4": (
            "### **3.1 Otimizador Clássico:**\n"
            "- COBYLA (Constrained Optimization By Linear Approximations):\n"
            "  Método que não utiliza derivadas, baseado em aproximações lineares. Funciona bem em problemas de baixa dimensão.\n"
            "- SPSA (Simultaneous Perturbation Stochastic Approximation):\n"
            "  Otimizador robusto contra ruído. Estima gradientes com apenas duas avaliações por iteração, sendo útil em ambientes quânticos.\n\n"
    
            "### **3.2 Método de Inicialização:**\n"
            "- Randômica: os parâmetros iniciais são escolhidos aleatoriamente.\n"
            "- LHS (Latin Hypercube Sampling): gera amostras representativas e bem distribuídas do espaço de busca.\n"
            "- Clusterização: usa agrupamento dos dados como ponto de partida mais estruturado.\n"
            "- Ponto Fixo: o usuário informa manualmente os valores iniciais.\n\n"
    
            "### **3.3 Número de Shots:**\n\n"
            "- Define quantas vezes o circuito quântico será executado."
            "- Circuitos são probabilísticos, então mais execuções fornecem uma estimativa mais precisa. \n\n"
            "- Recomenda-se usar valores entre 1000 e 8192.\n\n"
    
            "### **3.4 Número de Camadas (Profundidade do Circuito):**\n\n"
            "- Determina quantas vezes o bloco de operações é repetido no circuito."
            "- Aumentar esse número permite capturar padrões mais complexos, mas também aumenta o tempo de execução.\n\n"
    
            "### **3.5 Número de Rodadas (Iterações):**\n\n"
            "- Define o número de vezes que o otimizador irá atualizar os parâmetros do circuito.\n\n"
            "- Problemas mais difíceis podem requerer mais rodadas para alcançar boa convergência."
        ),
        "help1": "Como você prefere informar os dados do seu sistema para a ferramenta?",
        "help2": "OI", 
        "help3": "OI", 
        "help4": "OI",
        "help5": "OI",
        "help6": "OI",

        
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
        "descricao_rap": "Módulo dedicado à resolução de problemas combinatórios por meio de algoritmos quânticos de otimização, utilizando formulações baseadas em QUBO (Quadratic Unconstrained Binary Optimization). Essa abordagem permite representar funções-objetivo e restrições na forma de operadores quânticos, explorando a superposição, interferência e o paralelismo quântico para buscar soluções ótimas.",
        "algoritmos": "Problema abordado:",
        "inicializacoes_titulo": "- RAP (Reliability Allocation Problem)",

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
        "info1_titulo": "User Guide – Quantum Optimization for Redundancy Allocation",
        "info1": (
            "This guide is designed to help you correctly fill out all fields of the quantum optimization platform, "
            "applied to the problem of redundancy allocation in systems with multiple subsystems, under cost and reliability constraints. "
            "The platform transforms your problem into a QUBO (Quadratic Unconstrained Binary Optimization) formulation and solves it using "
            "variational quantum algorithms, which rely on parameterized circuits and classical optimizers."
        ),
    
        "info2_titulo": "1. Data Input Mode",
        "info2": (
            "1.1 Manual Input\n"
            "You can directly fill in all the required fields on the screen. Provide:\n"
            "- Number of subsystems: parts of the system where redundancy can be allocated.\n"
            "- Minimum and maximum number of components per subsystem.\n"
            "- Number of component types available: different models with specific cost and reliability values.\n"
            "- Reliability and cost for each component.\n"
            "- Maximum cost limit: total budget available for component allocation."
        ),
        "info2.1": (
            "1.2 File Upload (.txt)\n"
            "You may import data from a structured .txt file formatted according to the platform's specifications. "
            "Recommended for large-scale problems or when reusing predefined configurations."
        ),
    
        "info3_titulo": "2. Quantum Optimization Algorithms",
        "info3": (
            "You can choose between two variational quantum algorithms:\n\n"
            "2.1 QAOA (Quantum Approximate Optimization Algorithm):\n"
            "An algorithm designed for combinatorial problems expressed as QUBO. It uses parameterized layers composed of cost and mixing operations. "
            "These layers are tuned via classical optimization routines.\n\n"
            "2.2 VQE (Variational Quantum Eigensolver):\n"
            "Originally developed for quantum chemistry, VQE can also solve optimization problems by adjusting a quantum circuit called an ansatz. "
            "The ansatz structure defines how qubits are manipulated and how expressively the solution space is represented.\n\n"
            "If using VQE, you must configure:\n"
            "- Type of circuit (ansatz):\n"
            "  - Real Amplitudes: simple circuit using Ry rotations only.\n"
            "  - Two Local: more flexible and expressive structure, suitable for complex problems.\n"
            "- Available rotation gates: Rx, Ry, Rz — define single-qubit transformations.\n"
            "- Available entanglement gates: CX (CNOT), CZ, CRX, CRY, CRZ — define inter-qubit interactions."
        ),
    
        "info4_titulo": "3. Customizable Parameters",
        "info4": (
            "3.1 Classical Optimizer:\n"
            "- COBYLA (Constrained Optimization By Linear Approximations):\n"
            "  A gradient-free optimizer based on linear approximations. Lightweight and efficient for problems with few variables.\n"
            "- SPSA (Simultaneous Perturbation Stochastic Approximation):\n"
            "  A robust optimizer for noisy environments. It estimates gradients with just two evaluations per iteration.\n\n"
    
            "3.2 Initialization Method:\n"
            "- Random: parameters are initialized randomly.\n"
            "- LHS (Latin Hypercube Sampling): samples the parameter space in a balanced and distributed way.\n"
            "- Clustering: uses data grouping to propose an informed initial point.\n"
            "- Fixed Point: manually set the initial parameter values.\n\n"
    
            "3.3 Number of Shots:\n"
            "-Defines how many times the quantum circuit will be executed."
            "-Since quantum circuits are probabilistic, more shots yield better statistical estimates. Suggested values: between 1000 and 8192.\n\n"
    
            "3.4 Number of Layers (Circuit Depth):\n"
            "Represents how many times the block of operations is repeated in the circuit. More layers allow capturing complex patterns "
            "but increase runtime and overfitting risk.\n\n"
    
            "3.5 Number of Iterations (Rounds):\n"
            "Defines how many times the optimizer will update the circuit parameters. Start with 5 to 10 for initial testing, "
            "and increase if needed for better convergence."
        ), 
        "help1": "How would you like to provide your system data to the tool?",
        "help2": "OI", 
        "help3": "OI", 
        "help4": "OI",
        "help5": "OI",
        "help6": "OI",
        

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
        "help_7": "Nessa etapa você vai selecionar o tipo de emaranhamento quântico que gostaria de adicionar no circuito.",
        "info1_titulo": "Guia de Uso",
        "info1": "Esse guia de uso visa facilitar o uso de técnicas de Aprendizagem de Máquina Quântica em problemas de classificação e predição de falhas em equipamentos rotativos, como rolamentos. Mesmo sem conhecimento prévio em computação quântica, você poderá explorar os dados e configurar modelos com apenas alguns cliques. Abaixo, explicamos cada parte da interface.",
        "info2_titulo": "1. Escolha ou envio de base de dados",
        "info2": "Logo na tela inicial, você verá duas opções:",
        "info2.1": "**Selecionar uma base existente:** Você pode escolher entre bases de dados já conhecidas contendo medições de vibração de rolamentos, como a base da Universidade de CWRU ou da JNU.",
        "info2.2": "**Importar dados próprios:** Caso possua seus próprios dados (em formato CSV, Excel ou Parquet), é possível fazer o upload diretamente no sistema.",
        "info3_titulo": "2. Extração de características",
        "info3": "Depois de carregar sua base de dados com sinais de vibração, o aplicativo oferece a possibilidade de extrair automaticamente características (features) desses sinais. Essa etapa simplifica os dados e reduz a quantidade de informação a ser processada, tornando o modelo mais eficiente e interpretável.",
        "info4_titulo": "3. Codificação quântica",
        "info4": "Computadores quânticos não processam dados do mesmo modo que os computadores clássicos. Para que os dados possam ser utilizados em um modelo quântico, eles precisam ser convertidos em uma linguagem compreensível para os qubits. Essa conversão é chamada de codificação quântica. Cada método tem suas particularidades e pode influenciar na performance do modelo.",
        "info5_titulo": "4. Circuitos Quânticos Parametrizados (PQC)",
        "info5": "Os circuitos quânticos parametrizados, também chamados de Parametrized Quantum Circuits (PQC), são o núcleo dos modelos de Aprendizagem de Máquina Quântica. Eles funcionam como o “cérebro” do algoritmo, sendo responsáveis por aprender padrões a partir dos dados inseridos. A ideia central é que o circuito possui portas quânticas ajustáveis por parâmetros numéricos, que são modificados durante o treinamento para encontrar as melhores configurações.",
        "info5.1": "Esses parâmetros são análogos aos “pesos” de uma rede neural clássica. Durante o aprendizado, o modelo ajusta os valores dessas portas para minimizar os erros de classificação ou predição.",
        "info6_titulo": "4.1. Rotações de Euler",
        "info6": "Uma parte essencial dos PQCs são as chamadas rotações de Euler, que são operações aplicadas a cada qubit individualmente. Elas alteram o estado do qubit com base em ângulos parametrizáveis (por exemplo: Rx(θ), Ry(θ), Rz(θ)), permitindo que o circuito represente uma ampla gama de transformações.",
        "info6.1": "Você pode configurar a quantidade de rotações de Euler aplicadas em cada camada do circuito. Aumentar esse número torna o circuito mais expressivo (capaz de representar padrões mais complexos), mas também pode aumentar o tempo de execução e o risco de sobreajuste (overfitting).",
        "info7_titulo": "4.2. Emaranhamento entre Qubits",
        "info7": "Além das operações em qubits isolados, é fundamental que o circuito quântico conecte os qubits entre si. Isso é feito por meio de portas de emaranhamento, que criam correlações entre os estados dos qubits. O emaranhamento é uma das propriedades mais poderosas da computação quântica, pois permite representar relações complexas entre os dados.",
        "info7.1": "Você pode selecionar qual porta deseja usar no seu circuito. O tipo de emaranhamento afeta diretamente a arquitetura do modelo e pode influenciar sua capacidade de generalização."
    },
    "en": {
        "pagina_ml": "Quantum Machine Learning",
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
        "help_7": "At this stage, you will select the type of quantum entanglement you would like to add to the circuit.",
        "info1_titulo": "Usage Guide",
        "info1": "This user guide aims to facilitate the use of Quantum Machine Learning techniques in classification and failure prediction problems in rotating equipment, such as bearings. Even without prior knowledge of quantum computing, you will be able to explore the data and configure models with just a few clicks. Below, we explain each part of the interface.",
        "info2_titulo": "1. Choose or upload a database",
        "info2": "Right on the initial screen, you will see two options:",
        "info2.1": "Select an existing database: You can choose from well-known databases containing vibration measurements from bearings, such as the CWRU or JNU database.",
        "info2.2": "Upload your own data: If you have your own data (in CSV, Excel, or Parquet format), you can upload it directly into the system.",
        "info3_titulo": "2. Feature extraction",
        "info3": "After loading your vibration signal dataset, the application offers the option to automatically extract features from these signals. This step simplifies the data and reduces the amount of information to be processed, making the model more efficient and interpretable.",
        "info4_titulo": "3. Quantum encoding",
        "info4": "Quantum computers do not process data in the same way as classical computers. For data to be used in a quantum model, it must be converted into a language understandable by qubits. This conversion is called quantum encoding. Each method has its particularities and may influence the model’s performance.",
        "info5_titulo": "4. Parameterized Quantum Circuits (PQC)",
        "info5": "Parameterized Quantum Circuits, also known as PQCs, are the core of Quantum Machine Learning models. They act as the “brain” of the algorithm, being responsible for learning patterns from the input data. The central idea is that the circuit contains quantum gates adjustable by numerical parameters, which are modified during training to find the best configurations.",
        "info5.1": "These parameters are analogous to the “weights” of a classical neural network. During learning, the model adjusts the values of these gates to minimize classification or prediction errors.",
        "info6_titulo": "4.1. Euler Rotations",
        "info6": "An essential part of PQCs are the so-called Euler rotations, which are operations applied to each qubit individually. They change the state of the qubit based on parameterizable angles (for example: Rx(θ), Ry(θ), Rz(θ)), allowing the circuit to represent a wide range of transformations.",
        "info6.1": "You can configure the number of Euler rotations applied in each layer of the circuit. Increasing this number makes the circuit more expressive (capable of representing more complex patterns), but it can also increase execution time and the risk of overfitting.",
        "info7_titulo": "4.2. Qubit Entanglement",
        "info7": "In addition to operations on individual qubits, it is essential for the quantum circuit to connect the qubits with each other. This is done through entanglement gates, which create correlations between the states of the qubits. Entanglement is one of the most powerful properties of quantum computing, as it allows the representation of complex relationships within the data.",
        "info7.1": "You can choose which entangling gate to use in your circuit. The type of entanglement directly affects the model’s architecture and can influence its generalization ability."
    }
}

TEXTOS_INF = {
    "pt": {
        # Página / títulos
        "pagina_inf": "Redes Bayesianas Quânticas",
        #"titulo_app": "Inferência CQBN",
        #"subtitulo_app": "Monte a rede pela interface e rode inferência clássica e quântica.",
        
        "intro_qbn": r"""
        ### Redes Bayesianas
        
        Redes Bayesianas (BNs) são grafos direcionados acíclicos (DAGs) que representam dependências probabilísticas entre variáveis.  
        Na prática, elas permitem **atualizar probabilidades** quando você observa evidências (por exemplo: um sensor “falhou”, um teste “passou”, um componente “degradou”).
        
        ### O que você faz neste módulo
        - **Constrói a BN** (nós, estados, pais e CPTs).
        - **Define evidências** (o que foi observado).
        - **Consulta probabilidades** (posteriores) e compara métodos.
        
        ### Métodos comparados na mesma tela
        - **Exata** (quando viável)
        - **Monte Carlo**
        - **Quântico (shots)**
        - **Quântico + Amplitude Amplification (AA)**
        
        ### Intuição do método quântico (alto nível)
        A ideia é representar a BN como um **circuito quântico**:
        - nós raiz são preparados com rotações **RY**;
        - nós condicionais usam rotações **controladas** (ex.: **CRY**, **CnRY**);
        - quando há muitos controles, pode haver **decomposição** e uso de **ancillas**.
        
        ### Equação essencial (nó binário)
        Para um nó binário $V_i \in \{0,1\}$, o ângulo da rotação pode ser obtido por:
        
        $$
        \theta = 2\arctan\left(\sqrt{\frac{P(V_i=1)}{P(V_i=0)}}\right)
        $$
        
        No caso **condicional**, a mesma forma é aplicada **para cada configuração dos pais** $\Pi_i$.
        
        ### Recursos (qubits): regra rápida
        - Se um nó tem $n_i$ estados, você precisa de  
          $$m_i=\lceil \log_2(n_i)\rceil$$
          qubits para representá-lo.
        - Em BNs **binárias**, uma estimativa simples para o total é:
        
        $$
        m_{BN,2} = s + \max_i |\Pi_{V_i}| - 1
        $$
        
        onde $s$ é o número de nós e o termo extra reflete ancillas usadas em rotações multi-controladas.
        
        **Referência:** Borujeni et al., *Quantum circuit representation of Bayesian networks*, Expert Systems With Applications (2021).
        """,

        "guia_qbn": r"""
        ### Instruções:
        
        1) **Adicione os nós** (à esquerda): informe o nome e a quantidade de estados e clique em **Adicionar nó**.  
        2) **Edite um nó** (à direita): selecione o nó, defina os **pais** e preencha as probabilidades (**marginal** ou **condicionais**).  
        3) **Revise a rede montada**: confira o grafo e as probabilidades inseridas no painel “Probabilidades inseridas”.  
        4) **Defina evidências e consulta**: escolha os nós observados (e seus estados) e os nós de consulta, depois clique em **Rodar inferência**.
        """,
        
        "def_nos_desc": "Crie os nós da rede. Comece pelos nós principais (ex.: Falha, Causa, Sensor).",
        "nome_no_help": "Use um identificador curto e único (ex.: A, B, Pump, Sensor).",
        "card_no_help": "Número de estados do nó (mínimo 2). Ex.: 2 = {s0,s1}.",
        "edicao_no_desc": "Selecione um nó para definir pais e preencher probabilidades (marginal ou CPT).",
        "pais_do_no_help": "Escolha quais nós influenciam este nó.",
        "rede_montada_desc": "Visualize o grafo e revise as probabilidades antes de rodar a inferência.",
        "evidencia_desc": "Evidência = nós observados. Para cada nó escolhido, selecione o estado observado.",
        "query_desc": "Consulta = nós para os quais você quer obter as probabilidades posteriores.",

        # Seções principais
        "def_nos": "Definição dos nós",
        "evidencia": "Evidência e consulta",
        "query_nodes": "Nós de consulta",

        # Visualização da rede (NOVO)
        "rede_montada": "Rede Bayesiana montada",
        "sem_rede": "Adicione nós para visualizar a rede.",
        "probs_inseridas": "Probabilidades inseridas (marginais e CPTs)",

        # Entradas (nós / estados)
        "nome_no": "Nome do nó",
        "card_no": "Quantidade de estados",
        "add_no": "Adicionar nó",
        "remover_no": "Remover nó",
        "selecionar_no": "Selecionar nó",
        "sem_nos": "Nenhum nó definido ainda. Adicione o primeiro nó acima.",
        "nos_evidenciados": "Nós evidenciados",
        "limpar_rede": "Limpar rede",
        "edicao_no": "Edição do nó",
        
        # Estrutura
        "pais_do_no": "Pais do nó",
        "lbl_states": "Estados",
        "lbl_parents": "Pais",

        # Probabilidades
        "probs_raiz": "Probabilidades marginais (nó raiz)",
        "cpt": "Tabela de Probabilidades Condicionais (CPT)",
        "caption_probs": "Preencha as probabilidades. Para nós com pais, cada linha da CPT (combinação de estados dos pais) deve somar 1.",
        "caption_cpt": "Para cada combinação de estados dos pais, informe as probabilidades do nó (a linha deve somar 1).",

        # Execução (na página principal)
        "sidebar_execucao": "Execução",
        "safe_mode": "Safe mode",
        "shots": "Shots (Quântico / Monte Carlo)",
        "seed": "Seed (Monte Carlo)",
        "plots": "Gráficos",
        "topk": "Top-N outcomes (0 = todos)",
        "annotate": "Anotar barras (%)",
        "circuito_ok": "Circuito gerado com sucesso.",
        "exec_desc": "Ajuste como a inferência será executada. Se não tiver certeza, mantenha os valores padrão.",
        "shots_help": "Número de amostras. Mais shots = resultados mais estáveis, porém mais lentos.",
        "seed_help": "Semente do Monte Carlo (reprodutibilidade).",
        "topk_help": "Mostra apenas os Top-N outcomes no gráfico (0 = mostra todos).",
        "annotate_help": "Mostra o valor (%) em cima de cada barra do gráfico.",
        
        # Amplitude Amplification
        "aa": "Amplitude Amplification",
        "aa_enable": "Habilitar AA",
        "aa_k": "k",
        "aa_k_manual": "k (manual)",
        "aa_desc": "Opcional. Use AA para tentar aumentar a chance de medir eventos raros. Em geral, vale a pena quando as probabilidades são pequenas.",
        "aa_enable_help": "Ative para rodar também a versão quântica com Amplitude Amplification (além de shots padrão).",
        "aa_k_manual_help": "Se marcado, você escolhe k. Se desmarcado, o software usa um valor automático.",
        "aa_k_help": "Número de iterações de AA. k maior pode ajudar, mas também pode piorar se passar do ponto.",
        "aa_tip": "Dica: se você não sabe qual k usar, deixe o modo automático.",

        # Ação
        "run": "Rodar inferência",

        # Saídas
        "tabela_resultados": "Tabela de resultados",
        "tabela_a": "Tabela comparativa (Exata vs Monte Carlo vs Quantum)",
        "tabela_b": "Tabela comparativa (Quantum Shots vs Quantum + AA)",
        "graficos": "Gráficos",
        "outcomes_qshots": "Outcomes (Quantum Shots)",
        "outcomes_qaa": "Outcomes (Quantum + AA)",
        "col_node": "Nó",
        "col_state": "Estado",
        "col_exact_pct": "Exata (%)",
        "col_mc_pct": "Monte Carlo (%)",
        "col_mc_ci95": "Monte Carlo — IC95 (%)", 
        "col_qshots_pct": "Quantum Shots (%)",
        "col_qshots_ci95": "Quantum Shots — IC95 (%)",
        "col_qaa_pct": "Quantum + AA (%)",
        "col_qaa_ci95": "Quantum + AA — IC95 (%)",

        # Plot do Circuito
        "ver_circuito": "Circuito Quântico e Número de qubits",
        "circ_indisp": "Circuito indisponível para visualização nas configurações atuais.", 
        "circ_apenas_binaria": "Visualização do circuito disponível apenas para redes binárias (2 estados por nó).", 
        "circ_sem_nos": "Não há nós suficientes para montar o circuito.",
        "circ_muito_grande": "Circuito grande demais para visualização (muitos qubits).",
        "circ_desc_stateprep": "Este circuito mostra a preparação do estado da QBN. No modo padrão, blocos como 'State Preparation' aparecem como operações de alto nível. Ative a decomposição para visualizar as portas elementares (o circuito pode ficar bem maior).",
        "circ_decomp": "Mostrar decomposição (portas elementares)",
        "circ_decomp_level": "Nível de decomposição",
        "circ_force_basis": "Forçar base u/cx",

        # Número de qubits
        "q_header": "Recursos (qubits)", 
        "q_total": "Total de qubits", 
        "q_nodes": "Qubits (nós)", 
        "q_anc": "Qubits (ancillas)", 
        "q_caption": "Contagem de ancillas estimada conforme decomposição CnRY/rotações controladas.",  
        "q_role_node": "Nó", 
        "q_role_ancilla": "Ancilla",
        "q_col_qubit": "Qubit",
        "q_col_role": "Tipo",
        "q_col_node": "Nó", 
        "q_col_slot": "Índice (slot)",

        "node_continuous": "Variável contínua (distribuição)?",
        "node_continuous_help": "Marque se este nó representa uma variável contínua (ex.: Weibull, Normal, Exponencial). O nó será discretizado em pontos (estados) para uso no circuito.",
        "dist_family": "Distribuição",
        "dist_family_help": "Escolha a família da distribuição contínua.",
        "dist_params": "Parâmetros da distribuição",
        "dist_points": "Quantidade de pontos (estados) na discretização",
        "dist_points_help": "Esta quantidade vira o número de estados do nó. Ex.: 8 pontos → 8 estados.",
        "dist_quantile_range": "Faixa de quantis (para cortar caudas)",
        "dist_preview": "Prévia da discretização",
        "dist_root_autofill": "Preencher automaticamente a marginal (apenas nó raiz)",
        "dist_root_autofill_help": "Se o nó não tiver pais, você pode preencher a marginal automaticamente a partir da discretização.",


    },

    "en": {
        # Page / titles
        "pagina_inf": "Quantum Bayesian Networks",
        #"titulo_app": "CQBN Inference",
        #"subtitulo_app": "Build the network and run classical and quantum inference.",
        "intro_qbn_en": r"""
        ### Bayesian Networks
        
        Bayesian Networks (BNs) are directed acyclic graphs (DAGs) that represent probabilistic dependencies among variables.  
        In practice, they let you **update probabilities** when you observe evidence (e.g., a sensor “failed”, a test “passed”, a component “degraded”).
        
        ### What you do in this module
        - **Build the BN** (nodes, states, parents, and CPTs).
        - **Set evidence** (what has been observed).
        - **Query posterior probabilities** and compare methods.
        
        ### Methods compared side by side
        - **Exact** (when feasible)
        - **Monte Carlo**
        - **Quantum (shots)**
        - **Quantum + Amplitude Amplification (AA)** 
        
        ### Quantum approach (high-level intuition)
        The idea is to represent the BN as a **quantum circuit**:
        - root nodes are prepared with **RY** rotations;
        - conditional nodes use **controlled rotations** (e.g., **CRY**, **CnRY**);
        - when there are many controls, the circuit may require **decomposition** and **ancilla qubits**.
        
        ### Key equation (binary node)
        For a binary node $V_i \in \{0,1\}$, the rotation angle can be computed as:
        
        $$
        \theta = 2\arctan\left(\sqrt{\frac{P(V_i=1)}{P(V_i=0)}}\right)
        $$
        
        In the **conditional** case, the same formula is applied **for each parent configuration** $\Pi_i$.
        
        ### Resources (qubits): quick rule
        - If a node has $n_i$ states, you need  
          $$m_i=\lceil \log_2(n_i)\rceil$$
          qubits to represent it.
        - For **binary BNs**, a simple estimate of the total is:
        
        $$
        m_{BN,2} = s + \max_i |\Pi_{V_i}| - 1
        $$
        
        where $s$ is the number of nodes and the extra term reflects ancillas used by multi-controlled rotations.
        
        **Reference:** Borujeni et al., *Quantum circuit representation of Bayesian networks*, Expert Systems With Applications (2021).
        """,
        "guia_qbn": r"""
        ### Instructions:
        
        1) **Add nodes** (left): provide the node name and number of states, then click **Add node**.  
        2) **Edit a node** (right): select the node, define its **parents**, and fill probabilities (**marginal** or **conditional**).  
        3) **Review the assembled network**: check the graph and the “Inserted probabilities” panel.  
        4) **Set evidence and queries**: select observed nodes (and their states) and query nodes, then click **Run inference**.
        """,

        "def_nos_desc": "Create the network nodes. Start with the main variables (e.g., Failure, Cause, Sensor).",
        "nome_no_help": "Use a short, unique identifier (e.g., A, B, Pump, Sensor).",
        "card_no_help": "Number of states (min 2). Example: 2 = {s0,s1}.",
        "edicao_no_desc": "Select a node to define parents and fill probabilities (marginal or CPT).",
        "pais_do_no_help": "Select which nodes influence this node (incoming edges).",
        "rede_montada_desc": "Inspect the graph and probabilities before running inference.",
        "evidencia_desc": "Evidence = observed nodes. For each selected node, choose the observed state.",
        "query_desc": "Queries = nodes you want posterior probabilities for.",

        
        # Main sections
        "def_nos": "Node definition",
        "evidencia": "Evidence and query",
        "query_nodes": "Query nodes",

        # Network visualization (NEW)
        "rede_montada": "Built Bayesian network",
        "sem_rede": "Add nodes to visualize the network.",
        "probs_inseridas": "Entered probabilities (marginals and CPTs)",

        # Inputs (nodes / states)
        "nome_no": "Node name",
        "card_no": "Number of states",
        "add_no": "Add node",
        "remover_no": "Remove node",
        "selecionar_no": "Select node",
        "sem_nos": "No nodes defined yet. Add your first node above.",
        "nos_evidenciados": "Evidence nodes",
        "limpar_rede": "Clear network",
        "edicao_no": "Node editing",
        
        # Structure
        "pais_do_no": "Parents of node",
        "lbl_states": "States",
        "lbl_parents": "Parents",

        # Probabilities
        "probs_raiz": "Marginal probabilities (root node)",
        "cpt": "Conditional Probability Table (CPT)",
        "caption_probs": "Fill in the probabilities. For nodes with parents, each CPT row (a parent-state combination) must sum to 1.",
        "caption_cpt": "For each parent-state combination, enter the node probabilities (the row must sum to 1).",

        # Execution (on main page)
        "sidebar_execucao": "Execution",
        "safe_mode": "Safe mode",
        "shots": "Shots (Quantum / Monte Carlo)",
        "seed": "Seed (Monte Carlo)",
        "plots": "Plots",
        "topk": "Top-N outcomes (0 = all)",
        "annotate": "Annotate bars (%)",
        "circuito_ok": "Circuit generated successfully.",
        "exec_desc": "Controls how inference is executed. If unsure, keep the default values.",
        "shots_help": "Number of samples. More shots = more stable results, but slower.",
        "seed_help": "Monte Carlo random seed (reproducibility).",
        "topk_help": "Show only the Top-N outcomes in plots (0 = show all).",
        "annotate_help": "Display the percentage value on top of each bar.",


        # Amplitude Amplification
        "aa": "Amplitude Amplification",
        "aa_enable": "Enable AA",
        "aa_k": "k",
        "aa_k_manual": "k (manual)",
        "aa_desc": "Optional. Use AA to increase the chance of measuring rare events. It is most useful when probabilities are small.",
        "aa_enable_help": "Enable to also run the quantum version with Amplitude Amplification (in addition to standard shots).",
        "aa_k_manual_help": "If checked, you choose k. If unchecked, the app uses an automatic value.",
        "aa_k_help": "Number of AA iterations. Larger k may help, but can also hurt if it overshoots.",
        "aa_tip": "Tip: if you are unsure about k, keep automatic mode.",

        # Action
        "run": "Run inference",

        # Outputs
        "tabela_resultados": "Results table",
        "tabela_a": "Comparison table (Exact vs Monte Carlo vs Quantum)",
        "tabela_b": "Comparison table (Quantum Shots vs Quantum + AA)",
        "graficos": "Charts",
        "outcomes_qshots": "Outcomes (Quantum Shots)",
        "outcomes_qaa": "Outcomes (Quantum + AA)",
        "col_node": "Node",
        "col_state": "State",
        "col_exact_pct": "Exact (%)",
        "col_mc_pct": "Monte Carlo (%)",
        "col_mc_ci95": "Monte Carlo — CI95 (%)",
        "col_qshots_pct": "Quantum Shots (%)",
        "col_qshots_ci95": "Quantum Shots — CI95 (%)",
        "col_qaa_pct": "Quantum + AA (%)",
        "col_qaa_ci95": "Quantum + AA — CI95 (%)",
        

        # Circuit Plot
        "ver_circuito": "Quantum Circuit and Number of Qubits",
        "circ_indisp": "Circuit is not available for display under the current settings.",
        "circ_apenas_binaria": "Circuit display is available only for binary networks (2 states per node).",
        "circ_sem_nos": "Not enough nodes to build the circuit.",
        "circ_muito_grande": "Circuit is too large to display (too many qubits).",
        "circ_desc_stateprep": "This circuit shows the QBN state preparation. In the default view, blocks like 'State Preparation' appear as high-level operations. Enable decomposition to see elementary gates (the circuit may become much larger).",
        "circ_decomp": "Show decomposition (elementary gates)",
        "circ_decomp_level": "Decomposition level",
        "circ_force_basis": "Force u/cx basis",

        # Number of qubits
        "q_header": "Resources (qubits)",
        "q_total": "Total qubits",
        "q_nodes": "Qubits (nodes)",
        "q_anc": "Qubits (ancillas)",
        "q_caption": "Ancilla count estimated under CnRY / controlled-rotation decomposition (Borujeni et al., 2021 style).", 
        "q_role_node": "Node",
        "q_role_ancilla": "Ancilla",
        "q_col_qubit": "Qubit",
        "q_col_role": "Type",
        "q_col_node": "Node",
        "q_col_slot": "Index (slot)",

        "node_continuous": "Continuous variable (distribution)?",
        "node_continuous_help": "Enable if this node represents a continuous variable (e.g., Weibull, Normal, Exponential). The node will be discretized into points (states) for circuit use.",
        "dist_family": "Distribution",
        "dist_family_help": "Choose the continuous distribution family.",
        "dist_params": "Distribution parameters",
        "dist_points": "Number of discretization points (states)",
        "dist_points_help": "This becomes the number of node states. E.g., 8 points → 8 states.",
        "dist_quantile_range": "Quantile range (to trim tails)",
        "dist_preview": "Discretization preview",
        "dist_root_autofill": "Auto-fill marginal (root node only)",
        "dist_root_autofill_help": "If the node has no parents, you can auto-fill its marginal from the discretization.",



    },
}

def mostrar_rodape_logos2(textos):
    st.markdown("<div style='margin-top:40px'></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"**{textos['apoio']}**")

    logos = [
        "1.png", "2.png", "3.png", "4.png", "5.png",
        "6.png", "7.png", "8.png", "9.png", "10.png", "11.png",
    ]

    cols = st.columns(11)  
    for i, path in enumerate(logos):
        with cols[i]:
            st.image(path, width=80)  

    st.markdown(f"<div style='opacity:.75;font-size:13px;margin-top:10px'>{textos['apo']}</div>",
                unsafe_allow_html=True)


def aplicar_css_botoes():
    st.markdown(
        """
        <style>
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
            st.rerun()

    with col3:
        st.image("ml3.png", width=150)
        if st.button(textos["pagina_ml"], key="ml_btn"):
            st.session_state['pagina'] = 'ml'
            st.rerun()

    with st.expander(textos["inf_ref"], expanded=False):

        st.markdown("""
            <style>
            div[data-testid="stExpander"] div.stButton {
                display: inline-block;
                margin-right: 6px; /* aproxima os botões */
            }
            div[data-testid="stExpander"] button {
                background-color: white !important;
                color: #333333 !important;
                font-size: 13px !important;
                padding: 4px 10px !important;
                border-radius: 6px !important;
                border: 1px solid #cccccc !important;
                transition: 0.2s ease-in-out;
            }
            div[data-testid="stExpander"] button:hover {
                background-color: #f2f2f2 !important;
                border-color: #999999 !important;
            }
            </style>
        """, unsafe_allow_html=True)

        col1, col2, _, _, _ = st.columns([1, 1, 1, 1, 1])
        with col1:
            if st.button(textos["pagina_info"], key="btn_info"):
                st.session_state['pagina'] = 'info'
                st.rerun()
        with col2:
            if st.button(textos["pagina_referencias"], key="btn_ref"):
                st.session_state['pagina'] = 'ref'
                st.rerun()
    with col4:
        st.image("infer3.png", width=150)
        if st.button(textos["pagina_inferencia"], key="inferencia_btn"):
            st.session_state['pagina'] = 'inferencia'
            st.rerun()

    with col5:
        st.write("")

def mostrar_cartoes_de_info(textos):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.write("")
    with col2:
        if st.button(textos["pagina_otimizacao"], key="otimizacao_btn"):
            st.session_state['pagina'] = 'explicacao_otimizacao'
            st.rerun()
    with col3:
        if st.button(textos["pagina_ml"], key="ml_btn"):
            st.session_state['pagina'] = 'ml_info'
            st.title(textos_ml["info1_titulo"])
            st.header(textos_ml["info1"])
            st.rerun()
            
    with col4:
        if st.button(textos["pagina_inferencia"], key="inferencia_btn"):
            st.session_state['pagina'] = 'inferencia_info'
            st.rerun()
    with col5:
        st.write("")

def mostrar_ref(textos):
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.write("")

    with col2:
        if st.button(textos["pagina_otimizacao"], key="ref_otimizacao_btn"):
            st.session_state['pagina'] = 'otim_ref'
            st.rerun()

    with col3:
        if st.button(textos["pagina_ml"], key="ref_ml_btn"):
            st.session_state['pagina'] = 'ml_ref'
            st.rerun()

    with col4:
        if st.button(textos["pagina_inferencia"], key="ref_inferencia_btn"):
            st.session_state['pagina'] = 'inf_ref'
            st.rerun()

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
        
def mostrar_logos_parceiros():
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
            st.write("")

    with col2:
            st.write("")

    with col3:
        st.markdown(
            """
            <a href="https://www.ufpe.br/web/prh38.1" target="_blank">
                <img src="https://raw.githubusercontent.com/VMilena30/QApp/main/prh.png" width="180" style="border-radius:10px;"/>
            </a>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            """
            <a href="https://ceerma.org/" target="_blank">
                <img src="https://raw.githubusercontent.com/VMilena30/QApp/main/cer.png" width="180" style="border-radius:10px;"/>
            </a>
            """,
            unsafe_allow_html=True
        )
    
    with col5:
            st.write("")

    with col6:
            st.write("")
        
#Otimização

def ler_manualmente(textos_otim):
    st.write(textos_otim["insira_dados"])

    col1, col2 = st.columns(2)
    with col1:
        s = st.number_input(f"{textos_otim['s']}:", step=1, min_value=1, max_value=50)
        nj_min = st.number_input(f"{textos_otim['nj_min']}:", step=1, min_value=0)
    with col2:
        nj_max = st.number_input(f"{textos_otim['nj_max']}:", step=1, min_value=1)
        ctj_of = st.number_input(f"{textos_otim['ctj_of']}:", step=1, min_value=1, max_value=100)

    if nj_min > nj_max:
        st.error("O valor mínimo de componentes não pode ser maior que o valor máximo.")
        st.stop()

    st.markdown(f"**{textos_otim['lista_componentes']}**")

    Rjk_of, cjk_of = [], []
    for i in range(int(ctj_of)):
        col_r, col_c = st.columns(2)
        with col_r:
            Rjk_of.append(
                st.number_input(f"{textos_otim['confiabilidade']} [{i+1}]:",
                                key=f'Rjk_of_{i}',
                                step=0.001,
                                min_value=0.0,
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

    C_of = st.number_input(f"{textos_otim['custo_total_limite']}:", step=1, min_value=1)
    if C_of < min(cjk_of):
        st.warning("O limite de custo total é menor que o menor custo de componente. Ajuste os valores.")

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
            st.markdown(f"{textos_otim['descricao_rap']}")
            st.markdown(f"#### {textos_otim['algoritmos']}")
            st.markdown(f"{textos_otim['inicializacoes_titulo']}")
            
def mostrar_ml(textos_ml):
    with st.sidebar.expander(textos_ml['pagina_ml']):
        st.markdown(f"#### {textos_ml['metodos']}")
        st.markdown(f"**Angle Encoding:** {textos_ml['desc_angle']}")
        st.markdown(f"**Amplitude Encoding:** {textos_ml['desc_ampli']}")
        st.markdown(f"**Pauli Feature Map:** {textos_ml['desc_pauli']}")

def mostrar_ml_info(textos, textos_ml):
    st.header(textos_ml["pagina_ml"])
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
        
def mostrar_inf(textos):
    with st.sidebar.expander(textos["pagina_inferencia"]):
        st.markdown(f"#### {textos['inf1']}")
        st.markdown(f"{textos['inf2']}")
        st.markdown(f"#### {textos['inf3']}")
        st.markdown(f"{textos['inf4']}")


def main():
    import streamlit as st
    import os

    aplicar_css_botoes()

    # estados
    if "step" not in st.session_state:
        st.session_state.step = "login"   # login -> verify -> app
    if "otp_verified" not in st.session_state:
        st.session_state.otp_verified = False
    if "lang" not in st.session_state or st.session_state.lang is None:
        st.session_state.lang = "en"

    if "pending_user" not in st.session_state:
        st.session_state.pending_user = None
    if "otp_code" not in st.session_state:
        st.session_state.otp_code = None
    if "otp_email" not in st.session_state:
        st.session_state.otp_email = None

    valid_steps = {"login", "verify", "app"}
    if st.session_state.step not in valid_steps:
        st.session_state.step = "login"

    # não deixa entrar no app sem OTP
    if st.session_state.step == "app" and not st.session_state.otp_verified:
        st.session_state.step = "login"

    if st.session_state.step == "login":
        lang = st.session_state.lang
        t = TEXTOS_LOGIN[lang]

        left, right = st.columns([1.4, 1.0], gap="large")

        with left:
            from pathlib import Path
        
            BASE_DIR = Path(__file__).resolve().parent
            LOGO = BASE_DIR / "qplog.png"   # ajuste se estiver em assets/
            
            st.markdown(
                """
                <div style="
                    width:100%;
                    display:flex;
                    flex-direction:column;
                    align-items:center;
                    justify-content:center;
                    text-align:center;
                ">
                """,
                unsafe_allow_html=True
            )
            
            st.markdown(
                f"""
                <img src="data:image/png;base64,{LOGO.read_bytes().hex()}"
                     style="width:320px; display:block; margin:0 auto;" />
                """,
                unsafe_allow_html=True
            )

            
            # textos
            st.markdown("""
            <div style="font-size:42px; font-weight:800; margin-top:12px;">
                qPrism
            </div>
            <div style="font-size:16px; margin-top:8px; max-width:720px;">
                Quantum Platform for Reliability: Inference, Systems modeling, and Machine learning
            </div>
            </div>
            """, unsafe_allow_html=True)

        
            st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        
            c1, c2, c3, c4, c5 = st.columns([1, 0.6, 0.6, 0.6, 1])
            
            with c2:
                st.image("1.png", width=80)
            with c3:
                st.image("2.png", width=90)
            with c4:
                st.image("4.png", width=90)

        
            st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        

        with right:
            st.markdown("""
            <style>
            /* Remove espaço exagerado do radio */
            div[role="radiogroup"] {
                margin-top: -12px !important;
                margin-bottom: -14px !important;
                padding-top: 4 !important;
                padding-bottom: 4 !important;
                justify-content: center;
            }
            
            .element-container:has(div[role="radiogroup"]) {
                margin-top: 0 !important;
                margin-bottom: 0 !important;
            }
            </style>
            """, unsafe_allow_html=True)

            st.markdown("""
            <style>
            /* Remove espaço que o Streamlit adiciona a CADA markdown */
            div[data-testid="stMarkdownContainer"] {
                margin: 0 !important;
                padding: 0 !important;
            }
            
            /* Remove o wrapper invisível que cria o "buraco" */
            .element-container {
                margin: 0 !important;
                padding: 0 !important;
            }
            
            /* Remove gap entre elementos empilhados */
            div[data-testid="stVerticalBlock"] {
                gap: 0 !important;
            }
            </style>
            """, unsafe_allow_html=True)


            st.markdown(
                """
                <div style="
                    text-align:center;
                    font-size:22px;
                    font-weight:700;
                    border:0x solid #333;
                    padding:0px 5px;
                    background: rgba(255,255,255,0.75);
                    border-radius:0px;
                    margin-bottom:0px;">
                    Welcome to qPrism / Boas Vindas ao qPrism
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown(
                """
                <div style="
                    text-align:center;
                    font-size:18px;
                    font-weight:400;
                    background: rgba(255,255,255,0.75);
                    padding:0px 5px;
                    border-radius:0px;
                    margin-bottom:0px;
                ">
                    Select a language / Selecione um idioma
                </div>
                """,
                unsafe_allow_html=True
            )
            
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                escolha = st.radio(
                    "",
                    ["English", "Português (Brasil)"],
                    index=0 if st.session_state.lang == "en" else 1,
                    horizontal=True,
                    key="lang_choice_login"
                )
            
            new_lang = "pt" if "Português" in escolha else "en"
            if new_lang != st.session_state.lang:
                st.session_state.lang = new_lang
                st.rerun()

            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size:22px;font-weight:700;line-height:1.05;margin-bottom:4px;">
              {t["access_title"]}
            </div>
            <div style="font-size:16px;margin-bottom:8px;">
              {t["access_sub"]}
            </div>
            """, unsafe_allow_html=True)
        
            with st.form("login_form", clear_on_submit=False):
                name = st.text_input(t["name_opt"])
                email = st.text_input(t["email_req"])
                country = st.text_input(t["country_req"])
                company = st.text_input(t["company_req"])
                role = st.text_input(t["role_opt"])
                st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
                submitted = st.form_submit_button(t["continue"])

            if submitted:
                name_clean = (name or "").strip()
                email_clean = (email or "").strip().lower()
                country_clean = (country or "").strip()
                company_clean = (company or "").strip()
                role_clean = (role or "").strip()

                if not email_clean or not company_clean or not country_clean:
                    st.error(t["err_required"])
                elif not is_valid_email(email_clean):
                    st.error(t["err_email"])
                else:
                    otp = generate_otp()
                    try:
                        send_otp_email(email_clean, otp)
                    except Exception:
                        st.error(t["err_send"])
                        st.stop()

                    st.session_state.pending_user = {
                        "name": name_clean,
                        "email": email_clean,
                        "country": country_clean,
                        "company": company_clean,
                        "role": role_clean,
                        "lang": st.session_state.lang,
                    }
                    st.session_state.otp_code = otp
                    st.session_state.otp_email = email_clean
                    st.session_state.step = "verify"
                    st.rerun()

        st.stop()

    # ---------- VERIFY ----------
    if st.session_state.step == "verify":
        lang = st.session_state.lang or "en"
        t = TEXTOS_LOGIN[lang]

        left, mid, right = st.columns([1.2, 1.0, 1.2])
        with mid:
            st.markdown(f"### {t['verify_title']}")
            st.markdown(f"{t['verify_sub']} **{st.session_state.otp_email}**.")

            code_input = st.text_input(
                t["code_label"],
                max_chars=6,
                placeholder=t["code_ph"]
            )

            c1, c2 = st.columns(2)
            verify_btn = c1.button(t["verify_btn"])
            resend_btn = c2.button(t["resend_btn"])

            if verify_btn:
                if (code_input or "").strip() == (st.session_state.otp_code or ""):
                    from datetime import datetime
                    user = st.session_state.pending_user or {}
                    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    save_registration(
                        user.get("name", ""),
                        user.get("email", ""),
                        user.get("company", ""),
                        user.get("role", ""),
                        created_at
                    )
                    append_csv_log(
                        user.get("name", ""),
                        user.get("email", ""),
                        user.get("company", ""),
                        user.get("role", ""),
                        created_at
                    )

                    st.session_state.user = {**user, "created_at": created_at}
                    st.session_state.pending_user = None
                    st.session_state.otp_verified = True
                    st.session_state.step = "app"
                    st.rerun()
                else:
                    st.error(t["err_code"])

            if resend_btn:
                otp = generate_otp()
                try:
                    send_otp_email(st.session_state.otp_email, otp)
                except Exception:
                    st.error(t["err_send"])
                    st.stop()

                st.session_state.otp_code = otp
                st.success(t["resend_ok"])

        st.stop()

        # ---------- APP ----------
    if st.session_state.step != "app" or not st.session_state.otp_verified:
        st.stop()

    # sidebar idioma (com rerun)
    idioma_atual = "Português" if st.session_state.lang == "pt" else "English"
    idioma_selecionado = st.sidebar.selectbox(
        "Language / Idioma:",
        ("🇺🇸 English (US)", "🇧🇷 Português (BR)"),
        index=0 if idioma_atual == "English" else 1
    )

    changed = False
    if idioma_selecionado == "🇧🇷 Português (BR)" and st.session_state.lang != "pt":
        st.session_state.lang = "pt"
        changed = True
    elif idioma_selecionado == "🇺🇸 English (US)" and st.session_state.lang != "en":
        st.session_state.lang = "en"
        changed = True

    if changed:
        st.rerun()

    lang = st.session_state.lang
    textos = TEXTOS[lang]
    textos_otim = TEXTOS_OPT[lang]
    textos_ml = TEXTOS_ML[lang]
    textos_inf = TEXTOS_INF[lang]



    mostrar_otim(textos_otim)
    mostrar_ml(textos_ml)
    mostrar_inf(textos)

    if 'pagina' not in st.session_state:
        st.session_state['pagina'] = 'inicio'
    
    if st.session_state['pagina'] == 'inicio':
        mostrar_introducao_e_titulo(textos)
        mostrar_cartoes_de_area(textos)
        
        mostrar_rodape_logos2(textos)
        
    elif st.session_state['pagina'] == 'otimizacao':
        st.markdown(textos_otim["rap_descricao"])
        st.divider()
        
        col1, col2 = st.columns([9, 2])
        
        with col1:
            st.subheader("Aplicação")
        with col2:
            ajuda = st.button("?", key="botao_ajuda")
        
        st.markdown("""
            <style>
            /* Estilo apenas para o botão de ajuda */
            div[data-testid="stButton"] > button[kind="secondary"][aria-label="?"],
            div[data-testid="stButton"]:has(button[data-testid="baseButton-secondary"]) button {
                background-color: white !important;
                border: 1.5px solid #03518C !important;
                border-radius: 50% !important;
                width: 26px !important;
                height: 26px !important;
                font-size: 14px !important;
                font-weight: bold !important;
                color: #03518C !important;
                padding: 0 !important;
                margin-top: 2px !important;
                cursor: pointer !important;
            }
        
            /* Efeito hover */
            div[data-testid="stButton"]:has(button[data-testid="baseButton-secondary"]) button:hover {
                background-color: #f5f9ff !important;
                color: #02416B !important;
                border-color: #02416B !important;
            }
            </style>
        """, unsafe_allow_html=True)
            
        if ajuda:
            st.session_state["pagina"] = "explicacao_otimizacao"
            st.rerun()
    
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
       
        modo_leitura = st.radio(
            textos_otim["modo_leitura_label"],
            (textos_otim["modo_leitura_manual"], textos_otim["modo_leitura_upload"]),
            key=f"modo_leitura_{lang}", help= textos_otim["help1"]
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
                    modo_algoritmo = st.radio(textos_otim["selecionar_algoritmo"], ('QAOA', 'VQE'), help=textos_otim["help2"])
        
                    if modo_algoritmo == 'VQE':
                        tipo_circuito = st.radio(
                            textos_otim["selecionar_tipo_circuito"], 
                            (textos_otim["real_amplitudes"], textos_otim["two_local"]),  help=textos_otim["help3"]
                        )
        
                        if tipo_circuito == textos_otim["two_local"]:
                            col_rot, col_ent = st.columns(2)
                            with col_rot:
                                rotacao_escolhida = st.multiselect(
                                    textos_otim["selecionar_rotacao"],
                                    textos_otim["opcoes_rotacao"],  help=textos_otim["help4"]
                                )
                            with col_ent:
                                entanglement_escolhido = st.multiselect(
                                    textos_otim["selecionar_emaranhamento"],
                                    textos_otim["opcoes_emaranhamento"],  help=textos_otim["help5"]
                                )
        
                        tipo_inicializacao = st.radio(
                            textos_otim["tipo_inicializacao"],
                            textos_otim["tipos_inicializacao_vqe"],  help=textos_otim["help6"]
                        )
        
                        if tipo_inicializacao in ['Ponto Fixo', 'Fixed Point']:
                            numero_ponto_fixo = st.number_input(
                                textos_otim["inserir_ponto_fixo"], step=0.1, help=textos_otim["help6"]
                            )
        
                    elif modo_algoritmo == 'QAOA':
                        tipo_inicializacao = st.radio(
                            textos_otim["tipo_inicializacao"],
                            textos_otim["tipos_inicializacao_qaoa"],  help=textos_otim["help6"]
                        )
        
                        if tipo_inicializacao in ['Ponto Fixo', 'Fixed Point']:
                            numero_ponto_fixo = st.number_input(
                                textos_otim["inserir_ponto_fixo"], step=0.1,  help=textos_otim["help6"]
                            )
        
                with col_param:
                    otimizador = st.radio(
                        textos_otim["selecionar_otimizador"],
                        textos_otim["opcoes_otimizadores"], help=textos_otim["help5"]
                    )
                    camadas = st.number_input(
                        textos_otim["inserir_camadas"], min_value=1, max_value=3, value=1, help=textos_otim["help5"]
                    )
                    rodadas = st.number_input(
                        textos_otim["inserir_rodadas"], min_value=1, value=1, help=textos_otim["help5"]
                    )
                    shots = st.number_input(
                        textos_otim["inserir_shots"], min_value=100, value=1000, help=textos_otim["help5"]
                    )
                
        if st.button(textos_otim['executar']):

            # Verifica o modo leitura escolhido (upload/manual)
            if modo_leitura == textos_otim['modo_leitura_upload']:
                instancia = dados[0]  # Dados do upload
            else:
                instancia = dados    
        
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
                st.rerun()
                    

    elif st.session_state['pagina'] == 'explicacao_otimizacao':
        st.title(textos_otim["info1_titulo"])
        st.write(textos_otim["info1"])
        
        st.header(textos_otim["info2_titulo"])
        st.write(textos_otim["info2"])
        st.write(textos_otim["info21"])
        
        st.header(textos_otim["info3_titulo"])
        st.write(textos_otim["info3"])
        
        st.header(textos_otim["info4_titulo"])
        st.write(textos_otim["info4"])
        
        if st.button("Aplicação"):
            st.session_state['pagina'] = 'otimizacao'

    elif st.session_state['pagina'] == 'ml':
        import pandas as pd
        import numpy as np
        import streamlit as st
        import os
        import scipy
        from scipy.stats import kurtosis, skew
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score
    
        import pennylane as qml
        from pennylane import numpy as pnp
    
        textos_ml = TEXTOS_ML[st.session_state.lang]
        textos = TEXTOS[st.session_state.lang]
    
        st.subheader(textos["pagina_ml2"])
    
        # ========= 1) PERGUNTA INICIAL: usar base do app ou subir arquivo =========
        modo_dataset = st.radio(
            "Como deseja fornecer os dados?",
            ("Usar base de vibração do app", "Enviar minha própria base"),
            help=textos_ml["help_1"]
        )
    
        df = None
        y = None
        nome_base = None
    
        # --------- OPÇÃO 1: usar base do app (CWRU / JNU) ----------
        if modo_dataset == "Usar base de vibração do app":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(textos_ml["dataset_opcao"])
                dataset_opcao = st.selectbox(
                    textos_ml["selecione_base"],
                    [" - ", "CWRU", "JNU"]
                )
            with col2:
                st.info("Selecione uma das bases internas. Para usar seus próprios dados, escolha a outra opção acima.")
    
            def carregar_dados_brutos(nome):
                if nome == "CWRU":
                    # mesmo esquema que você tinha
                    df_raw = pd.DataFrame(columns=['DE_data', 'fault'])
    
                    for root, dirs, files in os.walk(r"C:\\Arthur\\load_12K", topdown=False):
                        for file_name in files:
                            path = os.path.join(root, file_name)
                            mat = scipy.io.loadmat(path)
                            key_name = list(mat.keys())[3]
                            DE_data = mat.get(key_name)
                            fault = np.full((len(DE_data), 1), file_name[:-4])
                            df_temp = pd.DataFrame({'DE_data': np.ravel(DE_data), 'fault': np.ravel(fault)})
                            df_raw = pd.concat([df_raw, df_temp], axis=0)
    
                    # janela
                    win_len = 1000
                    stride = 900
                    x = []
                    y = []
                    for k in df_raw['fault'].unique():
                        df_temp_2 = df_raw[df_raw['fault'] == k]
                        for i in np.arange(0, len(df_temp_2) - (win_len), stride):
                            temp = df_temp_2.iloc[i:i+win_len, :-1].values
                            temp = temp.reshape((1, -1))
                            x.append(temp)
                            y.append(df_temp_2.iloc[i+win_len, -1])
    
                    x = np.array(x).reshape((-1, win_len))
                    y = np.array(y)
                    return x, y
    
                elif nome == "JNU":
                    dataset = np.load(r"C:\\Arthur\\JNU_quantum_8.npz")
                    X = dataset['data']
                    y = dataset['label']
                    return X, y
                else:
                    return None, None
    
            if dataset_opcao != " - ":
                X_raw, y = carregar_dados_brutos(dataset_opcao)
                nome_base = dataset_opcao
                st.success(textos_ml["upload_sucesso"])
                st.write("Formato dos dados carregados:", X_raw.shape)
            else:
                X_raw, y = None, None
    
        # --------- OPÇÃO 2: upload de base própria ----------
        else:
            uploaded_file = st.file_uploader(
                textos_ml["upload_label"],
                type=["csv", "xlsx", "parquet"],
                help=textos_ml["help_2"]
            )
            if uploaded_file is not None:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(".xlsx"):
                    df = pd.read_excel(uploaded_file)
                else:
                    df = pd.read_parquet(uploaded_file)
    
                st.success(textos_ml["upload_sucesso"])
                st.dataframe(df.head())
                nome_base = uploaded_file.name
    
                # aqui supomos que tem uma coluna 'label' ou 'target'
                possible_labels = [c for c in df.columns if c.lower() in ["label", "target", "class"]]
                if len(possible_labels) == 0:
                    st.warning("Não encontrei coluna de rótulo. Vou assumir que a última coluna é o alvo.")
                    y = df.iloc[:, -1].values
                    X_raw = df.iloc[:, :-1].values
                else:
                    label_col = possible_labels[0]
                    y = df[label_col].values
                    X_raw = df.drop(columns=[label_col]).values
            else:
                X_raw, y = None, None
    
        st.divider()
    
        # ========= 2) FEATURES (agora OPCIONAL) =========
    
        st.markdown(textos_ml["selecione_features"])
        features_disponiveis = [
            "Média", "Variância", "Desvio-padrão", "RMS", "Kurtosis",
            "Peak to peak", "Max Amplitude", "Min Amplitude", "Skewness",
            "CrestFactor", "Mediana", "Energia", "Entropia"
        ]
        selected_features = st.multiselect(
            textos_ml["label_features"],
            options=features_disponiveis,
            help="Se não selecionar nada, vou extrair TODAS automaticamente."
        )
    
        def extrair_features_amostra(amostra):
            feats = {}
            feats["Média"] = np.mean(amostra)
            feats["Variância"] = np.var(amostra)
            feats["Desvio-padrão"] = np.std(amostra)
            feats["RMS"] = np.sqrt(np.mean(np.square(amostra)))
            feats["Kurtosis"] = kurtosis(amostra)
            feats["Peak to peak"] = np.ptp(amostra)
            feats["Max Amplitude"] = np.max(amostra)
            feats["Min Amplitude"] = np.min(amostra)
            feats["Skewness"] = skew(amostra)
            feats["CrestFactor"] = np.max(np.abs(amostra)) / (np.sqrt(np.mean(np.square(amostra))) + 1e-10)
            feats["Mediana"] = np.median(amostra)
            feats["Energia"] = np.sum(amostra ** 2)
            prob, _ = np.histogram(amostra, bins=30, density=True)
            prob = prob[prob > 0]
            feats["Entropia"] = -np.sum(prob * np.log(prob))
            return feats
    
        def extrair_features_dataset(dataset_bruto, selected_features):
            lista = []
            for amostra in dataset_bruto:
                f = extrair_features_amostra(amostra)
                if selected_features:
                    f = {k: f[k] for k in selected_features}
                lista.append(f)
            return pd.DataFrame(lista)
    
        st.divider()
    
        # ========= 3) ENCODING + PQC =========
    
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(textos_ml["encoding_title"])
            encoding_method = st.selectbox(
                textos_ml["encoding_label"],
                [" - ", "Angle encoding", "Amplitude encoding",
                 "ZFeaturemap", "XFeaturemap", "YFeaturemap", "ZZFeaturemap"]
            )
    
            # rotações de Euler (podem ser 1, 2 ou 3)
            st.markdown(textos_ml["euler_title"])
            rot = st.selectbox(
                textos_ml["euler_label"],
                [" - ", "1", "2", "3"],
                help="Pode usar 1 eixo só (rotação simples) ou 2/3 eixos (Euler)."
            )
    
            eixos = []
            if rot != " - ":
                rot_n = int(rot)
                for i in range(rot_n):
                    eixo_i = st.selectbox(
                        textos_ml["euler_eixo_n"].format(n=i+1),
                        ["X", "Y", "Z"],
                        key=f"eixo_{i}"
                    )
                    eixos.append(eixo_i)
    
        with col2:
            # aqui vem a correção da professora: QCNN não é porta
            tipo_circuito = st.selectbox(
                "Selecione o tipo de circuito / arquitetura quântica:",
                [" - ", "Camada parametrizada", "Real Amplitudes", "QCNN (experimental)"]
            )
    
            porta_emaranhamento = None
            if tipo_circuito == "Camada parametrizada":
                porta_emaranhamento = st.selectbox(
                    textos_ml["entanglement_title"],
                    [" - ", "CZ", "iSWAP"]
                )
    
            paciencia = st.number_input(textos_ml["paciencia"], min_value=0, max_value=400, value=0, step=1)
            epocas = st.number_input(textos_ml["epocas"], min_value=1, max_value=500, value=1, step=1)
    
        st.divider()
    
        # ========= 4) EXECUTAR =========
    
        def criar_circuito(encoding_method, eixos, tipo_circuito, porta_emaranhamento, n_qubits):
            dev = qml.device("default.qubit", wires=n_qubits)
    
            @qml.qnode(dev)
            def circuit(x, weights):
                # 1) encoding
                if encoding_method == "Angle encoding":
                    for i in range(n_qubits):
                        for eixo in eixos:
                            if eixo == "X":
                                qml.RX(x[i], wires=i)
                            elif eixo == "Y":
                                qml.RY(x[i], wires=i)
                            elif eixo == "Z":
                                qml.RZ(x[i], wires=i)
                elif encoding_method == "Amplitude encoding":
                    qml.AmplitudeEmbedding(features=x, wires=range(n_qubits), normalize=True)
                elif encoding_method == "ZFeaturemap":
                    for i in range(n_qubits):
                        qml.RZ(x[i], wires=i)
                elif encoding_method == "XFeaturemap":
                    for i in range(n_qubits):
                        qml.RX(x[i], wires=i)
                elif encoding_method == "YFeaturemap":
                    for i in range(n_qubits):
                        qml.RY(x[i], wires=i)
                elif encoding_method == "ZZFeaturemap":
                    for i in range(n_qubits):
                        qml.RZ(x[i], wires=i)
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i+1])
    
                # 2) parte variacional
                if tipo_circuito == "Real Amplitudes":
                    qml.templates.layers.RealAmplitudes(weights, wires=range(n_qubits))
                elif tipo_circuito == "QCNN (experimental)":
                    # placeholder
                    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                else:  # Camada parametrizada
                    for i in range(n_qubits):
                        qml.RX(weights[i, 0], wires=i)
                        qml.RY(weights[i, 1], wires=i)
                        qml.RZ(weights[i, 2], wires=i)
    
                    # emaranhamento só se foi selecionado
                    if porta_emaranhamento == "CZ":
                        for i in range(n_qubits - 1):
                            qml.CZ(wires=[i, i+1])
                    elif porta_emaranhamento == "iSWAP":
                        for i in range(n_qubits - 1):
                            qml.ISWAP(wires=[i, i+1])
    
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
            return circuit
    
        def proxima_potencia_de_2(n):
            """retorna o menor 2^k >= n"""
            k = 1
            while 2**k < n:
                k += 1
            return 2**k
    
        if st.button(textos_ml["exec_2"]):
            # ===== VALIDAÇÕES =====
            if (modo_dataset == "Usar base de vibração do app" and (X_raw is None or y is None)) or \
               (modo_dataset == "Enviar minha própria base" and (X_raw is None or y is None)):
                st.error("Dados não carregados. Selecione uma base ou envie seu arquivo.")
            elif encoding_method == " - ":
                st.error("Por favor, selecione um método de codificação quântica.")
            elif tipo_circuito == "Camada parametrizada" and (porta_emaranhamento is None or porta_emaranhamento == " - "):
                st.error("Selecione uma porta de emaranhamento.")
            else:
                # ===== PREPARAR FEATURES =====
                if len(selected_features) > 0:
                    X_feat = extrair_features_dataset(X_raw, selected_features)
                else:
                    # usa TODAS as features calculáveis
                    X_feat = extrair_features_dataset(X_raw, features_disponiveis)
    
                X_np = X_feat.values
                y_np = np.array(y)
    
                # ===== AMPLITUDE CASE: fazer padding =====
                if encoding_method == "Amplitude encoding":
                    dim = X_np.shape[1]
                    dim2 = proxima_potencia_de_2(dim)
                    if dim2 != dim:
                        # padding com zeros
                        pad_width = dim2 - dim
                        X_np = np.pad(X_np, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)
                    n_qubits = int(np.log2(X_np.shape[1]))
                    st.info(f"Amplitude encoding requer 2^n features. Ajustei para {X_np.shape[1]} e usei {n_qubits} qubits.")
                else:
                    n_qubits = X_np.shape[1]
    
                # normalizar
                scaler = StandardScaler()
                X_np = scaler.fit_transform(X_np)
    
                # criar pesos
                if tipo_circuito == "Real Amplitudes":
                    weights = pnp.random.uniform(0, 2*np.pi, size=(1, n_qubits))
                else:
                    weights = pnp.random.uniform(0, 2*np.pi, size=(n_qubits, 3))
    
                circuit = criar_circuito(
                    encoding_method,
                    eixos,
                    tipo_circuito,
                    porta_emaranhamento,
                    n_qubits
                )
    
                # gera saídas quânticas
                saidas = []
                for x in X_np:
                    saidas.append(circuit(x, weights))
                saidas = np.array(saidas)
    
                # classificador clássico em cima
                X_train, X_test, y_train, y_test = train_test_split(
                    saidas, y_np, test_size=0.2, random_state=42
                )
                clf = SVC()
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
    
                st.success(f"{textos_ml['acc']} {acc:.3f}")
    
        # botão pra voltar mesmo se der erro
        st.divider()
        if st.button("⬅ Voltar para a página inicial"):
            st.session_state["pagina"] = "inicio"
            st.rerun()
        
        with st.sidebar:
            if st.button(textos["ini"]):
                st.session_state['pagina'] = 'inicio'
                st.rerun()

    elif st.session_state['pagina'] == 'inferencia':
        st.subheader(textos["pagina_inferencia"])
        st.markdown(textos_inf["intro_qbn"])
        #st.write(textos_inf["pagina_inf"])

        # ============================================================
        # QBN — Classical + Quantum (shots + AA) inference helpers
        # ============================================================
        
        def _qbn_init_state():
            if "qbn" not in st.session_state:
                st.session_state.qbn = {"nodes": {}, "selected": None, "last": None}
        
        def _qbn_states_from_card(card: int) -> List[str]:
            card = int(card)
            return [f"s{i}" for i in range(max(1, card))]

        def _qbn_states_from_points(n: int) -> List[str]:
            n = int(n)
            return [f"x{i}" for i in range(max(1, n))]
        
        def _qbn_make_scipy_dist(dist_name: str, params: Dict[str, float]):
            from scipy import stats
            dist_name = (dist_name or "").lower().strip()
        
            if dist_name == "normal":
                mu = float(params.get("mu", 0.0))
                sigma = float(params.get("sigma", 1.0))
                sigma = max(sigma, 1e-12)
                return stats.norm(loc=mu, scale=sigma)
        
            if dist_name == "exponential":
                lam = float(params.get("lam", 1.0))  # rate λ
                lam = max(lam, 1e-12)
                return stats.expon(scale=1.0 / lam)
        
            if dist_name == "weibull":
                k = float(params.get("k", 1.5))       # shape
                scale = float(params.get("scale", 1.0))
                k = max(k, 1e-12)
                scale = max(scale, 1e-12)
                return stats.weibull_min(c=k, scale=scale)
        
            raise ValueError(f"Unsupported distribution: {dist_name}")
        
        def _qbn_discretize_continuous(dist, n_points: int, q_low: float, q_high: float) -> Tuple[List[float], List[float], List[float]]:
            """
            Returns: (edges, midpoints, probs)
            - edges: n_points+1 cut points
            - midpoints: representative value per bin (n_points)
            - probs: probability per bin (n_points), normalized
            """
            import numpy as np
        
            n_points = int(n_points)
            q_low = float(q_low)
            q_high = float(q_high)
        
            q_low = max(0.0, min(q_low, 1.0))
            q_high = max(0.0, min(q_high, 1.0))
            if q_high <= q_low:
                q_low, q_high = 0.001, 0.999
        
            # Finite support cut (avoid +/-inf)
            x_low = float(dist.ppf(q_low))
            x_high = float(dist.ppf(q_high))
        
            if not np.isfinite(x_low):
                x_low = float(dist.ppf(0.001))
            if not np.isfinite(x_high):
                x_high = float(dist.ppf(0.999))
        
            edges = np.linspace(x_low, x_high, n_points + 1)
        
            # probs from CDF differences
            cdf_edges = dist.cdf(edges)
            probs = np.diff(cdf_edges)
            probs = [float(x) for x in probs.tolist()]
            probs = _qbn_normalize_row(probs)
        
            mids = ((edges[:-1] + edges[1:]) / 2.0).tolist()
            edges = [float(x) for x in np.array(edges, dtype=float).tolist()]
            mids = [float(x) for x in mids]
        
            return edges, mids, probs



        
        def _qbn_normalize_row(vals: List[float]) -> List[float]:
            import numpy as np  # garante que np existe aqui
            arr = np.array([float(v) for v in vals], dtype=float)
            arr[arr < 0] = 0.0
            s = float(arr.sum())
            if s <= 0:
                # fallback uniform
                return [1.0 / len(arr)] * len(arr)
            return (arr / s).tolist()

        def _qbn_norm_cdf(x: float, mu: float, sigma: float) -> float:
            import math
            z = (x - mu) / (sigma * math.sqrt(2.0))
            return 0.5 * (1.0 + math.erf(z))
        
        def _qbn_norm_ppf(p: float, mu: float, sigma: float) -> float:
            # Approx. inverse CDF (Acklam), pure python
            import math
            if p <= 0.0:
                return -math.inf
            if p >= 1.0:
                return math.inf
        
            # Coefficients
            a = [-3.969683028665376e+01,  2.209460984245205e+02,
                 -2.759285104469687e+02,  1.383577518672690e+02,
                 -3.066479806614716e+01,  2.506628277459239e+00]
            b = [-5.447609879822406e+01,  1.615858368580409e+02,
                 -1.556989798598866e+02,  6.680131188771972e+01,
                 -1.328068155288572e+01]
            c = [-7.784894002430293e-03, -3.223964580411365e-01,
                 -2.400758277161838e+00, -2.549732539343734e+00,
                  4.374664141464968e+00,  2.938163982698783e+00]
            d = [ 7.784695709041462e-03,  3.224671290700398e-01,
                  2.445134137142996e+00,  3.754408661907416e+00]
        
            plow = 0.02425
            phigh = 1.0 - plow
        
            if p < plow:
                q = math.sqrt(-2.0 * math.log(p))
                x = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                    ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
            elif p > phigh:
                q = math.sqrt(-2.0 * math.log(1.0 - p))
                x = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                     ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
            else:
                q = p - 0.5
                r = q*q
                x = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
                    (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
        
            # One step of Halley refinement
            e = _qbn_norm_cdf(x, 0.0, 1.0) - p
            u = e * math.sqrt(2.0*math.pi) * math.exp(0.5*x*x)
            x = x - u / (1.0 + 0.5*x*u)
        
            return mu + sigma * x
        
        def _qbn_exp_cdf(x: float, lam: float) -> float:
            import math
            if x <= 0.0:
                return 0.0
            return 1.0 - math.exp(-lam * x)
        
        def _qbn_exp_ppf(p: float, lam: float) -> float:
            import math
            p = min(max(p, 1e-15), 1.0 - 1e-15)
            return -math.log(1.0 - p) / lam
        
        def _qbn_weibull_cdf(x: float, k: float, scale: float) -> float:
            import math
            if x <= 0.0:
                return 0.0
            return 1.0 - math.exp(-((x / scale) ** k))
        
        def _qbn_weibull_ppf(p: float, k: float, scale: float) -> float:
            import math
            p = min(max(p, 1e-15), 1.0 - 1e-15)
            return scale * ((-math.log(1.0 - p)) ** (1.0 / k))
        
        def _qbn_discretize_continuous(dist: str, params: dict, n: int, q_low=1e-3, q_high=1-1e-3):
            """
            Returns (probs, edges, mids)
            probs: length n, sums to 1
            edges: length n+1
            mids: length n
            """
            import numpy as np
        
            n = int(n)
            dist = (dist or "").lower().strip()
        
            if dist == "normal":
                mu = float(params.get("mu", 0.0))
                sigma = float(params.get("sigma", 1.0))
                sigma = max(sigma, 1e-12)
                lo = _qbn_norm_ppf(q_low, mu, sigma)
                hi = _qbn_norm_ppf(q_high, mu, sigma)
                cdf = lambda x: _qbn_norm_cdf(x, mu, sigma)
        
            elif dist == "exponential":
                lam = float(params.get("lam", 1.0))
                lam = max(lam, 1e-12)
                lo = 0.0
                hi = _qbn_exp_ppf(q_high, lam)
                cdf = lambda x: _qbn_exp_cdf(x, lam)
        
            elif dist == "weibull":
                k = float(params.get("k", 1.5))
                scale = float(params.get("scale", 1.0))
                k = max(k, 1e-12)
                scale = max(scale, 1e-12)
                lo = 0.0
                hi = _qbn_weibull_ppf(q_high, k, scale)
                cdf = lambda x: _qbn_weibull_cdf(x, k, scale)
        
            else:
                raise ValueError(f"Unknown distribution: {dist}")
        
            edges = np.linspace(lo, hi, n + 1)
            probs = []
            mids = []
            for i in range(n):
                a = float(edges[i])
                b = float(edges[i + 1])
                p = float(cdf(b) - cdf(a))
                probs.append(max(0.0, p))
                mids.append(0.5 * (a + b))
        
            probs = _qbn_normalize_row(probs)
            return probs, edges.tolist(), mids
        
        def _qbn_resize_cpt_for_new_card(info: dict, new_card: int):
            """Keeps CPT keys, but resets each row to uniform of length new_card."""
            new_card = int(new_card)
            info["card"] = new_card
            info["states"] = _qbn_states_from_card(new_card)
        
            cpt = info.get("cpt", {})
            new_cpt = {}
            for k in cpt.keys():
                new_cpt[k] = [1.0 / new_card] * new_card
            if not new_cpt:
                new_cpt = {(): [1.0 / new_card] * new_card}
            info["cpt"] = new_cpt



        
        
        def _qbn_topological_order(nodes: Dict[str, Any]) -> List[str]:
            # Kahn's algorithm
            indeg = {n: 0 for n in nodes}
            children = {n: [] for n in nodes}
            for n, info in nodes.items():
                for p in info.get("parents", []):
                    if p not in nodes:
                        continue
                    indeg[n] += 1
                    children[p].append(n)
        
            q = [n for n, d in indeg.items() if d == 0]
            order = []
            while q:
                cur = q.pop(0)
                order.append(cur)
                for ch in children[cur]:
                    indeg[ch] -= 1
                    if indeg[ch] == 0:
                        q.append(ch)
        
            # if cycle exists, just return insertion order to avoid crash
            if len(order) != len(nodes):
                return list(nodes.keys())
            return order
        
        def _qbn_cond_prob(bn: Dict[str, Any], node: str, asg: Dict[str, str]) -> float:
            info = bn["nodes"][node]
            parents = info["parents"]
            parent_vals = tuple(asg[p] for p in parents)
            probs = info["cpt"].get(parent_vals)
            if probs is None:
                # default uniform
                probs = [1.0 / len(info["states"])] * len(info["states"])
            probs = _qbn_normalize_row(probs)
            st_idx = info["states"].index(asg[node])
            return float(probs[st_idx])
        
        def _qbn_joint_prob(bn: Dict[str, Any], asg: Dict[str, str]) -> float:
            p = 1.0
            for n in bn["order"]:
                p *= _qbn_cond_prob(bn, n, asg)
            return float(p)
        
        def _qbn_total_state_space(bn: Dict[str, Any]) -> int:
            total = 1
            for n in bn["order"]:
                total *= max(1, len(bn["nodes"][n]["states"]))
            return int(total)
        
        def _qbn_exact_posterior(bn: Dict[str, Any], query_nodes: List[str], evidence: Dict[str, str],
                                max_states: int = 200000) -> Optional[Dict[Tuple[Tuple[str, str], ...], float]]:
            total = _qbn_total_state_space(bn)
            if total > max_states:
                return None
        
            nodes = bn["order"]
            domains = {n: bn["nodes"][n]["states"] for n in nodes}
        
            post: Dict[Tuple[Tuple[str, str], ...], float] = {}
            denom = 0.0
        
            for values in itertools.product(*[domains[n] for n in nodes]):
                asg = dict(zip(nodes, values))
                ok = True
                for evn, evv in evidence.items():
                    if asg.get(evn) != evv:
                        ok = False
                        break
                if not ok:
                    continue
        
                jp = _qbn_joint_prob(bn, asg)
                if jp <= 0:
                    continue
        
                denom += jp
                outcome = tuple((qn, asg[qn]) for qn in query_nodes)
                post[outcome] = post.get(outcome, 0.0) + jp
        
            if denom <= 0:
                return {tuple((qn, evidence.get(qn, "")) for qn in query_nodes): 0.0}
        
            for k in list(post.keys()):
                post[k] /= denom
            return post
        
        def _qbn_exact_marginals(bn: Dict[str, Any], evidence: Dict[str, str],
                                max_states: int = 200000) -> Optional[Dict[str, Dict[str, float]]]:
            
            import numpy as np
            total = _qbn_total_state_space(bn)
            if total > max_states:
                return None
        
            nodes = bn["order"]
            domains = {n: bn["nodes"][n]["states"] for n in nodes}
            marg = {n: {s: 0.0 for s in domains[n]} for n in nodes}
            denom = 0.0
        
            for values in itertools.product(*[domains[n] for n in nodes]):
                asg = dict(zip(nodes, values))
                ok = True
                for evn, evv in evidence.items():
                    if asg.get(evn) != evv:
                        ok = False
                        break
                if not ok:
                    continue
                jp = _qbn_joint_prob(bn, asg)
                if jp <= 0:
                    continue
                denom += jp
                for n in nodes:
                    marg[n][asg[n]] += jp
        
            if denom <= 0:
                return {n: {s: 0.0 for s in domains[n]} for n in nodes}
        
            for n in nodes:
                for s in domains[n]:
                    marg[n][s] /= denom
            return marg
        
        def _qbn_mc_likelihood_weighting(bn: Dict[str, Any], query_nodes: List[str], evidence: Dict[str, str],
                                        n_samples: int = 2000, seed: Optional[int] = None) -> Dict[Tuple[Tuple[str, str], ...], float]:
            import numpy as np
            rng = np.random.default_rng(seed)
            counts: Dict[Tuple[Tuple[str, str], ...], float] = {}
            weight_sum = 0.0
        
            for _ in range(int(n_samples)):
                w = 1.0
                asg: Dict[str, str] = {}
        
                for node in bn["order"]:
                    info = bn["nodes"][node]
                    parents = info["parents"]
                    parent_vals = tuple(asg[pn] for pn in parents)
                    states = info["states"]
                    probs = info["cpt"].get(parent_vals)
                    if probs is None:
                        probs = [1.0 / max(1, len(states))] * max(1, len(states))
                    probs = np.array(_qbn_normalize_row(probs), dtype=float)
        
                    if node in evidence:
                        st = evidence[node]
                        if st not in states:
                            st = states[0]
                        w *= float(probs[states.index(st)])
                        asg[node] = st
                    else:
                        st = rng.choice(states, p=probs)
                        asg[node] = st
        
                outcome = tuple((qn, asg[qn]) for qn in query_nodes)
                counts[outcome] = counts.get(outcome, 0.0) + w
                weight_sum += w
        
            if weight_sum <= 0:
                return {tuple((qn, evidence.get(qn, "")) for qn in query_nodes): 0.0}
        
            for k in list(counts.keys()):
                counts[k] /= weight_sum
            return counts
        
        def _qbn_mc_marginals_lw(bn: Dict[str, Any], evidence: Dict[str, str], n_samples: int = 2000,
                                 seed: Optional[int] = None) -> Dict[str, Dict[str, float]]:
            import numpy as np
            rng = np.random.default_rng(seed)
            nodes = bn["order"]
            domains = {n: bn["nodes"][n]["states"] for n in nodes}
            marg = {n: {s: 0.0 for s in domains[n]} for n in nodes}
            weight_sum = 0.0
        
            for _ in range(int(n_samples)):
                w = 1.0
                asg: Dict[str, str] = {}
                for node in nodes:
                    info = bn["nodes"][node]
                    parents = info["parents"]
                    parent_vals = tuple(asg[pn] for pn in parents)
                    states = info["states"]
                    probs = info["cpt"].get(parent_vals)
                    if probs is None:
                        probs = [1.0 / max(1, len(states))] * max(1, len(states))
                    probs = np.array(_qbn_normalize_row(probs), dtype=float)
        
                    if node in evidence:
                        st = evidence[node]
                        if st not in states:
                            st = states[0]
                        w *= float(probs[states.index(st)])
                        asg[node] = st
                    else:
                        st = rng.choice(states, p=probs)
                        asg[node] = st
        
                for n in nodes:
                    marg[n][asg[n]] += w
                weight_sum += w
        
            if weight_sum <= 0:
                return {n: {s: 0.0 for s in domains[n]} for n in nodes}
        
            for n in nodes:
                for s in domains[n]:
                    marg[n][s] /= weight_sum
            return marg
        
        def _qbn_outcome_label(bitstring: str) -> str:
            # Render in dirac notation: |0101>
            return f"|{bitstring}>"

        def _qbn_node_bits(bn: Dict[str, Any]) -> Dict[str, int]:
            import math
            bits: Dict[str, int] = {}
            for n in bn.get("order", []):
                ni = len(bn["nodes"][n]["states"])
                mi = int(math.ceil(math.log2(ni))) if ni > 1 else 1
                bits[n] = mi
            return bits

        def _qbn_node_offsets(bn: Dict[str, Any]) -> Dict[str, int]:
            offs: Dict[str, int] = {}
            cur = 0
            bits = _qbn_node_bits(bn)
            for n in bn.get("order", []):
                offs[n] = cur
                cur += int(bits.get(n, 1))
            return offs

        def _qbn_int_to_bits(x: int, width: int) -> str:
            return format(int(x), f"0{int(width)}b")

        def _qbn_bits_to_int(bits: str) -> int:
            return int(bits, 2) if bits else 0

        def _qbn_encode_assignment(bn: Dict[str, Any], asg: Dict[str, str]) -> str:
            bits = _qbn_node_bits(bn)
            out = []
            for n in bn.get("order", []):
                states = bn["nodes"][n]["states"]
                idx = states.index(asg[n])
                out.append(_qbn_int_to_bits(idx, bits[n]))
            return "".join(out)

        def _qbn_decode_bitstring(bn: Dict[str, Any], bitstring: str) -> Tuple[Optional[Dict[str, str]], bool]:
            bits = _qbn_node_bits(bn)
            offs = _qbn_node_offsets(bn)
            asg: Dict[str, str] = {}
            for n in bn.get("order", []):
                o = int(offs[n])
                w = int(bits[n])
                chunk = bitstring[o:o+w]
                idx = _qbn_bits_to_int(chunk)
                states = bn["nodes"][n]["states"]
                if idx >= len(states):
                    return None, False
                asg[n] = states[idx]
            return asg, True

        def _qbn_filter_evidence_bits(bn: Dict[str, Any], evidence: Dict[str, str]) -> Dict[int, int]:
            req: Dict[int, int] = {}
            bits = _qbn_node_bits(bn)
            offs = _qbn_node_offsets(bn)

            for n, st in evidence.items():
                if n not in bn.get("order", []):
                    continue
                states = bn["nodes"][n]["states"]
                if st not in states:
                    continue
                idx = states.index(st)
                bstr = _qbn_int_to_bits(idx, bits[n])
                o = int(offs[n])
                for j, ch in enumerate(bstr):
                    req[o + j] = int(ch)
            return req

        
        def _qbn_is_binary_bn(bn: Dict[str, Any]) -> bool:
            return all(len(bn["nodes"][n]["states"]) == 2 for n in bn["order"])
        
        def _qbn_joint_distribution_enumerate(bn: Dict[str, Any], max_states: int = 200000) -> Optional[Tuple[List[str], Any]]:
            import numpy as np
            total = _qbn_total_state_space(bn)
            if total > max_states:
                return None

            nodes = bn["order"]
            domains = {n: bn["nodes"][n]["states"] for n in nodes}

            outcomes: List[str] = []
            probs: List[float] = []
            for values in itertools.product(*[domains[n] for n in nodes]):
                asg = dict(zip(nodes, values))
                jp = _qbn_joint_prob(bn, asg)
                if jp < 0:
                    jp = 0.0
                outcomes.append(_qbn_encode_assignment(bn, asg))
                probs.append(float(jp))

            p = np.array(probs, dtype=float)
            s = float(p.sum())
            if s <= 0:
                return None
            p = p / s
            return outcomes, p
        
        def _qbn_filter_evidence_bitstrings(bn: Dict[str, Any], evidence: Dict[str, str]) -> Dict[int, int]:
            # returns mapping from global bit position -> required bit (0/1)
            return _qbn_filter_evidence_bits(bn, evidence)
        
        def _qbn_quantum_shots(bn: Dict[str, Any], query_nodes: List[str], evidence: Dict[str, str],
                              shots: int = 5000, seed: Optional[int] = None,
                              max_states: int = 200000) -> Optional[Dict[str, Any]]:
            import numpy as np
            # Quantum shots: ideal q-sampling of the joint distribution + postselection on evidence.
            #if not _qbn_is_binary_bn(bn):
            #    return None
        
            jd = _qbn_joint_distribution_enumerate(bn, max_states=max_states)
            if jd is None:
                return None
            outcomes, p = jd
        
            rng = np.random.default_rng(seed)
            idxs = rng.choice(len(outcomes), size=int(shots), replace=True, p=p)
            samples = [outcomes[i] for i in idxs]
        
            req = _qbn_filter_evidence_bitstrings(bn, evidence)
            accepted = []
            for bs in samples:
                ok = True
                for pos, bit in req.items():
                    if int(bs[pos]) != int(bit):
                        ok = False
                        break
                if ok:
                    accepted.append(bs)
                    
            acc_n = len(accepted)
            acc_rate = acc_n / max(1, int(shots))
        
            # posterior over query nodes, computed from accepted
            nodes = bn["order"]

            post: Dict[Tuple[Tuple[str, str], ...], float] = {}
            # marginals over all nodes
            marg = {n: {s: 0.0 for s in bn["nodes"][n]["states"]} for n in nodes}

            if acc_n <= 0:
                return {"post": {tuple((qn, evidence.get(qn, "")) for qn in query_nodes): 0.0},
                        "marg": marg, "accepted": 0, "acc_rate": acc_rate,
                        "counts": {}}

            counts_full: Dict[str, int] = {}
            for bs in accepted:
                counts_full[bs] = counts_full.get(bs, 0) + 1

                asg, ok = _qbn_decode_bitstring(bn, bs)
                if not ok or asg is None:
                    continue

                # marginals
                for n in nodes:
                    marg[n][asg[n]] += 1

                # query outcome
                out_pairs = []
                for qn in query_nodes:
                    if qn not in nodes:
                        continue
                    out_pairs.append((qn, asg[qn]))
                outcome = tuple(out_pairs)
                post[outcome] = post.get(outcome, 0.0) + 1

        
            for k in list(post.keys()):
                post[k] /= acc_n
            for n in nodes:
                for s in bn["nodes"][n]["states"]:
                    marg[n][s] /= acc_n
        
            return {"post": post, "marg": marg, "accepted": acc_n, "acc_rate": acc_rate,
                    "counts": counts_full}
        
        def _qbn_aa_amplified_distribution(outcomes: List[str], p: Any, good_mask: Any, k: int) -> Any:
            import numpy as np
            p_good = float(p[good_mask].sum())
            if p_good <= 0:
                return p.copy()
            if p_good >= 1:
                return p.copy()
        
            theta = math.asin(math.sqrt(p_good))
            # Grover iterations rotate by (2k+1)theta
            p_good_prime = (math.sin((2 * int(k) + 1) * theta) ** 2)
        
            p_bad = 1.0 - p_good
            p_bad_prime = 1.0 - p_good_prime
        
            p2 = np.zeros_like(p)
            # keep relative weights inside good/bad
            p2[good_mask] = p[good_mask] / p_good * p_good_prime
            p2[~good_mask] = p[~good_mask] / p_bad * p_bad_prime
            # renormalize numerical drift
            s = float(p2.sum())
            if s > 0:
                p2 /= s
            return p2
        
        def _qbn_quantum_aa_shots(bn: Dict[str, Any], query_nodes: List[str], evidence: Dict[str, str],
                                 shots: int = 5000, seed: Optional[int] = None,
                                 k: int = 0, max_states: int = 200000) -> Optional[Dict[str, Any]]:
            import numpy as np
            #if not _qbn_is_binary_bn(bn):
            #    return None
        
            jd = _qbn_joint_distribution_enumerate(bn, max_states=max_states)
            if jd is None:
                return None
            outcomes, p = jd
        
            # good states = those matching evidence
            req = _qbn_filter_evidence_bitstrings(bn, evidence)
            if len(req) == 0:
                # nothing to amplify
                return _qbn_quantum_shots(bn, query_nodes, evidence, shots=shots, seed=seed, max_states=max_states)
        
            good_mask = np.ones(len(outcomes), dtype=bool)
            for i, bs in enumerate(outcomes):
                ok = True
                for pos, bit in req.items():
                    if int(bs[pos]) != int(bit):
                        ok = False
                        break
                good_mask[i] = ok
        
            p2 = _qbn_aa_amplified_distribution(outcomes, p, good_mask, k=int(k))
        
            rng = np.random.default_rng(seed)
            idxs = rng.choice(len(outcomes), size=int(shots), replace=True, p=p2)
            samples = [outcomes[i] for i in idxs]
        
            # postselect (still needed because AA isn't perfect unless k is optimal)
            accepted = []
            for bs in samples:
                ok = True
                for pos, bit in req.items():
                    if int(bs[pos]) != int(bit):
                        ok = False
                        break
                if ok:
                    accepted.append(bs)
        
            acc_n = len(accepted)
            acc_rate = acc_n / max(1, int(shots))
        
            nodes = bn["order"]
            post: Dict[Tuple[Tuple[str, str], ...], float] = {}
            marg = {n: {s: 0.0 for s in bn["nodes"][n]["states"]} for n in nodes}
        
            if acc_n <= 0:
                return {"post": {tuple((qn, evidence.get(qn, "")) for qn in query_nodes): 0.0},
                        "marg": marg, "accepted": 0, "acc_rate": acc_rate,
                        "counts": {}}
        
            counts_full: Dict[str, int] = {}
            for bs in accepted:
                counts_full[bs] = counts_full.get(bs, 0) + 1

                asg, ok = _qbn_decode_bitstring(bn, bs)
                if not ok or asg is None:
                    continue

                for n in nodes:
                    marg[n][asg[n]] += 1

                out_pairs = []
                for qn in query_nodes:
                    if qn not in nodes:
                        continue
                    out_pairs.append((qn, asg[qn]))
                outcome = tuple(out_pairs)
                post[outcome] = post.get(outcome, 0.0) + 1
        
            for k0 in list(post.keys()):
                post[k0] /= acc_n
            for n in nodes:
                for s in bn["nodes"][n]["states"]:
                    marg[n][s] /= acc_n
        
            return {"post": post, "marg": marg, "accepted": acc_n, "acc_rate": acc_rate,
                    "counts": counts_full}
        
        def _qbn_best_grover_k(p_good: float, k_max: int = 12) -> int:
            # heuristic optimal k = floor(pi/(4*theta) - 1/2)
            if p_good <= 0 or p_good >= 1:
                return 0
            theta = math.asin(math.sqrt(p_good))
            k = int(max(0, math.floor((math.pi / (4 * theta)) - 0.5)))
            return int(min(k, int(k_max)))
        

        def _qbn_build_dot_from_nodes(bn_nodes: dict) -> str:
            # Graphviz DOT
            lines = ["digraph BN {", 'rankdir="LR";', 'node [shape=box];']
            # nodes
            for n in bn_nodes.keys():
                safe = n.replace('"', '\\"')
                lines.append(f'"{safe}";')
            # edges parent -> child
            for child, info in bn_nodes.items():
                for parent in info.get("parents", []):
                    p = parent.replace('"', '\\"')
                    c = child.replace('"', '\\"')
                    lines.append(f'"{p}" -> "{c}";')
            lines.append("}")
            return "\n".join(lines)

        def _qbn_qubit_accounting(bn: Dict[str, Any]) -> Dict[str, Any]:
            import math
        
            order = bn.get("order", [])
            nodes = bn.get("nodes", {})
        
            # qubits por nó: mi = ceil(log2(ni))
            node_bits: Dict[str, int] = {}
            for n in order:
                ni = len(nodes[n]["states"])
                mi = int(math.ceil(math.log2(ni))) if ni > 1 else 1
                node_bits[n] = mi
        
            q_nodes = int(sum(node_bits.values()))
        
            # ancillas (estimativa “paper-style”):
            # - se binária: mBN,2 = s + max(|Pi|) - 1  => anc = max(|Pi|) - 1
            # - caso geral: anc = max_i( mq,Pi(Vi) + mi - 1 )
            all_binary = all(len(nodes[n]["states"]) == 2 for n in order) if order else True
        
            if all_binary:
                max_parents = 0
                for n in order:
                    max_parents = max(max_parents, len(nodes[n].get("parents", [])))
                q_anc = int(max(0, max_parents - 1))
            else:
                anc_candidates = []
                for n in order:
                    parents = nodes[n].get("parents", [])
                    if len(parents) == 0:
                        continue
                    mq_parents = int(sum(node_bits.get(p, 1) for p in parents))
                    anc_candidates.append(int(mq_parents + node_bits[n] - 1))
                q_anc = int(max(anc_candidates)) if anc_candidates else 0
        
            q_total = int(q_nodes + q_anc)
        
            # mapeamento: indices sequenciais para qubits de nós; ancillas no final
            mapping = []
            q = 0
            for n in order:
                mi = int(node_bits[n])
                for j in range(mi):
                    mapping.append({"qubit": q, "role": "node", "node": n, "slot": j})
                    q += 1
            for a in range(int(q_anc)):
                mapping.append({"qubit": q, "role": "ancilla", "node": "", "slot": a})
                q += 1
        
            return {
                "q_total": q_total,
                "q_nodes": q_nodes,
                "q_anc": q_anc,
                "node_bits": node_bits,
                "mapping": mapping,
                "all_binary": all_binary,
            }

        def _qbn_node_bits(bn: Dict[str, Any]) -> Dict[str, int]:
            import math
            bits = {}
            for n in bn["order"]:
                ni = len(bn["nodes"][n]["states"])
                bits[n] = int(math.ceil(math.log2(ni))) if ni > 1 else 1
            return bits
        
        def _qbn_node_offsets(bn: Dict[str, Any]) -> Dict[str, int]:
            # offset (posição inicial) de cada nó no bitstring global
            offs = {}
            cur = 0
            bits = _qbn_node_bits(bn)
            for n in bn["order"]:
                offs[n] = cur
                cur += bits[n]
            return offs
        
        def _qbn_int_to_bits(x: int, width: int) -> str:
            # MSB->LSB
            return format(int(x), f"0{int(width)}b")
        
        def _qbn_bits_to_int(bits: str) -> int:
            return int(bits, 2) if bits else 0
        
        def _qbn_encode_assignment(bn: Dict[str, Any], asg: Dict[str, str]) -> str:
            bits = _qbn_node_bits(bn)
            out = []
            for n in bn["order"]:
                states = bn["nodes"][n]["states"]
                idx = states.index(asg[n])
                out.append(_qbn_int_to_bits(idx, bits[n]))
            return "".join(out)
        
        def _qbn_decode_bitstring(bn: Dict[str, Any], bitstring: str) -> Tuple[Optional[Dict[str, str]], bool]:
            bits = _qbn_node_bits(bn)
            offs = _qbn_node_offsets(bn)
            asg: Dict[str, str] = {}
            for n in bn["order"]:
                o = offs[n]
                w = bits[n]
                chunk = bitstring[o:o+w]
                idx = _qbn_bits_to_int(chunk)
                states = bn["nodes"][n]["states"]
                if idx >= len(states):
                    return None, False  # bitstring inválido para esse nó
                asg[n] = states[idx]
            return asg, True
        
        def _qbn_filter_evidence_bits(bn: Dict[str, Any], evidence: Dict[str, str]) -> Dict[int, int]:
            # retorna mapping posicao_bit_global -> bit requerido (0/1), agora multi-bit por nó
            req: Dict[int, int] = {}
            bits = _qbn_node_bits(bn)
            offs = _qbn_node_offsets(bn)
        
            for n, st in evidence.items():
                if n not in bn["order"]:
                    continue
                states = bn["nodes"][n]["states"]
                if st not in states:
                    continue
                idx = states.index(st)
                bstr = _qbn_int_to_bits(idx, bits[n])
                o = offs[n]
                for j, ch in enumerate(bstr):
                    req[o + j] = int(ch)
            return req

        def pagina_inferencia_qbn(textos: Dict[str, str], textos_inf: Dict[str, str]):
            import pandas as pd
            import numpy as np
            _qbn_init_state()
        
            #st.title(textos_inf["titulo_app"])
            #st.caption(textos_inf["subtitulo_app"])
            st.divider()
            st.markdown(textos_inf["guia_qbn"])

            # ============================
            # UI — build BN
            # ============================
            col_list, col_edit = st.columns([1, 2], gap="large")
        
            with col_list:
                st.subheader(textos_inf["def_nos"])
                st.caption(textos_inf["def_nos_desc"])
            
                with st.container(border=True):
                    nome = st.text_input(textos_inf["nome_no"], value="", key="qbn_new_node_name")
                
                    is_cont = st.checkbox(
                        textos_inf["node_continuous"],
                        value=bool(st.session_state.get("qbn_new_node_cont", False)),
                        help=textos_inf.get("node_continuous_help", ""),
                        key="qbn_new_node_cont",
                    )
                
                    # defaults
                    card = 2
                    cont_payload = None
                
                    if not is_cont:
                        card = st.number_input(
                            textos_inf["card_no"],
                            min_value=2,
                            max_value=8,
                            value=2,
                            step=1,
                            key="qbn_new_node_card",
                        )
                    else:
                        st.caption(textos_inf.get("dist_params", "Parâmetros da distribuição"))
                
                        dist_family = st.selectbox(
                            textos_inf["dist_family"],
                            options=["Weibull", "Normal", "Exponential"],
                            index=0,
                            help=textos_inf.get("dist_family_help", ""),
                            key="qbn_new_node_dist_family",
                        )
                
                        # params UI
                        params = {}
                        if dist_family == "Normal":
                            c1, c2 = st.columns(2)
                            with c1:
                                params["mu"] = st.number_input("μ (mean)", value=0.0, step=0.1, key="qbn_new_node_mu")
                            with c2:
                                params["sigma"] = st.number_input("σ (std)", min_value=1e-6, value=1.0, step=0.1, key="qbn_new_node_sigma")
                            dist_key = "normal"
                
                        elif dist_family == "Exponential":
                            params["lam"] = st.number_input("λ (rate)", min_value=1e-6, value=1.0, step=0.1, key="qbn_new_node_lam")
                            dist_key = "exponential"
                
                        else:  # Weibull
                            c1, c2 = st.columns(2)
                            with c1:
                                params["k"] = st.number_input("k (shape)", min_value=1e-6, value=1.5, step=0.1, key="qbn_new_node_k")
                            with c2:
                                params["scale"] = st.number_input("scale", min_value=1e-6, value=1.0, step=0.1, key="qbn_new_node_scale")
                            dist_key = "weibull"
                
                        n_points = st.number_input(
                            textos_inf["dist_points"],
                            min_value=2,
                            max_value=32,
                            value=8,
                            step=1,
                            help=textos_inf.get("dist_points_help", ""),
                            key="qbn_new_node_npoints",
                        )
                
                
                        q_low, q_high = st.slider(
                            textos_inf["dist_quantile_range"],
                            min_value=0.0,
                            max_value=1.0,
                            value=(0.001, 0.999),
                            step=0.001,
                            key="qbn_new_node_qrange",
                        )
                
                        auto_root = st.checkbox(
                            textos_inf["dist_root_autofill"],
                            value=True,
                            help=textos_inf.get("dist_root_autofill_help", ""),
                            key="qbn_new_node_autoroot",
                        )
                
                        # preview (agora atualiza ao vivo)
                        try:
                            dist = _qbn_make_scipy_dist(dist_key, params)
                            edges, mids, probs = _qbn_discretize_continuous(dist, int(n_points), float(q_low), float(q_high))
                            with st.expander(textos_inf["dist_preview"], expanded=False):
                                import pandas as pd
                                dfp = pd.DataFrame({
                                    "state": _qbn_states_from_points(int(n_points)),
                                    "x_mid": [round(x, 6) for x in mids],
                                    "prob": [round(p, 6) for p in probs],
                                })
                                st.dataframe(dfp, use_container_width=True, hide_index=True)
                        except Exception as e:
                            st.warning(f"{textos_inf.get('dist_preview','Prévia')}: {e}")
                
                        # for creation
                        card = int(n_points)
                        cont_payload = {
                            "enabled": True,
                            "dist": dist_key,
                            "params": params,
                            "n_points": int(n_points),
                            "q_low": float(q_low),
                            "q_high": float(q_high),
                            "auto_root": bool(auto_root),
                        }
                
                    # botão normal (fora do form)
                    submitted = st.button(textos_inf["add_no"], key="qbn_add_node_btn")
                
                    if submitted:
                        nome = (nome or "").strip()
                        if nome and (nome not in st.session_state.qbn["nodes"]):
                
                            if cont_payload and cont_payload.get("enabled", False):
                                dist = _qbn_make_scipy_dist(cont_payload["dist"], cont_payload["params"])
                                edges, mids, probs = _qbn_discretize_continuous(
                                    dist,
                                    cont_payload["n_points"],
                                    cont_payload["q_low"],
                                    cont_payload["q_high"],
                                )
                
                                states = _qbn_states_from_points(int(cont_payload["n_points"]))
                                init_probs = probs if cont_payload.get("auto_root", True) else ([1.0 / len(states)] * len(states))
                
                                st.session_state.qbn["nodes"][nome] = {
                                    "card": int(card),
                                    "states": states,
                                    "parents": [],
                                    "cpt": {(): init_probs},
                                    "continuous": True,
                                    "continuous_dist": cont_payload["dist"],
                                    "continuous_params": cont_payload["params"],
                                    "continuous_qrange": (cont_payload["q_low"], cont_payload["q_high"]),
                                    "continuous_midpoints": mids,
                                    "continuous_edges": edges,
                                }
                            else:
                                st.session_state.qbn["nodes"][nome] = {
                                    "card": int(card),
                                    "states": _qbn_states_from_card(int(card)),
                                    "parents": [],
                                    "cpt": {(): [1.0 / int(card)] * int(card)},
                                }
                
                            st.session_state.qbn["selected"] = nome
                            st.rerun()

        
                nodes = list(st.session_state.qbn["nodes"].keys())
                if nodes:
                    sel = st.selectbox(textos_inf["selecionar_no"], options=nodes, index=nodes.index(st.session_state.qbn["selected"]) if st.session_state.qbn["selected"] in nodes else 0)
                    st.session_state.qbn["selected"] = sel
        
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button(textos_inf["remover_no"]):
                            # remove node + remove as parent from others
                            del st.session_state.qbn["nodes"][sel]
                            for n2 in list(st.session_state.qbn["nodes"].keys()):
                                ps = st.session_state.qbn["nodes"][n2]["parents"]
                                st.session_state.qbn["nodes"][n2]["parents"] = [p for p in ps if p != sel]
                            st.session_state.qbn["selected"] = (list(st.session_state.qbn["nodes"].keys())[0] if st.session_state.qbn["nodes"] else None)
                            st.rerun()
                    with c2:
                        if st.button(textos_inf["limpar_rede"]):
                            st.session_state.qbn = {"nodes": {}, "selected": None, "last": None}
                            st.rerun()
                else:
                    st.info(textos_inf["sem_nos"])
            
                with col_edit:
                    nodes = list(st.session_state.qbn["nodes"].keys())
                
                    if nodes and st.session_state.qbn["selected"]:
                        nsel = st.session_state.qbn["selected"]
                        info = st.session_state.qbn["nodes"][nsel]
                
                        st.subheader(textos_inf["edicao_no"])
                        st.caption(textos_inf["edicao_no_desc"])
                
                        parent_opts = [n for n in nodes if n != nsel]
                        parents = st.multiselect(
                            textos_inf["pais_do_no"],
                            options=parent_opts,
                            default=info.get("parents", []),
                            help=textos_inf["pais_do_no_help"],
                        )
                        info["parents"] = parents
                
                        parents_now = info.get("parents", [])
                
                        # título do editor
                        st.markdown(
                            f"**{textos_inf['probs_raiz']}**" if len(parents_now) == 0 else f"**{textos_inf['cpt']}**"
                        )
                
                        # ----------------------------
                        # Build editor dataframe
                        # ----------------------------
                        if len(parents_now) == 0:
                            df = pd.DataFrame({
                                "state": info["states"],
                                "prob": info.get("cpt", {}).get((), [1.0 / len(info["states"])] * len(info["states"]))
                            })
                
                            edited = st.data_editor(df, num_rows="fixed", hide_index=True, key=f"qbn_root_{nsel}")
                            probs = _qbn_normalize_row(edited["prob"].tolist())
                            info["cpt"] = {(): probs}
                            st.caption(textos_inf["caption_probs"])
                
                        else:
                            parent_states = [st.session_state.qbn["nodes"][p]["states"] for p in parents_now]
                            combos = list(itertools.product(*parent_states))
                
                            rows = []
                            for comb in combos:
                                row = {f"{parents_now[i]}": comb[i] for i in range(len(parents_now))}
                                key = tuple(comb)
                
                                probs = info.get("cpt", {}).get(key)
                                if probs is None:
                                    probs = [1.0 / len(info["states"])] * len(info["states"])
                
                                for j, stt in enumerate(info["states"]):
                                    row[stt] = probs[j]
                
                                rows.append(row)
                
                            df = pd.DataFrame(rows)
                            edited = st.data_editor(df, num_rows="fixed", hide_index=True, key=f"qbn_cpt_{nsel}")
                
                            cpt = {}
                            for _, r in edited.iterrows():
                                key = tuple(r[p] for p in parents_now)
                                probs = [float(r[stt]) for stt in info["states"]]
                                cpt[key] = _qbn_normalize_row(probs)
                
                            info["cpt"] = cpt
                            st.caption(textos_inf["caption_cpt"])
                
                    else:
                        st.info(textos_inf["sem_nos"])

        
                st.divider()

                st.subheader(textos_inf.get("rede_montada", "Rede Bayesiana montada"))
                st.caption(textos_inf["rede_montada_desc"])
                
                bn_nodes = st.session_state.qbn["nodes"]
                
                if not bn_nodes:
                    st.info(textos_inf.get("sem_rede", "Adicione nós para visualizar a rede."))
                else:
                    # 1) Grafo
                    dot = _qbn_build_dot_from_nodes(bn_nodes)
                    st.graphviz_chart(dot, use_container_width=True)
                
                    # 2) Probabilidades inseridas (marginais e condicionais)
                    import pandas as pd
                
                    with st.expander(textos_inf.get("probs_inseridas", "Probabilidades inseridas (marginais e CPTs)"), expanded=True):
                        order = _qbn_topological_order(bn_nodes)
                
                        for n in order:
                            info = bn_nodes[n]
                            states = info["states"]
                            parents = info.get("parents", [])
                
                            st.markdown(f"**{n}**")
                            st.caption(f"{textos_inf['lbl_states']}: {states} | {textos_inf['lbl_parents']}: {parents if parents else '-'}")

                
                            if len(parents) == 0:
                                # root marginal
                                probs = info.get("cpt", {}).get((), [1.0 / len(states)] * len(states))
                                df = pd.DataFrame({"state": states, "P(state)": [float(x) for x in probs]})
                                st.dataframe(df, use_container_width=True, hide_index=True)
                            else:
                                # CPT: each row = combination of parents; columns = child states
                                parent_states = [bn_nodes[p]["states"] for p in parents]
                                combos = list(itertools.product(*parent_states))
                
                                rows = []
                                cpt = info.get("cpt", {})
                                for comb in combos:
                                    key = tuple(comb)
                                    probs = cpt.get(key)
                                    if probs is None:
                                        probs = [1.0 / len(states)] * len(states)
                
                                    row = {parents[i]: comb[i] for i in range(len(parents))}
                                    for j, stt in enumerate(states):
                                        row[f"P({n}={stt})"] = float(probs[j])
                                    rows.append(row)
                
                                df = pd.DataFrame(rows)
                                st.dataframe(df, use_container_width=True, hide_index=True)
                
                            st.markdown("---")


                st.divider()
                st.subheader(textos_inf["evidencia"])
                st.caption(textos_inf["evidencia_desc"])
                # evidence selection (applied to all methods; AA uses it explicitly)
                nodes = list(st.session_state.qbn["nodes"].keys())
                if not nodes:
                    st.info(textos_inf.get("sem_rede", "Adicione nós para visualizar a rede."))
                    evidence: Dict[str, str] = {}
                    query_nodes: List[str] = []
                else:
                    ev_nodes = st.multiselect(textos_inf["nos_evidenciados"], options=nodes, default=[])
                    evidence: Dict[str, str] = {}
                    for evn in ev_nodes:
                        stt = st.selectbox(
                            f"{evn}",
                            options=st.session_state.qbn["nodes"][evn]["states"],
                            key=f"ev_{evn}",
                        )
                        evidence[evn] = stt
        
                st.divider()
                query_nodes = st.multiselect(
                    textos_inf.get("query_nodes", "Query nodes"),
                    options=nodes,
                    default=nodes[:1] if nodes else [],
                )
                # query nodes
                if not query_nodes and nodes:
                    query_nodes = [nodes[0]]

            # ============================================================
            # EXECUÇÃO (NA PÁGINA PRINCIPAL)
            # ============================================================
            
            st.markdown("---")
            st.subheader(textos_inf["sidebar_execucao"])
            st.caption(textos_inf["exec_desc"])
            
            colA, colB, colC = st.columns(3)
            
            with colA:
                shots = st.number_input(
                    textos_inf["shots"],
                    min_value=100,
                    max_value=200000,
                    value=int(st.session_state.get("qbn_shots", 2000)),
                    step=100,
                    help=textos_inf["shots_help"],
                )
                st.session_state["qbn_shots"] = int(shots)
            
            with colB:
                seed = st.number_input(
                    textos_inf["seed"],
                    min_value=0,
                    max_value=10**9,
                    value=int(st.session_state.get("qbn_seed", 123)),
                    step=1,
                    help=textos_inf["seed_help"],
                )
                st.session_state["qbn_seed"] = int(seed)
            
            with colC:
                topk = st.number_input(
                    textos_inf["topk"],
                    min_value=0,
                    max_value=200,
                    value=int(st.session_state.get("qbn_topk", 20)),
                    step=1,
                    help=textos_inf["topk_help"],
                )
                st.session_state["qbn_topk"] = int(topk)
            
            colD, colE, colF = st.columns(3)
            
            colD = st.columns(1)[0]
            
            with colD:
                annotate = st.checkbox(
                    textos_inf["annotate"],
                    value=bool(st.session_state.get("qbn_annotate", True)),
                    help=textos_inf["annotate_help"]
                )
                st.session_state["qbn_annotate"] = bool(annotate)
            
            # ----------------------------
            # AA (seção separada)
            # ----------------------------
            st.markdown("---")
            st.subheader(textos_inf["aa"])
            st.caption(textos_inf["aa_desc"])
            
            aa_enable = st.checkbox(
                textos_inf["aa_enable"],
                value=bool(st.session_state.get("qbn_aa_enable", True)),
                help=textos_inf["aa_enable_help"],
            )
            st.session_state["qbn_aa_enable"] = bool(aa_enable)
            
            aa_k_manual = st.checkbox(
                textos_inf["aa_k_manual"],
                value=bool(st.session_state.get("qbn_aa_k_manual", False)),
                help=textos_inf["aa_k_manual_help"],
            )
            st.session_state["qbn_aa_k_manual"] = bool(aa_k_manual)
            
            aa_k = None
            if aa_enable:
                if aa_k_manual:
                    aa_k = int(
                        st.number_input(
                            textos_inf["aa_k"],
                            min_value=0,
                            max_value=50,
                            value=int(st.session_state.get("qbn_aa_k", 1)),
                            step=1,
                            help=textos_inf["aa_k_help"],
                        )
                    )
                    st.session_state["qbn_aa_k"] = int(aa_k)
                else:
                    st.caption(textos_inf["aa"])  # texto explicativo (opcional)
                    st.info(textos_inf["aa_tip"])

            
            st.markdown("")
            run = st.button(textos_inf["run"], type="primary")

            
            # ============================
            # Run inference
            # ============================
            bn_ready = (len(st.session_state.qbn["nodes"]) > 0)
            if run and bn_ready:
                # Build BN object
                bn_nodes = st.session_state.qbn["nodes"]
                order = _qbn_topological_order(bn_nodes)
                bn = {"nodes": bn_nodes, "order": order}
        
                # Basic consistency check: all parents exist and CPT rows exist (we allow missing rows but warn)
                consistent = True
                for n in order:
                    info = bn_nodes[n]
                    for p in info.get("parents", []):
                        if p not in bn_nodes:
                            consistent = False
                if not consistent:
                    st.warning(textos_inf["warning_bn_inconsistente"])

                safe_mode = st.checkbox(
                    textos_inf["safe_mode"],
                    value=bool(st.session_state.get("qbn_safe_mode", True)),
                )
                st.session_state["qbn_safe_mode"] = bool(safe_mode)

                
                max_states = 200000 if safe_mode else 2000000
        
                # Exact (when possible)
                exact_post = _qbn_exact_posterior(bn, query_nodes=query_nodes, evidence=evidence, max_states=max_states)
                exact_marg = _qbn_exact_marginals(bn, evidence=evidence, max_states=max_states)
        
                # Monte Carlo (always)
                mc_post = _qbn_mc_likelihood_weighting(bn, query_nodes=query_nodes, evidence=evidence, n_samples=int(shots), seed=int(seed))
                mc_marg = _qbn_mc_marginals_lw(bn, evidence=evidence, n_samples=int(shots), seed=int(seed))
        
                # Quantum shots (binary only + feasible enumeration)
                qshots = _qbn_quantum_shots(bn, query_nodes=query_nodes, evidence=evidence, shots=int(shots), seed=int(seed), max_states=max_states)
        
                qaa = None
                k_used = None
                if aa_enable and qshots is not None and len(evidence) > 0:
                    # compute p_good from joint distribution to pick k (if not manual)
                    jd = _qbn_joint_distribution_enumerate(bn, max_states=max_states)
                    if jd is not None:
                        outs, p = jd
                        req = _qbn_filter_evidence_bitstrings(bn, evidence)
                        good_mask = np.ones(len(outs), dtype=bool)
                        for i, bs in enumerate(outs):
                            ok = True
                            for pos, bit in req.items():
                                if int(bs[pos]) != int(bit):
                                    ok = False
                                    break
                            good_mask[i] = ok
                        p_good = float(p[good_mask].sum())
                        if aa_k_manual:
                            k_used = int(aa_k)
                        else:
                            k_used = _qbn_best_grover_k(p_good, k_max=12)
                        qaa = _qbn_quantum_aa_shots(bn, query_nodes=query_nodes, evidence=evidence, shots=int(shots),
                                                    seed=int(seed), k=int(k_used), max_states=max_states)
        
                st.session_state.qbn["last"] = {
                    "bn": bn,
                    "evidence": evidence,
                    "query_nodes": query_nodes,
                    "exact_post": exact_post,
                    "exact_marg": exact_marg,
                    "mc_post": mc_post,
                    "mc_marg": mc_marg,
                    "qshots": qshots,
                    "qaa": qaa,
                    "k_used": k_used,
                    "shots": int(shots),
                    "seed": int(seed),
                }

            def _qbn_build_stateprep_circuit_for_display(bn: Dict[str, Any], max_states: int, textos_inf: Dict[str, str]):
                # Circuito de preparação de estado (display): suporta nós binários e multiestados.
                # Estratégia:
                # - Cada nó Vi usa mi = ceil(log2(|states(Vi)|)) qubits (encoding por índice do estado).
                # - Nós raiz: StatePreparation(amplitudes) no registrador do nó.
                # - Nós condicionais: StatePreparation controlado pelos qubits dos pais, para cada combinação de estados dos pais.
                from qiskit import QuantumCircuit
                from qiskit.circuit.library import StatePreparation
                import numpy as np
                import itertools

                order = bn.get("order", [])
                nodes = bn.get("nodes", {})
                if not order:
                    return None, textos_inf["circ_sem_nos"]

                acct = _qbn_qubit_accounting(bn)
                bits = acct["node_bits"]
                offs = _qbn_node_offsets(bn)
                q_nodes = int(acct["q_nodes"])

                # segurança: circuito pode explodir para redes grandes e/ou CPTs muito ramificadas (só display)
                if q_nodes > 18:
                    return None, textos_inf["circ_muito_grande"]

                # limite simples por número total de linhas de CPT (somatório produto das cardinalidades dos pais)
                total_rows = 0
                for child in order:
                    parents = nodes[child].get("parents", [])
                    if not parents:
                        continue
                    row_count = 1
                    for p in parents:
                        row_count *= int(len(nodes[p]["states"]))
                    total_rows += int(row_count)
                if total_rows > 512:
                    return None, textos_inf["circ_muito_grande"]

                qc = QuantumCircuit(q_nodes, q_nodes)

                def _amps_from_probs(probs: List[float], m_bits: int) -> List[complex]:
                    """Converte probs (tamanho ni) em amplitudes sqrt(p) (tamanho 2^m_bits), com padding em zeros."""
                    probs = _qbn_normalize_row(list(probs))
                    dim = int(2 ** int(m_bits))
                    amps = [0.0] * dim
                    for i, pi in enumerate(probs):
                        if i >= dim:
                            break
                        amps[i] = float(np.sqrt(max(0.0, float(pi))))

                    # normalização numérica (StatePreparation espera vetor normalizado)
                    norm = float(np.linalg.norm(np.array(amps, dtype=float)))
                    if norm <= 0:
                        amps[0] = 1.0
                        norm = 1.0
                    amps = [complex(a / norm) for a in amps]
                    return amps

                for child in order:
                    info = nodes[child]
                    parents = info.get("parents", [])

                    mi = int(bits[child])
                    t0 = int(offs[child])
                    target_qubits = list(range(t0, t0 + mi))

                    ni = int(len(info["states"]))
                    cpt = info.get("cpt", {}) or {}

                    if len(parents) == 0:
                        # raiz
                        probs = cpt.get((), [1.0 / ni] * ni)
                        amps = _amps_from_probs(probs, mi)
                        gate = StatePreparation(amps)
                        try:
                            gate.name = f"Prep({child})"
                        except Exception:
                            pass
                        qc.append(gate, target_qubits)
                        qc.barrier()
                        continue

                    # condicional: para cada combinação de estados dos pais, aplicar Prep controlado
                    parent_states = [nodes[p]["states"] for p in parents]
                    combos = list(itertools.product(*parent_states))

                    controls: List[int] = []
                    for p in parents:
                        mp = int(bits[p])
                        p0 = int(offs[p])
                        controls.extend(list(range(p0, p0 + mp)))

                    for comb in combos:
                        probs = cpt.get(tuple(comb), [1.0 / ni] * ni)
                        amps = _amps_from_probs(probs, mi)
                        base_gate = StatePreparation(amps)
                        try:
                            base_gate.name = f"Prep({child})"
                        except Exception:
                            pass

                        # ctrl_state em bits (concatenados na mesma ordem dos controls)
                        ctrl_bits = ""
                        for i, p in enumerate(parents):
                            p_states = nodes[p]["states"]
                            idx = int(p_states.index(comb[i]))
                            ctrl_bits += _qbn_int_to_bits(idx, int(bits[p]))
                        ctrl_state_int = int(ctrl_bits, 2) if ctrl_bits else 0

                        cgate = base_gate.control(num_ctrl_qubits=len(controls), ctrl_state=ctrl_state_int)
                        qc.append(cgate, controls + target_qubits)

                    qc.barrier()

                qc.measure(list(range(q_nodes)), list(range(q_nodes)))
                return qc, None

            
            
            def _qbn_build_aa_preview_circuit_for_display(bn: Dict[str, Any], k: int):
                from qiskit import QuantumCircuit
                from qiskit.circuit import Gate
            
                n = len(bn["order"])
                if n <= 0:
                    return None
            
                qc = QuantumCircuit(n, n)
                qc.barrier(label="A")
            
                oracle = Gate(name="Oracle(evidence)", num_qubits=n, params=[])
                diffusion = Gate(name="Diffusion", num_qubits=n, params=[])
            
                for _ in range(int(max(0, k))):
                    qc.append(oracle, list(range(n)))
                    qc.append(diffusion, list(range(n)))
            
                qc.barrier(label="Measure")
                qc.measure(list(range(n)), list(range(n)))
                return qc


            
            # ============================
            # Results
            # ============================
            last = st.session_state.qbn.get("last")
            if last:
                bn = last["bn"]
                nodes_order = bn["order"]
        
                st.success(textos_inf["circuito_ok"])

                # ---- Circuit plot (QBN)
                with st.expander(textos_inf["ver_circuito"], expanded=False):
                    safe_mode_now = bool(st.session_state.get("qbn_safe_mode", True))
                    max_states_now = 200000 if safe_mode_now else 2000000
                    
                    qc, err = _qbn_build_stateprep_circuit_for_display(
                        bn,
                        max_states=max_states_now,
                        textos_inf=textos_inf
                    )
                                        
                    if qc is None:
                        st.info(err if err else textos_inf["circ_indisp"])
                    else:
                        st.caption(textos_inf["circ_desc_stateprep"])
                        
                        show_decomposed = st.checkbox(
                            textos_inf["circ_decomp"],
                            value=True,
                            key="qbn_show_decomposed"
                        )
                        
                        # só mostra o nível se estiver decomposto
                        if show_decomposed:
                            reps = st.slider(
                                textos_inf["circ_decomp_level"],
                                1, 6, 3,
                                key="qbn_decomp_reps"
                            )
                            force_basis = st.checkbox(
                                textos_inf["circ_force_basis"],
                                value=False,
                                key="qbn_force_basis"
                            )
                        else:
                            reps = 0
                            force_basis = False
                        
                        qc_show = qc
                        if show_decomposed:
                            qc_show = qc_show.decompose(reps=int(reps))
                            if force_basis:
                                try:
                                    from qiskit import transpile
                                    qc_show = transpile(qc_show, basis_gates=["u", "cx"], optimization_level=1)
                                except Exception:
                                    pass
                        
                        try:
                            fig = qc_show.draw(output="mpl")
                            st.pyplot(fig)
                        except Exception:
                            st.code(qc_show.draw(output="text"))



                    acct = _qbn_qubit_accounting(bn)

                    st.subheader(textos_inf["q_header"])
                    c1, c2, c3 = st.columns(3)
                    c1.metric(textos_inf["q_total"], acct["q_total"])
                    c2.metric(textos_inf["q_nodes"], acct["q_nodes"])
                    c3.metric(textos_inf["q_anc"], acct["q_anc"])
                    
                    st.caption(textos_inf["q_caption"])
                    
                    import pandas as pd
                    df_map = pd.DataFrame(acct["mapping"])
                    
                    df_map["role"] = df_map["role"].map({
                        "node": textos_inf["q_role_node"],
                        "ancilla": textos_inf["q_role_ancilla"],
                    })
                    
                    df_map = df_map.rename(columns={
                        "qubit": textos_inf["q_col_qubit"],
                        "role": textos_inf["q_col_role"],
                        "node": textos_inf["q_col_node"],
                        "slot": textos_inf["q_col_slot"],
                    })
                    
                    st.dataframe(df_map, use_container_width=True, hide_index=True)

                
                # ---- Results table (single): node/state probabilities by method (%) + CI95 where applicable
                exact_marg = last.get("exact_marg")
                mc_marg = last.get("mc_marg")
                qshots = last.get("qshots")
                qaa = last.get("qaa")
                
                def _get_marg(marg, node, state):
                    if marg is None:
                        return None
                    return float(marg.get(node, {}).get(state, 0.0))
                
                def _ci95_pct(p: float, n_eff: int):
                    # CI95 for a Bernoulli indicator of "being in this state" (per-state CI).
                    if (n_eff is None) or (n_eff <= 1):
                        return ""
                    p = float(max(0.0, min(1.0, p)))
                    se = math.sqrt(max(1e-12, p * (1.0 - p) / float(n_eff)))
                    lo = max(0.0, p - 1.96 * se)
                    hi = min(1.0, p + 1.96 * se)
                    return f"[{100.0*lo:.1f}, {100.0*hi:.1f}]"
                
                n_eff_mc = int(last.get("shots", 0))
                n_eff_qshots = int(qshots.get("accepted", 0)) if qshots else 0
                n_eff_qaa = int(qaa.get("accepted", 0)) if qaa else 0
                
                rows = []
                for n in nodes_order:
                    for s in bn["nodes"][n]["states"]:
                        p_exact = _get_marg(exact_marg, n, s) if exact_marg is not None else None
                        p_mc = _get_marg(mc_marg, n, s) if mc_marg is not None else None
                        p_qs = _get_marg(qshots["marg"], n, s) if qshots is not None else None
                        p_qa = _get_marg(qaa["marg"], n, s) if qaa is not None else None
                
                        rows.append({
                            textos_inf["col_node"]: n,
                            textos_inf["col_state"]: s,
                
                            textos_inf["col_exact_pct"]: (100.0 * p_exact) if (p_exact is not None) else None,
                
                            textos_inf["col_mc_pct"]: (100.0 * p_mc) if (p_mc is not None) else None,
                            textos_inf["col_mc_ci95"]: _ci95_pct(p_mc, n_eff_mc) if (p_mc is not None) else "",
                
                            textos_inf["col_qshots_pct"]: (100.0 * p_qs) if (p_qs is not None) else None,
                            textos_inf["col_qshots_ci95"]: _ci95_pct(p_qs, n_eff_qshots) if (p_qs is not None) else "",
                
                            textos_inf["col_qaa_pct"]: (100.0 * p_qa) if (p_qa is not None) else None,
                            textos_inf["col_qaa_ci95"]: _ci95_pct(p_qa, n_eff_qaa) if (p_qa is not None) else "",
                        })
                
                df_results = pd.DataFrame(rows)
                
                st.subheader(textos_inf["tabela_resultados"])
                st.dataframe(df_results, use_container_width=True)

        
                # ---- Charts: quantum outcomes (top-k)
                def _plot_outcomes(counts: Dict[str, int], title: str):
                    if not counts:
                        st.info(title + " — (no accepted shots)")
                        return
                    total = sum(counts.values())
                    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
                    if int(topk) > 0:
                        items = items[: int(topk)]
                    labels = [_qbn_outcome_label(k) for k, _ in items]

                    # remover notação de Dirac (|...⟩ / |...>)
                    labels = [
                        str(l).strip().replace("|", "").replace("⟩", "").replace(">", "")
                        for l in labels
                    ]
                    
                    vals = [v / total * 100 for _, v in items]
                    fig, ax = plt.subplots(figsize=(6.0, 3.0), dpi=120)


                    ax.bar(range(len(labels)), vals)
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels, rotation=90, ha = "center")
                    ax.set_ylabel("Probability (%)")
                    ax.set_title(title)
                    if annotate:
                        for i, v in enumerate(vals):
                            ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
                    fig.tight_layout()
                    st.pyplot(fig)
                

                
                
                plots = st.checkbox(
                    textos_inf["plots"],
                    value=bool(st.session_state.get("qbn_plots", True)),
                )
                st.session_state["qbn_plots"] = bool(plots)

                
                if plots:
                    st.subheader(textos_inf["graficos"])
                    c1, c2 = st.columns(2)

                    with c1:
                        if last.get("qshots") is not None:
                            st.markdown(f"**{textos_inf['outcomes_qshots']}** (accepted={last['qshots']['accepted']}, acc_rate={last['qshots']['acc_rate']:.3f})")
                            _plot_outcomes(last["qshots"]["counts"], textos_inf["outcomes_qshots"])
                        
                    with c2:
                        if last.get("qaa") is not None:
                            k_used = last.get("k_used")
                            st.markdown(f"**{textos_inf['outcomes_qaa']}** (k={k_used}, accepted={last['qaa']['accepted']}, acc_rate={last['qaa']['acc_rate']:.3f})")
                            _plot_outcomes(last["qaa"]["counts"], textos_inf["outcomes_qaa"])
        
        # Render QBN inference page
        pagina_inferencia_qbn(textos, textos_inf)
        
        with st.sidebar:
            if st.button(textos["ini"]):
                st.session_state['pagina'] = 'inicio'
                st.rerun()

    
    elif st.session_state['pagina'] == 'info':
        st.subheader(textos["pagina_info2"])
        mostrar_cartoes_de_info(textos)

        with st.sidebar:
            if st.button(textos["ini"]):
                st.session_state['pagina'] = 'inicio'
    
    elif st.session_state['pagina'] == 'otimizacao_info':
        st.title(textos.get("pagina_referencias_titulo", "Referências"))
        st.header(textos_otim["pagina_otimizacao"])

    elif st.session_state['pagina'] == 'ml_info':
        st.title(textos_ml["info1_titulo"])
        st.write(textos_ml["info1"])

        st.subheader(textos_ml["info2_titulo"])
        st.write(textos_ml["info2"])
        st.write(textos_ml["info2.1"])
        st.write(textos_ml["info2.2"])

        st.subheader(textos_ml["info3_titulo"])
        st.write(textos_ml["info3"])

        st.subheader(textos_ml["info4_titulo"])
        st.write(textos_ml["info4"])

        st.subheader(textos_ml["info5_titulo"])
        st.write(textos_ml["info5"])
        st.write(textos_ml["info5.1"])

        st.subheader(textos_ml["info6_titulo"])
        st.write(textos_ml["info6"])
        st.write(textos_ml["info6.1"])

        st.subheader(textos_ml["info7_titulo"])
        st.write(textos_ml["info7"])
        st.write(textos_ml["info7.1"])
        
    elif st.session_state['pagina'] == 'inferencia_info':
        st.subheader(textos["pagina_info"])

    elif st.session_state['pagina'] == 'ref':
        st.subheader(textos["pagina_referencias"])
        st.write(textos["ref"])
        mostrar_ref(textos)

        with st.sidebar:
            if st.button(textos["ini"]):
                st.session_state['pagina'] = 'inicio'
                st.rerun()
                
    elif st.session_state['pagina'] == 'otim_ref':
        mostrar_referencias(textos, textos_otim)

        with st.sidebar:
            if st.button(textos["ini"]):
                st.session_state['pagina'] = 'inicio'
                st.rerun()
                
    elif st.session_state['pagina'] == 'ml_ref':
        st.subheader(textos["pagina_referencias"])
        st.write(oie)

        with st.sidebar:
            if st.button(textos["ini"]):
                st.session_state['pagina'] = 'inicio'
                st.rerun()

    elif st.session_state['pagina'] == 'inf_ref':
        st.subheader(textos["pagina_referencias"])

        with st.sidebar:
            if st.button(textos["ini"]):
                st.session_state['pagina'] = 'inicio'
                st.rerun()

if __name__ == "__main__":
    main()






















































































































































































































































