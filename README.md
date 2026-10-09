# Simulador Didático de Perceptron Multicamadas (PMC) & Backpropagation

Este projeto consiste em uma aplicação web interativa desenvolvida em HTML5, CSS3 puro (Glassmorphic Dark UI) e JavaScript Vanilla. Ela implementa e visualiza detalhadamente o funcionamento de uma **Rede Neural Artificial Perceptron Multicamadas (PMC)** treinada via **Algoritmo de Retropropagação de Erro (Backpropagation)**, baseando-se estritamente na teoria e nos números de exemplo do **Capítulo 5 (Figura 5.5)** do livro *"Redes Neurais Artificiais para Engenharia e Ciências Aplicadas"* (Silva, Spatti & Flauzino).

---

## 📌 Visão Geral e Recursos

1. **Topologia Exata do Livro (2 ➔ 3 ➔ 2 ➔ 1)**:
   - **Camada de Entrada**: 2 entradas ($x_1, x_2$) + Entrada fixa de bias ($x_0 = -1$).
   - **1ª Camada Oculta**: 3 neurônios ($H_1^{(1)}, H_2^{(1)}, H_3^{(1)}$) com matriz de pesos $W^{(1)}$ de dimensão $3 \times 3$.
   - **2ª Camada Oculta**: 2 neurônios ($H_1^{(2)}, H_2^{(2)}$) com matriz de pesos $W^{(2)}$ de dimensão $2 \times 4$.
   - **Camada de Saída**: 1 neurônio ($Y_1^{(3)}$) com matriz de pesos $W^{(3)}$ de dimensão $1 \times 3$.

2. **Fases Interativas Didáticas**:
   - **Fase Forward (Propagação Adiante)**: Cálculo passo a passo dos somatórios ponderados ($I$) e saídas ativadas ($Y$).
   - **Fase Backward (Retropropagação)**: Cálculo dos gradientes locais ($\delta^{(3)}, \delta^{(2)}, \delta^{(1)}$) e atualização dos pesos sinápticos ($\Delta W$).
   - **Execução por Épocas ou Loop Contínuo**: Opções para avançar passo a passo, rodar 1 época inteira, rodar em lotes de 50 épocas ou iniciar o treinamento contínuo com pausa automática ao atingir a convergência.

3. **Interface e Visualização Avançada**:
   - **Renderização via Canvas HTML5**: Exibição da rede em tempo real com partículas animadas indicando a direção do sinal (verde no Forward, rosa/vermelho no Backward).
   - **Raio-X Matemático em Tempo Real**: Painel com fórmulas e substituição numérica direta mostrando as equações exatas do capítulo.
   - **Inspeção por Passagem de Mouse (Hover Tooltip)**: Passe o ponteiro sobre qualquer nó ou sinapse para examinar seus valores internos de entrada ponderada ($I$), ativação ($Y$), gradiente local ($\delta$) ou peso ($W$).
   - **Gráfico Histórico de Erro**: Acompanhamento gráfico da descida do gradiente e da evolução do erro quadrático $E = \frac{1}{2}(d - y)^2$.

---

## 📐 Fundamentação Matemática (Conforme Capítulo 5)

### 1. Fase Forward

Para cada camada $k$, o valor de entrada ponderada de cada neurônio $j$ é dado por:

$$I_j^{(k)} = \sum_{i} W_{j,i}^{(k)} \cdot Y_i^{(k-1)}$$

Onde $Y_0^{(k-1)} = -1.0$ representa a entrada de bias (limiar de ativação).

A saída ativada do neurônio é calculada por:

$$Y_j^{(k)} = g(I_j^{(k)})$$

Por padrão, a função de ativação utilizada no exemplo do livro é a **Tangente Hiperbólica**:

$$g(I) = \tanh(I), \quad g'(I) = 1 - \tanh^2(I) = 1 - Y^2$$

O simulador também permite alternar para a **Sigmoide Logística**:

$$g(I) = \frac{1}{1 + e^{-I}}, \quad g'(I) = Y(1 - Y)$$

### 2. Erro da Rede

- **Erro Linear**:
  $$e = d - Y_1^{(3)}$$
- **Erro Quadrático Instantâneo**:
  $$E = \frac{1}{2} (d - Y_1^{(3)})^2$$

### 3. Fase Backward (Retropropagação)

1. **Camada de Saída ($k = 3$)** *(Equação 5.15)*:
   $$\delta_1^{(3)} = e \cdot g'(I_1^{(3)})$$
   $$\Delta W_{1,i}^{(3)} = \eta \cdot \delta_1^{(3)} \cdot Y_i^{(2)} + \alpha \cdot \Delta W_{1,i}^{(3)}(t-1)$$

2. **2ª Camada Oculta ($k = 2$)** *(Equação 5.26)*:
   $$\delta_j^{(2)} = g'(I_j^{(2)}) \cdot \left[ \delta_1^{(3)} \cdot W_{1,j}^{(3)} \right]$$
   $$\Delta W_{j,i}^{(2)} = \eta \cdot \delta_j^{(2)} \cdot Y_i^{(1)} + \alpha \cdot \Delta W_{j,i}^{(2)}(t-1)$$

3. **1ª Camada Oculta ($k = 1$)** *(Equação 5.37)*:
   $$\delta_j^{(1)} = g'(I_j^{(1)}) \cdot \sum_{k} \left[ \delta_k^{(2)} \cdot W_{k,j}^{(2)} \right]$$
   $$\Delta W_{j,i}^{(1)} = \eta \cdot \delta_j^{(1)} \cdot x_i + \alpha \cdot \Delta W_{j,i}^{(1)}(t-1)$$

4. **Atualização dos Pesos**:
   $$W_{j,i}^{(k)}(t+1) = W_{j,i}^{(k)}(t) + \Delta W_{j,i}^{(k)}$$

### 4. Critérios de Parada (Equação 5.40)

O treinamento é interrompido quando qualquer um dos seguintes critérios for atendido:
- $E = \frac{1}{2}(d - y)^2 \le \varepsilon$ (Erro abaixo da precisão tolerada).
- $|E(t) - E(t-1)| \le \varepsilon$ (Variação do erro estagnada entre épocas).
- $t \ge t_{\text{máx}}$ (Número máximo de épocas atingido).

---

## 🔢 Configuração Inicial Padrão (Exemplo Figura 5.5)

Ao clicar no botão **"Restaurar Livro"** ou selecionar o preset **"★ Livro (Exemplo Fig. 5.5)"**, o simulador carrega os parâmetros exatos apresentados pelo autor:

- **Entradas**: $x_1 = 0.30$, $x_2 = 0.70$
- **Alvo Desejado**: $d = 0.85$
- **Taxa de Aprendizado**: $\eta = 0.30$
- **Termo de Momento**: $\alpha = 0.00$
- **Matriz de Pesos Inicial $W^{(1)}$ (3x3)**:
  $$\begin{bmatrix} 0.2 & 0.4 & 0.5 \\ 0.3 & 0.6 & 0.7 \\ 0.4 & 0.8 & 0.3 \end{bmatrix}$$
- **Matriz de Pesos Inicial $W^{(2)}$ (2x4)**:
  $$\begin{bmatrix} -0.7 & 0.6 & 0.2 & 0.7 \\ -0.3 & 0.7 & 0.2 & 0.8 \end{bmatrix}$$
- **Matriz de Pesos Inicial $W^{(3)}$ (1x3)**:
  $$\begin{bmatrix} 0.1 & 0.8 & 0.5 \end{bmatrix}$$

---

## 🚀 Como Executar

Por ser uma aplicação totalmente client-side (HTML/JS/CSS puros em arquivo único), não requer instalação de dependências ou servidores Node.js.

1. Abra o arquivo [`index.html`](file:///home/lazaro/20261/ia/PerceptronMultiCamadas/index.html) em qualquer navegador moderno (Chrome, Firefox, Edge, Safari).
2. Interaja com os botões de controle:
   - **1. Forward ➔**: Executa somente a propagação dos sinais.
   - **2. Backward ⬅**: Calcula a retropropagação dos erros e ajusta os pesos sinápticos.
   - **Executar 1 Ciclo Completo**: Realiza uma época completa (Forward + Backward).
   - **▶ Iniciar Treino**: Executa épocas continuamente até a convergência.
