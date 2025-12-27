
# 🤝 Guia de Contribuição para PETR4.SA Prediction System

Bem-vindo\! Estamos muito felizes com seu interesse em contribuir para o sistema de predição PETR4.SA. Contribuições em qualquer forma — sejam relatórios de bugs, sugestões de recursos ou código — são sempre bem-vindas.

-----

## 🗺️ Como Posso Contribuir?

Seu projeto tem múltiplas camadas. Você pode ajudar em qualquer uma delas:

### 1\. 🐛 Relatório de Bugs

Problemas na coleta de dados, falhas na API ou erros no cálculo das métricas.

### 2\. 💡 Sugestão de Recursos

Proponha novas funcionalidades (e.g., prever outras ações, adicionar MLflow, etc.).

### 3\. 🧠 Melhorias no Modelo ML

Refine o treinamento, otimize a arquitetura da LSTM, ou ajuste o pré-processamento.

### 4\. 🚀 Melhorias na API/Infraestrutura

Otimize os endpoints do FastAPI, a estrutura do Docker, ou a lógica de caching.

-----

## 💻 Configuração do Ambiente de Desenvolvimento

Para começar a codificar, siga estes passos para configurar seu ambiente. **O uso do Poetry é obrigatório para manter a consistência das dependências.**

1.  **Clone o Repositório:**
    ```bash
    git clone https://github.com/ghpascon/tech_challenge_fase4.git
    cd tech_challenge_fase4
    ```
2.  **Instale as Dependências (via Poetry):**
    ```bash
    poetry install
    poetry shell
    ```
3.  **Execute a Aplicação:**
    ```bash
    python main.py
    ```

-----

## 🚀 Fluxo de Contribuição de Código (Pull Requests)

Siga este fluxo de trabalho para garantir que sua contribuição seja revisada e integrada rapidamente:

1.  **Faça um Fork** do repositório para sua conta.
2.  **Crie um Branch** descritivo para suas alterações. Use um prefixo baseado no tipo de contribuição:
      * `feat/`: Para novas funcionalidades.
      * `fix/`: Para correções de bugs.
      * `refactor/`: Para reestruturações de código sem mudança de comportamento.
      * `docs/`: Para alterações na documentação.
    <!-- end list -->
    ```bash
    git checkout -b fix/problema-no-scaler
    ```
3.  **Faça suas Alterações.**
4.  **Escreva Testes.** Todas as novas funcionalidades ou correções de bugs devem vir acompanhadas de testes unitários ou de integração que comprovem o comportamento esperado. Os testes estão localizados na pasta `tests/`.
5.  **Garanta a Qualidade do Código.**
      * Execute o linter/formatter (se configurado no projeto).
      * Garanta que não há erros de tipagem.
6.  **Crie Commits Atômicos e Descritivos.**
      * *Exemplo ruim:* `git commit -m "fiz umas mudancas"`
      * *Exemplo bom:* `git commit -m "fix(predictor): Corrige desnormalização do scaler"`
7.  **Abra um Pull Request (PR):**
      * Preencha o **Template de PR**.
      * Descreva **o quê** foi feito e **por que** foi feito.
      * Mencione a Issue que o PR resolve (e.g., `Fecha #123`).

-----

## 🧠 Guia Específico para Contribuições de Machine Learning

A contribuição para o modelo LSTM é a mais sensível. Exigimos as seguintes verificações:

### 1\. Otimização do Modelo (`model_pipeline/lstm.ipynb`)

Se você alterar a arquitetura (e.g., número de camadas LSTM, *dropout*, *epochs*):

  * **Documente a Mudança:** Explique a justificativa técnica para a alteração (e.g., "Adicionado Dropout para reduzir Overfitting").
  * **Compare Métricas:** Garanta que o novo modelo tenha **pelo menos** a mesma performance (ou melhor) que o modelo atual, usando métricas como **RMSE** e **MSE**.
      * *Sugestão:* Se o projeto usar **MLflow**, registre os resultados do experimento.
  * **Atualize Artefatos:** Se o treinamento for bem-sucedido, substitua os arquivos `best_model.pth` e `scaler.joblib` na pasta `model_pipeline/artifacts/` e **inclua-os no PR**.

### 2\. Pré-processamento e Sequenciamento

  * **Imutabilidade:** Mantenha a sequência de 10 dias (`TIME_STEP`) e o uso do `MinMaxScaler` na faixa de 0 a 1, a menos que haja uma justificativa de *Série Temporal* muito forte.
  * **Reversibilidade:** Certifique-se de que o novo pré-processamento/normalização é **reversível**, pois a API depende da desnormalização correta para retornar o preço real.

### 3\. Integração na API (`app/services/ml/petr4_predictor.py`)

  * **Assinatura:** Mantenha a assinatura de métodos essenciais no preditor, garantindo que a API do FastAPI possa continuar a chamar o modelo sem alterações estruturais.

-----

## 💾 Estrutura do Projeto para Navegação

| Caminho | Foco da Contribuição |
| :--- | :--- |
| `app/routers/api/pred.py` | Lógica de **Endpoints** e tratamento de *requests* da API. |
| `app/services/ml/*.py` | Lógica de **Carregamento e Predição** do modelo LSTM. |
| `model_pipeline/lstm.ipynb` | **Treinamento, Arquitetura e Otimização** do modelo LSTM. |
| `Dockerfile` & `docker-compose.yml` | **Containerização** (melhorias de *build*, otimização de imagem). |
| `pyproject.toml` | **Gerenciamento de Dependências** (adição ou atualização). |

-----

## ❓ Precisa de Ajuda?

Se você está tendo problemas para configurar o ambiente, treinar o modelo ou entender alguma parte da lógica, sinta-se à vontade para:

  * Abrir uma **Issue** na categoria "Ajuda/Dúvida".

Agradecemos sua contribuição\!