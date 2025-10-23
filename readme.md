# Prova – Regressão Logística (Voice Gender)
# Alunos: Pedro Chaves (RM553988), Iago Diniz (RM553776), Enzzo Monteiro (RM552616) e Lucas Garcia (RM554070)

**Objetivo:** classificar **voz masculina/feminina** com regressão logística, passando por **EDA**, **padronização**, **validação cruzada com GridSearchCV (L1/L2)**, **modelo final**, e — na Parte B — treinar uma variação **usando RMSE como perda** e **comparar** com o modelo padrão (log-loss).

**Dataset:** *Gender Recognition by Voice and Speech Analysis* (`voice.csv`).  
**Arquivo usado:** cópia **local** (ex.: `C:\Users\pedro\Downloads\IA-CP2\voice.csv`).  
**Alvo:** `label` → mapeado para `0 = female`, `1 = male`.  
**Features:** todas as colunas numéricas (medidas acústicas).

---

## Parte A — Análise Exploratória, Validação Cruzada, Lasso e Ridge

### 1) Preparação e Análise Exploratória
- **Carregamento:** leitura do `voice.csv` local; criação de **X** (features) e **y** (alvo).  
- **Balanceamento do alvo:** contagens e percentuais por classe → dataset **bem balanceado (~50/50)**.  
- **Distribuições:** **histogramas + KDE** por feature (assimetria, dispersão, possíveis outliers).  
- **Correlação e multicolinearidade:** **matriz de correlação** e “Top pares” por \|corr\| → há **pares altamente correlacionados** (comportamento esperado em medidas acústicas).  
- **Escalas diferentes:** estatísticas (mean/std/min/max/range) mostram **amplitudes bem distintas** → **padronização (StandardScaler)** **necessária** para:
  - estabilizar a otimização;
  - tornar a regularização **justa** entre variáveis;
  - melhorar leitura das magnitudes dos coeficientes.

**Conclusão da EDA:** seguir com **Pipeline(StandardScaler → LogisticRegression)** e validação cruzada, testando **L1 (Lasso)** e **L2 (Ridge)** com diferentes **C**.

---

### 2) Grid Search com Validação Cruzada (k=5)
- **Pipeline:** `StandardScaler()` → `LogisticRegression(solver="liblinear", max_iter=2000)`.  
- **Grid:** `penalty ∈ {L1, L2}` e `C ∈ {0.01, 0.1, 1, 10, 100}`.  
- **CV:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.  
- **Métrica:** **accuracy** média de validação.

**Resultado (substitua pelos seus números):**  
- **Melhores parâmetros:** `{'logreg__penalty': <L1_ou_L2>, 'logreg__C': <VALOR_C>}`  
- **Acurácia média (5-fold):** **<BEST_MEAN_VAL_ACC>**

**Leitura rápida:**  
- **L1** tende a **zerar** coeficientes → seleção de variáveis e modelo mais simples.  
- **L2** **encolhe** coeficientes (raramente zera) → mais **estável** com multicolinearidade.  

---

### 3) Modelo Final (Treino/Teste)
- **Split:** `train/test = 80/20` **estratificado** (`random_state=42`).  
- **Reajuste:** pipeline com **melhores hiperparâmetros** **apenas no treino**.  
- **Métricas no teste:**  
  - **Acurácia:** **<ACC_TESTE>**  
  - **Precisão (precision):** **<PREC_TESTE>**  
  - **Recall (sensibilidade):** **<RECALL_TESTE>**  
  - **F1-score:** **<F1_TESTE>**  
  - **Matriz de confusão** (`[[TN FP], [FN TP]]`): **<MATRIZ_CONFUSAO>**

**Dicas de leitura:** precisão foca em **FP**, recall foca em **FN**, e **F1** equilibra ambos.

---

### 4) Discussão — L1 (Lasso) vs L2 (Ridge)
- **Coeficientes (padronizados):**
  - **L1 (Lasso):** força **esparsidade** → alguns coeficientes **= 0** (seleção automática).  
  - **L2 (Ridge):** **encolhe** coeficientes, mas **mantém** todas as variáveis.  
- **Variáveis eliminadas pelo L1:** **<LISTA_DE_FEATURES_ZERADAS_POR_L1_ou_"Nenhuma">**  
- **Efeitos práticos:**  
  - **L1**: melhor **interpretabilidade** (modelo mais simples) e possível redução de overfitting.  
  - **L2**: melhor com **multicolinearidade**, mais **estável**.  
  - Em muitos casos, **desempenho similar**; escolha guiada pelo **GridSearch** e necessidade de **simplicidade vs. estabilidade**.

---

## Parte B — Alterando a Função de Perda

### 1) Experimento — “Logística” treinada por **RMSE** (vs. log-loss)
- **Implementação didática:** modelo com saída **sigmoide** `p = σ(wᵀx + b)`, **treinado via MSE/RMSE** entre `p` e `y ∈ {0,1}` (gradiente descendente).  
- **Pré-processamento:** **StandardScaler**.  
- **Métricas no teste:** **Acurácia** e **Curva ROC (AUC)**.

**Resultado (substitua pelos seus números):**  
- **Acurácia (RMSE):** **<ACC_RMSE>**  
- **AUC (RMSE):** **<AUC_RMSE>**

> **Nota:** isto é **experimental**; a regressão logística **padrão** minimiza **log-loss** (MLE), geralmente oferecendo **probabilidades melhor calibradas**.

---

### 2) Comparação — **RMSE vs Log-loss** + **Calibração**
- **Baseline padrão:** `LogisticRegression` (**log-loss**, L2, `C=1`).  
- **Comparação no teste:**  
  - **Acurácia (log-loss):** **<ACC_LOGLOSS>**  
  - **AUC (log-loss):** **<AUC_LOGLOSS>**
- **Curva de calibração (reliability curve):**  
  - **Log-loss:** tende a ficar **mais próximo** da diagonal (probabilidades **bem calibradas**).  
  - **RMSE:** costuma desviar mais (probabilidades **menos calibradas**, com **sub/superconfiança**).

---

### 3) Discussão Crítica — Por que **RMSE é uma má ideia** aqui?
- **Probabilidades vs. valores contínuos:** regressão logística modela **probabilidades** (Bernoulli) e o critério natural é a **log-loss** (máxima verossimilhança).  
- **Penalização de erros extremos:** **RMSE** **não pune o bastante** previsões **de alta confiança** erradas; a **log-loss** cresce muito nesses casos, **corrigindo overconfidence**.  
- **Otimização:** com RMSE, o gradiente inclui `p(1−p)` e **apaga nas extremidades** (0/1), dificultando corrigir previsões ruins; na **log-loss**, o gradiente é **mais informativo** para correção.  
- **Conclusão:** ao trocar **log-loss por RMSE**, a **interpretação** (em **log-odds**, MLE) **se perde**, e a **qualidade das probabilidades** tende a **piorar** (**calibração** mais fraca), mesmo que a **acurácia** às vezes pareça parecida.

---

## Notas de Reprodutibilidade
- **Divisão dos dados:** `train/test = 80/20`, **estratificado**.  
- **Padronização:** `StandardScaler()` **sempre antes** da regressão logística.  
- **Busca de hiperparâmetros:** `GridSearchCV` com `k=5`, **métrica = accuracy**.  
- **Substitua os placeholders** (`<...>`) pelos valores **observados no seu run** (melhores hiperparâmetros, métricas e listas de features zeradas pelo L1).
