# CP5 — Machine Learning: agrupamento de vinhos

Notebook do Checkpoint 5 da FIAP, focado na **Parte 2 — Machine Learning & Modelling**.
Agrupa vinhos com K-means e usa estatística descritiva para apoiar a interpretação.

## Arquivos

- [`CP5_Machine_Learning_Wines.ipynb`](CP5_Machine_Learning_Wines.ipynb): análise comentada com gráficos, métricas, resultados e conclusões; células executadas.
- [`wines.csv`](wines.csv): 178 amostras, 13 características químicas e a classe `Wine`, excluída das entradas do modelo.
- [`requirements.txt`](requirements.txt): versões utilizadas na execução.
- [`tests/test_notebook.py`](tests/test_notebook.py): execução em kernel novo e validações da escala, métricas, escolha de K e perfis.

## Conteúdo do notebook

1. Problema e proposta de agrupamento não supervisionado.
2. Origem e significado das variáveis; verificação de nulos, duplicatas e tipos.
3. Estatística descritiva, histogramas, boxplots e análise de extremos.
4. Padronização das 13 características com `StandardScaler`.
5. Ajustes de K-means para K=1 a K=10; Elbow e Silhouette Score.
6. Modelo final, silhuetas individuais e comparação dos grupos nas escalas originais.
7. Interpretações de álcool, ácido málico e outras características, com limitações.

Na execução salva, o maior Silhouette foi **0,2849**, em **K=3**, com grupos de **65, 51 e 62 vinhos**.
O Elbow também indica uma mudança de inclinação nessa região. Há sobreposição entre os perfis;
o resultado não deve ser interpretado como separação perfeita.

`Wine` não participa do ajuste nem da escolha de K. Os identificadores dos grupos não
representam qualidade ou uma ordem entre vinhos. Ácido málico é uma medida relacionada à
acidez; não permite concluir diretamente sobre acidez total, pH ou percepção sensorial.

## Executar no computador

Ambiente de referência: **Python 3.12**. Abra um terminal na pasta deste repositório.
No Windows/PowerShell, não é necessário ativar o ambiente:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name cp5-wines --display-name "Python (CP5 Wines)"
```

Abra o `.ipynb` em um editor com suporte a notebooks (por exemplo VS Code com a extensão
Jupyter), selecione o kernel **Python (CP5 Wines)** e execute todas as células em ordem.
Mantenha `wines.csv` na pasta do notebook e use essa pasta como diretório de trabalho.
As saídas já salvas também podem ser consultadas sem executar a análise.

Se preferir JupyterLab:

```powershell
.\.venv\Scripts\python.exe -m pip install jupyterlab
.\.venv\Scripts\python.exe -m jupyter lab
```

Para verificar a reprodução e a consistência dos resultados:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Os testes usam o Python que os executa para iniciar um kernel novo e não precisam
de um kernelspec global. O notebook lê apenas o CSV local; não instala pacotes nem faz downloads.

## Fonte e integridade dos dados

Fonte indicada no enunciado: [GitHub Gist — Wine Dataset](https://gist.github.com/tijptjik/9408623).
O arquivo da fonte é `wine.csv`, salvo aqui como `wines.csv`, sem alteração dos bytes.

- Revisão: `b237fa5848349a14a14e5d4107dc7897c21951f5`.
- SHA256: `f295f06d18e62351ee3c0ea3b508558d07e1d72d7fb5bb848fe066d7c1b9884e`.
- Origem e dicionário: [UCI — Wine](https://archive.ics.uci.edu/dataset/109/wine).
- Referência didática principal: `Clusterizacao_I.ipynb`, Aula 08 de 01/10/2026, fornecida pelo aluno em `Sem2.zip`.

## Escopo da entrega

Este trabalho cobre a Parte 2 e a estatística necessária para apoiá-la. A análise
probabilística e as tabelas de frequência exigidas pela Parte 1 não estão incluídas.
Para a entrega acadêmica, ainda é necessário o PDF com nomes e RMs dos integrantes
e a apresentação oral, conforme o enunciado. Esses dados pessoais não foram fornecidos.
