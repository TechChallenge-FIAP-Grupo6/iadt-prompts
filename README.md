# iadt-perplexidade

Este repositório apresenta um notebook Jupyter que demonstra como calcular a perplexidade de prompts de texto usando o modelo GPT-2 (Hugging Face Transformers). O objetivo é comparar o grau de "surpresa" (incerteza) do modelo ao processar diferentes prompts.

## Estrutura do repositório
- `iadt_perplexidade.ipynb` — Notebook principal com o código e exemplos.
- (Opcional) `requirements.txt` — lista de dependências (recomendado).
- (Opcional) `compute_perplexity.py` — script auxiliar para calcular perplexidade em lote (se presente).
- `README.md` — este arquivo.

> Observação: se o repositório contiver outros arquivos, adicione aqui uma linha descrevendo cada um.

## Descrição do notebook
O notebook:
- Importa as bibliotecas necessárias (`torch` e `transformers`).
- Carrega o tokenizador e o modelo GPT-2 pré-treinado.
- Implementa `calcular_perplexidade(prompt)` que:
  - Tokeniza o texto,
  - Passa os tensores ao modelo com `labels=input_ids` para obter a loss,
  - Calcula a perplexidade como `exp(loss)`.
- Compara dois prompts de exemplo e interpreta os resultados.

## Requisitos / Instalação
Recomenda-se criar um ambiente virtual (venv ou conda). Pacotes principais:

- Python 3.8+
- torch
- transformers

Exemplo usando pip:
```bash
python -m venv .venv
source .venv/bin/activate    # ou .venv\Scripts\activate no Windows
pip install --upgrade pip
pip install -r requirements.txt
```

Exemplo (caso não haja requirements.txt):
```bash
pip install torch transformers
```

Observações:
- A instalação do `torch` pode variar conforme sistema e suporte a CUDA. Consulte https://pytorch.org/ para instruções de instalação específicas.
- O modelo GPT-2 será baixado automaticamente da Hugging Face na primeira execução.

## Como executar

Opção 1 — Local (Jupyter Notebook):
1. Ative o ambiente virtual.
2. Instale dependências.
3. Abra o Jupyter: `jupyter notebook`.
4. Abra `iadt_perplexidade.ipynb` e execute as células.

Opção 2 — Google Colab:
- Abra o notebook no Colab (o notebook possui link "Abrir no Colab").
- Caso necessário, instale versões específicas de `transformers` e `torch` via pip no Colab.

## Função principal
Exemplo da função usada no notebook:
```python
def calcular_perplexidade(prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model(**inputs, labels=inputs['input_ids'])
    loss = outputs.loss
    perplexidade = torch.exp(loss)
    return perplexidade.item()
```

Exemplo de uso:
```python
prompt1 = "Descreva a estrutura e usos do ácido acético."
prompt2 = "Fale sobre o ácido acético."

perplexidade1 = calcular_perplexidade(prompt1)
perplexidade2 = calcular_perplexidade(prompt2)

print(f"Perplexidade do Prompt 1: {perplexidade1}")
print(f"Perplexidade do Prompt 2: {perplexidade2}")
```

Saída de exemplo (registrada no notebook):
- Perplexidade do Prompt 1: 228.6809
- Perplexidade do Prompt 2: 440.0884

## Interpretação
- Perplexidade mais baixa → modelo mais "confortável" com a sequência (menor incerteza).
- Prompts mais específicos e informativos tendem a gerar perplexidade menor; prompts curtos/genéricos tendem a ter perplexidade maior.

## Boas práticas e observações técnicas
- Uso de GPU: se disponível, mova tensores e modelo para `cuda` para acelerar:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  inputs = tokenizer(prompt, return_tensors="pt").to(device)
  ```
- Desligar gradientes e colocar o modelo em modo avaliação para medir perplexidade:
  ```python
  model.eval()
  with torch.no_grad():
      outputs = model(**inputs, labels=inputs['input_ids'])
  ```
- Reprodutibilidade:
  ```python
  torch.manual_seed(42)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(42)
  ```
- Textos longos: para textos com muitos tokens, calcule perplexidade por chunk (janelamento) e normalize por número de tokens para comparar corretamente.
- Normalização: para comparações justas entre textos de comprimentos diferentes, calcule perda média por token e então exponencie:
  - loss média por token = total_loss / total_tokens
  - perplexidade = exp(loss média por token)
- Performance / processamento em lote: use batching para processar múltiplos prompts eficientemente.

## Exemplo de script em lote (opcional)
Se desejar um script `compute_perplexity.py`, ele pode:
- Ler um CSV/JSON com prompts,
- Calcular perplexidade em lote (batching),
- Registrar resultados em CSV com colunas: prompt, perplexidade, n_tokens.

Posso gerar este script se quiser.

## Melhoria sugerida para o repositório
- Adicionar `requirements.txt` com versões fixas (ex.: transformers==4.xx, torch==x.y.z).
- Incluir `LICENSE` (ex.: MIT) e arquivo `CONTRIBUTING.md` se for colaborar.
- Incluir scripts auxiliares para execução em lote e testes.
- Registrar resultados/outputs de exemplo em `examples/` ou no próprio notebook.

## Licença
Adicione uma licença apropriada (por exemplo MIT). No momento, não há licença definida.

## Contato
Para dúvidas ou melhorias, abra uma issue no repositório ou envie um pull request.
