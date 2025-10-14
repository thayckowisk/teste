# 🌾 Simulador Inteligente de Produtividade Agrícola

## 📝 Descrição:

Sistema web de predição agrícola que combina **machine learning** e **análise econômica** para auxiliar na tomada de decisão no agronegócio. 

## 🎯 Necessidade do Cliente:

O cliente identificou uma demanda crescente no agronegócio por ferramentas que auxiliem produtores rurais, consultores técnicos e cooperativas na tomada de decisão sobre plantio e investimentos agrícolas. Atualmente, estes profissionais dependem de experiência empírica ou planilhas simples que não consideram a complexidade das variáveis agrícolas e econômicas envolvidas.

## 📊 Dados da Aplicação:

### **Dataset**
- **Fonte**: crop_yield.csv (1M registros agrícolas globais)
- **Utilizado**: 40.000 registros (amostra otimizada)
- **Divisão**: 80% treino / 20% teste
- **Variáveis**: 9 features + target (produtividade t/ha)

### **Tecnologia**
| Componente | Especificação |
|------------|---------------|
| **Algoritmo** | Random Forest (30 árvores) |
| **Performance** | R² > 0.85 |
| **Frontend** | Streamlit + Plotly |
| **Backend** | Python + scikit-learn |

### **Variáveis de Entrada**
- **Localização**: Região, tipo de solo, condições climáticas
- **Cultivo**: Cultura, precipitação, temperatura, ciclo
- **Manejo**: Fertilizantes, irrigação (impacto quantificado)

## 🖼️ Mockup do Projeto:

<!-- Inserir imagem do projeto aqui -->
*[Espaço reservado para screenshot/imagem da interface do simulador]*

## 📈 Funcionalidades:

### **🎯 Predições**
- Produtividade: 0.5 - 8.0 t/ha com percentil de performance
- Processamento em tempo real
- Confiabilidade baseada em validação estatística

### **💰 Análise Econômica**  
- Custos detalhados por cultura (R$ 2.000 - R$ 4.500/ha)
- Receitas com preços atualizados de commodities
- ROI automático e análise de viabilidade

### **📊 Interface**
- Métricas em tempo real com status visual
- Análise econômica detalhada (custos, receitas, ROI)
- Interface intuitiva

## 🚀 Execução:

```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8508
```

## 🏆 Solução Entregue:

O simulador desenvolvido atende precisamente às necessidades identificadas, utilizando **40.000 registros reais** do dataset crop_yield.csv para gerar com **precisão superior a 85%** (R² > 0.85). A interface, completamente intuitiva, integra análise técnica e econômica em uma ferramenta única que transforma dados complexos em decisões práticas e lucrativas para o setor agrícola.

---
*🏆 Solução completa para inteligência agrícola aplicada*