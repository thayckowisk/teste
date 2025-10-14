import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import os

# Configuração da página
st.set_page_config(
    page_title="🌾 Simulador Agrícola",
    page_icon="🌾",
    layout="centered"
)

# Mapeamentos unificados
MAPA = {
    'North': 'Norte', 'South': 'Sul', 'East': 'Leste', 'West': 'Oeste',
    'Clay': 'Argila', 'Sandy': 'Arenoso', 'Loamy': 'Franco', 'Silt': 'Silte',
    'Chalky': 'Calcário', 'Loam': 'Húmus', 'Peaty': 'Turfoso',
    'Rice': 'Arroz', 'Wheat': 'Trigo', 'Corn': 'Milho', 'Barley': 'Cevada', 
    'Soybeans': 'Soja', 'Cotton': 'Algodão', 'Soybean': 'Soja', 'Maize': 'Milho',
    'Sunny': 'Ensolarado', 'Rainy': 'Chuvoso', 'Cloudy': 'Nublado', 'Dry': 'Seco'
}
MAPA_INV = {v: k for k, v in MAPA.items()}

def traduzir(opcoes):
    return [MAPA.get(op, op) for op in opcoes]

def original(opcao):
    return MAPA_INV.get(opcao, opcao)

@st.cache_data
def carregar_dados():
    """Carrega dataset local crop_yield.csv com 40.000 registros"""
    
    if os.path.exists("crop_yield.csv"):
        try:
            df = pd.read_csv("crop_yield.csv")
            if len(df) > 40000:
                df = df.sample(n=40000, random_state=42)
            st.success(f"✅ Dataset LOCAL: {len(df):,} registros carregados")
            return df.dropna()
        except Exception as e:
            st.error(f"❌ Erro ao carregar dataset: {e}")
            st.stop()
    
    st.error("❌ Dataset crop_yield.csv não encontrado na pasta!")
    st.stop()

class ModeloML:
    def __init__(self, df):
        self.df = df
        self.model = RandomForestRegressor(n_estimators=30, max_depth=8, n_jobs=-1, random_state=42)
        self.encoders = {}
        self.trained = False
        
    def preparar(self):
        cat_cols = ['Region', 'Soil_Type', 'Crop', 'Weather_Condition']
        num_cols = ['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest']
        bool_cols = ['Fertilizer_Used', 'Irrigation_Used']
        
        X = self.df[cat_cols + num_cols + bool_cols].copy()
        y = self.df['Yield_tons_per_hectare']
        
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.encoders[col] = le
        
        for col in bool_cols:
            X[col] = X[col].astype(int)
        
        return X, y
    
    def treinar(self):
        X, y = self.preparar()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        self.trained = True
        return {'mae': mean_absolute_error(y_test, y_pred), 'r2': r2_score(y_test, y_pred)}
    
    def predizer(self, dados):
        if not self.trained:
            return {"error": "Modelo não treinado"}
        
        try:
            cats = ['Region', 'Soil_Type', 'Crop', 'Weather_Condition']
            for col in cats:
                if col in dados:
                    dados[col] = self.encoders[col].transform([dados[col]])[0]
            
            df_input = pd.DataFrame([dados])
            pred = self.model.predict(df_input)[0]
            
            all_preds = self.model.predict(self.preparar()[0])
            percentil = (np.sum(all_preds <= pred) / len(all_preds)) * 100
            
            return {'prediction': pred, 'percentile': percentil}
            
        except Exception as e:
            return {"error": f"Erro: {str(e)}"}

def main():
    """Função principal"""
    
    # Título principal
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #4CAF50, #8BC34A); border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0;">🌾 Simulador de Produtividade Agrícola</h1>
        <p style="color: white; margin: 0.5rem 0 0 0;">Predição Inteligente para o Agronegócio</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Carregar dados
    with st.spinner("🔄 Carregando dados..."):
        df = carregar_dados()
    
    # Treinar modelo
    if 'modelo_ml' not in st.session_state:
        with st.spinner("🤖 Treinando modelo..."):
            st.session_state.modelo_ml = ModeloML(df)
            metrics = st.session_state.modelo_ml.treinar()
            st.success(f"✅ Modelo treinado! Precisão R²: {metrics['r2']:.3f}")
    
    simulador = st.session_state.modelo_ml
    
    # Interface do simulador
    st.markdown("### 🎛️ Configure suas Condições de Cultivo")
    
    with st.form("form_simulador"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Região
            opcoes_regiao = traduzir(df['Region'].unique())
            region_selecionada = st.selectbox("🌍 Região:", opcoes_regiao)
            region = original(region_selecionada)
            
            # Solo
            opcoes_solo = traduzir(df['Soil_Type'].unique())
            solo_selecionado = st.selectbox("🌱 Solo:", opcoes_solo)
            soil_type = original(solo_selecionado)
            
            # Cultura
            opcoes_cultura = traduzir(df['Crop'].unique())
            cultura_selecionada = st.selectbox("🌾 Cultura:", opcoes_cultura)
            crop = original(cultura_selecionada)
            
            # Clima
            opcoes_clima = traduzir(df['Weather_Condition'].unique())
            clima_selecionado = st.selectbox("🌤️ Clima:", opcoes_clima)
            weather = original(clima_selecionado)
        
        with col2:
            # Precipitação
            rainfall = st.slider(
                "☔ Chuva (mm):",
                min_value=int(df['Rainfall_mm'].min()),
                max_value=int(df['Rainfall_mm'].max()),
                value=int(df['Rainfall_mm'].mean())
            )
            
            # Temperatura
            temperature = st.slider(
                "🌡️ Temperatura (°C):",
                min_value=int(df['Temperature_Celsius'].min()),
                max_value=int(df['Temperature_Celsius'].max()),
                value=int(df['Temperature_Celsius'].mean())
            )
            
            # Dias até colheita
            days_harvest = st.slider(
                "📅 Dias até Colheita:",
                min_value=int(df['Days_to_Harvest'].min()),
                max_value=int(df['Days_to_Harvest'].max()),
                value=int(df['Days_to_Harvest'].mean())
            )
            
            # Práticas
            fertilizer = st.checkbox("🧪 Fertilizante", value=True)
            irrigation = st.checkbox("💧 Irrigação", value=True)
        
        # Botão
        predict_button = st.form_submit_button("🔮 Simular Produtividade", type="primary")
    
    # Predição
    if predict_button:
        input_data = {
            'Region': region,
            'Soil_Type': soil_type,
            'Crop': crop,
            'Weather_Condition': weather,
            'Rainfall_mm': rainfall,
            'Temperature_Celsius': temperature,
            'Days_to_Harvest': days_harvest,
            'Fertilizer_Used': fertilizer,
            'Irrigation_Used': irrigation
        }
        
        resultado = simulador.predizer(input_data)
        
        if 'error' in resultado:
            st.error(f"❌ {resultado['error']}")
        else:
            prediction = resultado['prediction']
            percentile = resultado['percentile']
            
            # Status
            if percentile < 25:
                status = "🔴 Baixo"
                color = "#ff4444"
            elif percentile < 50:
                status = "⚠️ Moderado"
                color = "#ff8800"
            elif percentile < 75:
                status = "🟡 Bom"
                color = "#ffbb33"
            else:
                status = "🟢 Excelente"
                color = "#44ff44"
            
            # Resultado
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, {color}22, {color}44); border-radius: 10px; border: 2px solid {color}; margin: 1rem 0;">
                <h2 style="color: {color}; margin: 0;">🎯 Predição: {prediction:.2f} t/ha</h2>
                <h3 style="color: {color}; margin: 0.5rem 0;">{status} - Percentil {percentile:.0f}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Métricas
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Produtividade", f"{prediction:.2f} t/ha")
            
            with col2:
                media_cultura = df[df['Crop'] == crop]['Yield_tons_per_hectare'].mean()
                diferenca = prediction - media_cultura
                cultura_nome = MAPA.get(crop, crop)
                st.metric(f"Média {cultura_nome}", f"{media_cultura:.2f} t/ha", f"{diferenca:+.2f}")
            
            with col3:
                receita_estimada = prediction * 1000  # R$ 1000/tonelada
                st.metric("Receita/ha", f"R$ {receita_estimada:,.0f}")
            
            st.markdown("### 💰 Análise Econômica (por hectare)")
            
            custos = {
                'Rice': {'base': 2500, 'fert': 800, 'irrig': 600},
                'Wheat': {'base': 2200, 'fert': 700, 'irrig': 400},
                'Corn': {'base': 2800, 'fert': 900, 'irrig': 500},
                'Maize': {'base': 2800, 'fert': 900, 'irrig': 500},
                'Barley': {'base': 2000, 'fert': 600, 'irrig': 300},
                'Soybeans': {'base': 2600, 'fert': 500, 'irrig': 450},
                'Soybean': {'base': 2600, 'fert': 500, 'irrig': 450},
                'Cotton': {'base': 3500, 'fert': 1200, 'irrig': 800}
            }
            
            cultura_custo = custos.get(crop, custos['Corn'])
            base = cultura_custo['base']
            fert = cultura_custo['fert'] if fertilizer else 0
            irrig = cultura_custo['irrig'] if irrigation else 0
            total = base + fert + irrig
            
            precos = {
                'Rice': 1200, 'Wheat': 800, 'Corn': 600, 'Maize': 600,
                'Barley': 700, 'Soybeans': 1500, 'Soybean': 1500, 'Cotton': 3200
            }
            preco = precos.get(crop, 800)
            
            receita = prediction * preco
            lucro = receita - total
            roi = (lucro / total) * 100 if total > 0 else 0
            
            with st.expander("📊 Detalhamento", expanded=False):
                st.write(f"**Base**: R$ {base:,}")
                if fertilizer: st.write(f"**Fertilizantes**: R$ {fert:,}")
                if irrigation: st.write(f"**Irrigação**: R$ {irrig:,}")
                st.write(f"**Preço {MAPA.get(crop, crop)}**: R$ {preco:,}/ton")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"💰 **Receita**: R$ {receita:,.0f}")
                st.write(f"💸 **Custos Totais**: R$ {total:,}")
                st.write(f"💎 **Lucro**: R$ {lucro:,.0f}")
            
            with col2:
                st.write(f"📈 **ROI**: {roi:.1f}%")
                if roi > 30:
                    st.write("🎉 **Excelente retorno!**")
                elif roi > 15:
                    st.write("✅ **Bom retorno**")
                elif roi > 0:
                    st.write("⚠️ **Retorno moderado**")
                else:
                    st.write("❌ **Prejuízo estimado**")
                
                # Análise adicional
                if lucro > 0:
                    margem = (lucro / receita) * 100
                    st.write(f"📊 **Margem**: {margem:.1f}%")
                    
                    if margem > 25:
                        st.write("🟢 **Margem excelente**")
                    elif margem > 15:
                        st.write("🟡 **Margem boa**")
                    else:
                        st.write("🟠 **Margem apertada**")

if __name__ == "__main__":
    main()